import logging
import hashlib
from typing import Optional
from collections.abc import Iterable

import numpy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from kazu.utils.caching import kazu_disk_cache
from kazu.data import LinkingCandidate, LinkingMetrics
from kazu.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
    NormalisedSynonymStr,
)
from kazu.language.string_similarity_scorers import BooleanStringSimilarityScorer
from kazu.ontology_preprocessing.base import (
    OntologyParser,
)
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import create_char_ngrams

logger = logging.getLogger(__name__)


class DictionaryIndex:
    """The dictionary index looks for LinkingCandidates via a char ngram search between
    the normalised version of the query string and the synonym_norm of all
    LinkingCandidates associated with the provided OntologyParser."""

    def __init__(
        self,
        parser: OntologyParser,
        boolean_scorers: Optional[list[BooleanStringSimilarityScorer]] = None,
    ):
        """

        :param parser:
        :param boolean_scorers: precision can be increased by applying boolean checks on the returned result, for
            instance, checking that all integers are represented, that any noun modifiers are present etc
        """

        parser.populate_databases(force=False)
        self.synonym_db: SynonymDatabase = SynonymDatabase()
        self.synonyms_for_parser = self.synonym_db.get_all(parser.name)
        self.entity_class = parser.entity_class
        self.parser_name = parser.name
        self.metadata_db: MetadataDatabase = MetadataDatabase()
        self.namespace = self.__class__.__name__
        self.boolean_scorers = boolean_scorers
        self.normalized_synonyms: list[str] = []
        self.synonym_list: list[LinkingCandidate] = []
        # we sort here to ensure that the order of self.synonym_list matches the
        # order of the synonyms in the cached vectorized from _build_index_cache
        # if present, without relying on the order in the SynonymDatabase being the
        # same.
        for norm_syn, linking_candidate in sorted(self.synonyms_for_parser.items()):
            self.normalized_synonyms.append(norm_syn)
            self.synonym_list.append(linking_candidate)
        self.vectorizer, self.tf_idf_matrix = self.build_index_cache()

    def apply_boolean_scorers(self, reference_term: str, query_term: str) -> bool:

        if self.boolean_scorers is not None:
            return all(
                scorer(reference_term=reference_term, query_term=query_term)
                for scorer in self.boolean_scorers
            )
        else:
            return True

    def search(
        self, query: str, top_n: int = 15
    ) -> Iterable[tuple[LinkingCandidate, LinkingMetrics]]:
        """Search the index with a query string.

        .. note::
           This method will only return results with search scores above 0.
           As a result, it will return fewer than ``top_n`` results when there are not
           ``top_n`` :class:`~.LinkingCandidate`\\ s in the index that score about 0 for the given
           query.

        :param query: query string to search
        :param top_n: max number of results
        :return:
        """

        match_norm = StringNormalizer.normalize(query, entity_class=self.entity_class)
        exact_match_candidate = self.synonyms_for_parser.get(match_norm)
        if exact_match_candidate is not None:
            yield exact_match_candidate, LinkingMetrics(exact_match=True)

        else:
            # benchmarking suggests converting to dense is faster than using the
            # csr_matrix version. Strange...
            query_arr = self.vectorizer.transform([match_norm]).todense()
            # minus to negate, so arg sort works in correct order
            score_matrix = np.squeeze(-np.asarray(self.tf_idf_matrix.dot(query_arr.T)))
            neighbours = score_matrix.argsort()[:top_n]
            # don't use torch for this - it's slow
            # query = torch.FloatTensor(query)
            # score_matrix = self.tf_idf_matrix_torch.matmul(query.T)
            # score_matrix = torch.squeeze(score_matrix.T)
            # neighbours = torch.argsort(score_matrix, descending=True)[:top_n]

            distances = score_matrix[neighbours]
            distances = 100 * -distances
            for neighbour, score in zip(neighbours, distances):
                if score > 0.0:
                    # get by index
                    candidate = self.synonym_list[neighbour]
                    if self.apply_boolean_scorers(
                        reference_term=match_norm, query_term=candidate.synonym_norm
                    ):
                        yield candidate, LinkingMetrics(
                            exact_match=False, search_score=score, bool_score=True
                        )
                    else:
                        logger.debug("filtered candidate %s as failed boolean checks", candidate)
                else:
                    logger.debug("score is 0.0")

    @kazu_disk_cache.memoize(ignore={0, 1})
    def _build_index_cache(
        self, synonyms_for_parser: Iterable[NormalisedSynonymStr], _cache_key: bytes
    ) -> tuple[TfidfVectorizer, numpy.ndarray]:
        logger.info("building TfidfVectorizer for %s", self.parser_name)
        vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
        tf_idf_matrix = vectorizer.fit_transform(synonyms_for_parser)
        return vectorizer, tf_idf_matrix

    def build_index_cache(self) -> tuple[TfidfVectorizer, numpy.ndarray]:
        """Build the cache for the index."""

        h = hashlib.new("sha1", usedforsecurity=False)
        for norm_syn in self.normalized_synonyms:
            h.update(norm_syn.encode(encoding="utf-8"))

        return self._build_index_cache(self.normalized_synonyms, h.digest())
