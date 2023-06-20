import logging
from typing import Tuple, List, Iterable, Optional, Dict

import numpy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from kazu.utils.caching import kazu_disk_cache
from kazu.data.data import SynonymTermWithMetrics, SynonymTerm
from kazu.modelling.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
    NormalisedSynonymStr,
)
from kazu.modelling.language.string_similarity_scorers import BooleanStringSimilarityScorer
from kazu.modelling.ontology_preprocessing.base import (
    OntologyParser,
)
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import create_char_ngrams

logger = logging.getLogger(__name__)


class DictionaryIndex:
    """
    The dictionary index looks for SynonymTerms via a char ngram search between the normalised version of the
    query string and the term_norm of all SynonymTerms associated with the provided OntologyParser.
    """

    def __init__(
        self,
        parser: OntologyParser,
        boolean_scorers: Optional[List[BooleanStringSimilarityScorer]] = None,
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
        self.vectorizer, self.tf_idf_matrix = self.build_index_cache()
        self.synonym_list = list(self.synonyms_for_parser.values())

    def apply_boolean_scorers(self, reference_term: str, query_term: str) -> bool:

        if self.boolean_scorers is not None:
            return all(
                scorer(reference_term=reference_term, query_term=query_term)
                for scorer in self.boolean_scorers
            )
        else:
            return True

    def search(self, query: str, top_n: int = 15) -> Iterable[SynonymTermWithMetrics]:
        """Search the index with a query string.

        :param query: term to search
        :param top_n: max number of results
        :return:
        """

        match_norm = StringNormalizer.normalize(query, entity_class=self.entity_class)
        exact_match_term = self.synonyms_for_parser.get(match_norm)
        if exact_match_term is not None:
            term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                exact_match_term, search_score=100.0, bool_score=True, exact_match=True
            )
            yield term_with_metrics

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
                # get by index
                term = self.synonym_list[neighbour]
                if self.apply_boolean_scorers(reference_term=match_norm, query_term=term.term_norm):
                    term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                        term, search_score=score, bool_score=True, exact_match=False
                    )
                    yield term_with_metrics
                else:
                    logger.debug("filtered term %s as failed boolean checks", term)

    @kazu_disk_cache.memoize(ignore={0, 1})
    def _build_index_cache(
        self, synonyms_for_parser: Dict[NormalisedSynonymStr, SynonymTerm], _cache_key: int
    ) -> Tuple[TfidfVectorizer, numpy.ndarray]:
        logger.info("building TfidfVectorizer for %s", self.parser_name)
        vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
        tf_idf_matrix = vectorizer.fit_transform(synonyms_for_parser.keys())
        return vectorizer, tf_idf_matrix

    def build_index_cache(self) -> Tuple[TfidfVectorizer, numpy.ndarray]:
        """Build the cache for the index."""

        return self._build_index_cache(
            self.synonyms_for_parser, hash(frozenset(self.synonyms_for_parser.values()))
        )
