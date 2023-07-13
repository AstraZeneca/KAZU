import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from os import getenv
from typing import Tuple, Optional, Set, Dict, Iterable, FrozenSet

import numpy as np

from kazu.data.data import (
    Document,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    DisambiguationConfidence,
)
from kazu.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
    NormalisedSynonymStr,
)
from kazu.steps.linking.post_processing.disambiguation.context_scoring import TfIdfScorer

logger = logging.getLogger(__name__)


class DisambiguationStrategy(ABC):
    """
    The job of a DisambiguationStrategy is to filter a Set of :class:`.EquivalentIdSet` into a
    (hopefully) smaller set.

    A :meth:`prepare` method is available, which can be cached in the event of any duplicated
    preprocessing work that may be required (see :class:`.StrategyRunner` for the complexities of
    how MappingStrategy and DisambiguationStrategy are coordinated).
    """

    def __init__(self, confidence: DisambiguationConfidence):
        """

        :param confidence: the level of confidence that should be assigned to this strategy. This is simply a label
            for human users, and has no bearing on the actual algorithm.
        """
        self.confidence = confidence

    @abstractmethod
    def prepare(self, document: Document):
        """
        perform any preprocessing required

        :param document:
        :return:
        """
        pass

    @abstractmethod
    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        """
        subset a set of :class:`.EquivalentIdSet`\\ .

        :param id_sets:
        :param document:
        :param parser_name:
        :return:
        """
        pass

    def __call__(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        self.prepare(document)
        return self.disambiguate(id_sets, document, parser_name)


class DefinedElsewhereInDocumentDisambiguationStrategy(DisambiguationStrategy):
    """
    1. look for entities on the document that have mappings
    2. see if any of these mappings correspond to any ids in the :class:`.EquivalentIdSet` on each
       hit
    """

    def __init__(self, confidence: DisambiguationConfidence):
        super().__init__(confidence)
        self.mapped_ids: Set[Tuple[str, str, str]] = set()

    def prepare(self, document: Document):
        """
        note, this method can't be cached, as the state of the document may change between executions

        :param document:
        :return:
        """
        self.mapped_ids = set()
        entities = document.get_entities()
        self.mapped_ids.update(
            (
                mapping.parser_name,
                mapping.source,
                mapping.idx,
            )
            for ent in entities
            for mapping in ent.mappings
        )

    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        found_id_sets = set()
        for id_set in id_sets:
            for idx, source in id_set.ids_and_source:
                if (
                    parser_name,
                    source,
                    idx,
                ) in self.mapped_ids:
                    found_id_sets.add(id_set)
                    break
        return found_id_sets


class TfIdfDisambiguationStrategy(DisambiguationStrategy):
    """
    1. retrieve all synonyms associated with a :class:`.EquivalentIdSet`, filter out ambiguous ones
       and build a query matrix with the unambiguous ones.
    2. retrieve a list of all detected entity strings from the document, regardless of source and
       build a document representation matrix of these.
    3. perform TFIDF on the query vs document, and sort according to most likely synonym hit from 1.
    4. if the score is above the minimum threshold, create a mapping.

    """

    CONTEXT_SCORE = "context_score"

    def __init__(
        self,
        confidence: DisambiguationConfidence,
        scorer: TfIdfScorer,
        context_threshold: float = 0.7,
        relevant_aggregation_strategies: Optional[Iterable[EquivalentIdAggregationStrategy]] = None,
    ):
        """

        :param scorer: handles scoring of contexts
        :param context_threshold: only consider terms above this search threshold
        :param relevant_aggregation_strategies: Only consider these strategies when selecting synonyms from the
            synonym database, when building a representation. If none, all strategies will be considered
        """
        super().__init__(confidence)
        self.context_threshold = context_threshold
        if relevant_aggregation_strategies is None:
            self.relevant_aggregation_strategies = {EquivalentIdAggregationStrategy.UNAMBIGUOUS}
        else:
            self.relevant_aggregation_strategies = set(relevant_aggregation_strategies)
        self.synonym_db = SynonymDatabase()
        self.scorer = scorer
        self.parser_name_to_doc_representation: Dict[str, np.ndarray] = {}

    @functools.lru_cache(maxsize=int(getenv("KAZU_TFIDF_DISAMBIGUATION_DOCUMENT_CACHE_SIZE", 1)))
    def prepare(self, document: Document):
        """
        build document representations by parser names here, and store in a dict. This method is cached so
        we don't need to call it multiple times per document

        :param document:
        :return:
        """
        parser_names = frozenset(
            term.parser_name
            for ent in document.get_entities()
            for term in ent.syn_term_to_synonym_terms
        )
        self.parser_name_to_doc_representation = self.cacheable_build_document_representation(
            scorer=self.scorer, doc=document, parsers=parser_names
        )

    @staticmethod
    @functools.lru_cache(maxsize=int(getenv("KAZU_TFIDF_DISAMBIGUATION_CACHE_SIZE", 20)))
    def cacheable_build_document_representation(
        scorer: TfIdfScorer, doc: Document, parsers: FrozenSet[str]
    ) -> Dict[str, np.ndarray]:
        """
        static cached method, so we don't need to recalculate document representation between different instances
        of this class.

        :param scorer:
        :param doc:
        :param parsers: technically this only has to be a hashable iterable of string - but it should also be
            unique otherwise duplicate work will be done and thrown away, so pragmatically a frozenset makes sense.
        :return:
        """
        strings = " ".join(x.match_norm for x in doc.get_entities())
        res = {}
        for parser in parsers:
            vectorizer = scorer.parser_to_vectorizer[parser]
            res[parser] = vectorizer.transform([strings])
        return res

    def build_id_set_representation(
        self,
        parser_name: str,
        id_sets: Set[EquivalentIdSet],
    ) -> Dict[NormalisedSynonymStr, Set[EquivalentIdSet]]:
        result = defaultdict(set)
        for id_set in id_sets:
            for idx in id_set.ids:

                syns_this_id = self.synonym_db.get_syns_for_id(
                    parser_name,
                    idx,
                    self.relevant_aggregation_strategies,
                )
                for syn in syns_this_id:
                    result[syn].add(id_set)
        return result

    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        if parser_name not in self.scorer.parser_to_vectorizer:
            return set()

        document_query_matrix = self.parser_name_to_doc_representation[parser_name]
        id_set_representation = self.build_id_set_representation(parser_name, id_sets)
        if len(id_set_representation) == 0:
            return set()
        else:
            indexed_non_ambiguous_syns = list(id_set_representation.keys())
            for best_syn, score in self.scorer(
                strings=indexed_non_ambiguous_syns,
                matrix=document_query_matrix,
                parser=parser_name,
            ):
                if score >= self.context_threshold and len(id_set_representation[best_syn]) == 1:
                    return id_set_representation[best_syn]
            return set()


class AnnotationLevelDisambiguationStrategy(DisambiguationStrategy):
    """
    certain entities are often mentioned by some colloquial name, even if it's technically incorrect. In these cases,
    we may have an annotation_score field in the metadata_db, as a proxy of how widely studied the entity is. We
    use this annotation score as a proxy for 'given a random mention of the entity, how likely is it that the
    author is referring to instance x vs instance y'. Naturally, this is a pretty unsophisticated disambiguation
    strategy, so should generally only be used as a last resort!
    """

    def prepare(self, document: Document):
        pass

    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        best_score = 0
        best_equiv_id_sets = set()

        for id_set in id_sets:
            for idx in id_set.ids:
                score = int(
                    MetadataDatabase().get_by_idx(parser_name, idx).get("annotation_score", 0)
                )
                if score > best_score:
                    best_score = score
                    best_equiv_id_sets = {id_set}
                elif score == best_score:
                    best_equiv_id_sets.add(id_set)

        return best_equiv_id_sets
