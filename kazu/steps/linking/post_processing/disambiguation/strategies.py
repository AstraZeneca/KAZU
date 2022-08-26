import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Optional, Set, Dict

import numpy as np

from kazu.data.data import (
    Document,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
)
from kazu.modelling.database.in_memory_db import MetadataDatabase, SynonymDatabase
from kazu.steps.linking.post_processing.disambiguation.context_scoring import TfIdfScorerManager

logger = logging.getLogger(__name__)


class DisambiguationStrategy(ABC):
    @abstractmethod
    def prepare(self, document: Document):
        pass

    @abstractmethod
    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        pass

    def __call__(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        self.prepare(document)
        return self.disambiguate(id_sets, document, parser_name)


class DefinedElsewhereInDocumentDisambiguationStrategy(DisambiguationStrategy):
    """
    1) look for entities on the document that have mappings
    2) see if any of these mappings correspond to ay ids in the EquivalentIdSets on each hit
    3) if only a single hit is found, create a new mapping from the matched hit
    4) if more than one hit is found, create multiple mappings, with the AMBIGUOUS flag
    """

    def __init__(
        self,
    ):
        self.found_equivalent_ids: Set[Tuple[str, str, str]] = set()

    def prepare(self, document: Document):
        """
        note, this method can't be cached, as the state of the document may change between executions
        :param document:
        :return:
        """
        self.found_equivalent_ids.clear()
        entities = document.get_entities()
        for ent in entities:
            for mapping in ent.mappings:
                self.found_equivalent_ids.add(
                    (
                        mapping.parser_name,
                        mapping.source,
                        mapping.idx,
                    )
                )

    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        found_id_sets = set()
        for id_set in id_sets:
            for idx in id_set.ids:
                if (
                    parser_name,
                    id_set.ids_to_source[idx],
                    idx,
                ) in self.found_equivalent_ids:
                    found_id_sets.add(id_set)
                    break
        return found_id_sets


class TfIdfDisambiguationStrategy(DisambiguationStrategy):
    """
    1) retrieve all synonyms associated with an equivalent ID set, and filter out ambiguous ones
        and build a query matrix with the unambiguous ones
    2) retrieve a list of all detected entity strings from the document, regardless of source and
        build a document representation matrix of these
    3) perform TFIDF on the query vs document, and sort according to most likely synonym hit from 1)
    4) if the score is above the minimum threshold, create a mapping

    """

    CONTEXT_SCORE = "context_score"

    def __init__(
        self,
        scorer_manager: TfIdfScorerManager,
        context_threshold: float = 0.7,
        aggregation_strategies_to_build_id_set_representation: Optional[
            List[EquivalentIdAggregationStrategy]
        ] = None,
    ):
        """

        :param path:
        :param id_filters:
        :param threshold: only consider terms above this search threshold
        :param differential: only consider terms with a search score within x of the best hit
        :param aggregation_strategies_to_build_id_set_representation:
        """
        self.context_threshold = context_threshold
        self.aggregation_strategies_to_build_id_set_representation: Set[
            EquivalentIdAggregationStrategy
        ] = (
            {EquivalentIdAggregationStrategy.UNAMBIGUOUS}
            if aggregation_strategies_to_build_id_set_representation is None
            else set(aggregation_strategies_to_build_id_set_representation)
        )
        self.synonym_db = SynonymDatabase()
        self.scorer_manager = scorer_manager

    def prepare(self, document: Document):
        pass

    def build_document_representation(self, parser_name: str, doc: Document) -> np.ndarray:
        strings = " ".join([x.match_norm for x in doc.get_entities()])
        return self.scorer_manager.parser_to_scorer[parser_name].transform(strings=[strings])

    def build_id_set_representation(
        self,
        parser_name: str,
        id_sets: Set[EquivalentIdSet],
    ) -> Dict[str, Set[EquivalentIdSet]]:
        # all normalised syns should be in the database
        result = defaultdict(set)
        for id_set in id_sets:
            for idx in id_set.ids:

                syns_this_id = self.synonym_db.get_syns_for_id(
                    parser_name,
                    idx,
                    self.aggregation_strategies_to_build_id_set_representation,
                )
                for syn in syns_this_id:
                    result[syn].add(id_set)
        return result

    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        scorer = self.scorer_manager.parser_to_scorer.get(parser_name)
        if scorer is None:
            return set()
        else:
            document_query_matrix = self.build_document_representation(parser_name, document)
            id_set_representation = self.build_id_set_representation(parser_name, id_sets)
            if len(id_set_representation) > 0:
                indexed_non_ambiguous_syns = list(id_set_representation.keys())
                for best_syn, score in scorer(indexed_non_ambiguous_syns, document_query_matrix):
                    if (
                        score >= self.context_threshold
                        and len(id_set_representation[best_syn]) == 1
                    ):
                        return id_set_representation[best_syn]
                else:
                    return set()
            else:
                return set()


class AnnotationLevelDisambiguationStrategy(DisambiguationStrategy):
    def prepare(self, document: Document):
        pass

    def __init__(self):
        self.metadata_db = MetadataDatabase()

    def disambiguate(
        self, id_sets: Set[EquivalentIdSet], document: Document, parser_name: str
    ) -> Set[EquivalentIdSet]:
        score_to_id_set = defaultdict(set)
        for id_set in id_sets:
            for idx in id_set.ids:
                score = self.metadata_db.get_by_idx(parser_name, idx).get("annotation_score", 0)
                score_to_id_set[score].add(id_set)
        best = max(score_to_id_set.keys())

        return score_to_id_set[best]
