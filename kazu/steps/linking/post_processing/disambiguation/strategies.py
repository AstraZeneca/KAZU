import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from os import getenv
from typing import Optional

import numpy as np
from kazu.data import (
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
from kazu.language.string_similarity_scorers import StringSimilarityScorer
from kazu.ontology_preprocessing.base import DEFAULT_LABEL
from kazu.steps.linking.post_processing.disambiguation.context_scoring import (
    TfIdfScorer,
    GildaTfIdfScorer,
)
from kazu.utils.grouping import sort_then_group
from kazu.utils.string_normalizer import StringNormalizer
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class DisambiguationStrategy(ABC):
    """The job of a DisambiguationStrategy is to produce a Set of
    :class:`.EquivalentIdSet`\\.

    .. warning::
       The :class:`.EquivalentIdSet`\\s produced needn't map to those contained within
       :attr:`~.LinkingCandidate.associated_id_sets`\\. This may cause confusing behaviour
       during debugging.


    A :meth:`prepare` method is available, which can be cached in the
    event of any duplicated preprocessing work that may be required (see
    :class:`.StrategyRunner` for the complexities of how MappingStrategy
    and DisambiguationStrategy are coordinated).
    """

    def __init__(self, confidence: DisambiguationConfidence):
        """

        :param confidence: the level of confidence that should be assigned to this strategy. This is simply a label
            for human users, and has no bearing on the actual algorithm.
        """
        self.confidence = confidence

    @abstractmethod
    def prepare(self, document: Document) -> None:
        """Perform any preprocessing required.

        :param document:
        :return:
        """
        pass

    @abstractmethod
    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
        """Select a subset of :class:`.EquivalentIdSet`\\.

        :param id_sets: disambiguation result should be based on these id_sets -
            either a standard subset, or subset based on modified :class:`.EquivalentIdSet`
        :param document: source document
        :param parser_name: name of parser that the id_set comes from
        :param ent_match: matched entity string
        :param ent_match_norm: normalised version of entity string
        :return:
        """
        pass

    def __call__(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
        self.prepare(document)
        return self.disambiguate(id_sets, document, parser_name, ent_match, ent_match_norm)


class DefinedElsewhereInDocumentDisambiguationStrategy(DisambiguationStrategy):
    """
    1. look for entities on the document that have mappings
    2. filter the incoming set of :class:`.EquivalentIdSet` based on these mappings
    """

    def __init__(self, confidence: DisambiguationConfidence):
        super().__init__(confidence)
        self.mapped_ids: set[tuple[str, str, str]] = set()

    def prepare(self, document: Document) -> None:
        """Note, this method can't be cached, as the state of the document may change
        between executions.

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
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
        found_id_sets = set()
        for id_set in id_sets:
            filtered_equivalent_id_set_items = set()
            for idx, source in id_set.ids_and_source:
                if (
                    parser_name,
                    source,
                    idx,
                ) in self.mapped_ids:
                    filtered_equivalent_id_set_items.add((idx, source))
            if len(filtered_equivalent_id_set_items) > 0:
                found_id_sets.add(EquivalentIdSet(frozenset(filtered_equivalent_id_set_items)))
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

    def __init__(
        self,
        confidence: DisambiguationConfidence,
        scorer: TfIdfScorer,
        context_threshold: float = 0.7,
        relevant_aggregation_strategies: Optional[Iterable[EquivalentIdAggregationStrategy]] = None,
    ):
        """

        :param confidence: the level of confidence that should be assigned to this strategy. This is simply a label
            for human users, and has no bearing on the actual algorithm.
        :param scorer: handles scoring of contexts
        :param context_threshold: only consider synonyms above this search threshold
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
        self.parser_name_to_doc_representation: dict[str, np.ndarray] = {}

    @functools.lru_cache(maxsize=int(getenv("KAZU_TFIDF_DISAMBIGUATION_DOCUMENT_CACHE_SIZE", 1)))
    def prepare(self, document: Document) -> None:
        """Build document representations by parser names here, and store in a dict.
        This method is cached so we don't need to call it multiple times per document.

        :param document:
        :return:
        """
        parser_names = frozenset(
            candidate.parser_name
            for ent in document.get_entities()
            for candidate in ent.linking_candidates
        )
        self.parser_name_to_doc_representation = self.cacheable_build_document_representation(
            scorer=self.scorer, doc=document, parsers=parser_names
        )

    @staticmethod
    @functools.lru_cache(maxsize=int(getenv("KAZU_TFIDF_DISAMBIGUATION_CACHE_SIZE", 20)))
    def cacheable_build_document_representation(
        scorer: TfIdfScorer, doc: Document, parsers: frozenset[str]
    ) -> dict[str, csr_matrix]:
        """Static cached method, so we don't need to recalculate document representation
        between different instances of this class.

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
        id_sets: set[EquivalentIdSet],
    ) -> dict[NormalisedSynonymStr, set[EquivalentIdSet]]:
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
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
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


class GildaTfIdfDisambiguationStrategy(DisambiguationStrategy):
    def __init__(
        self,
        confidence: DisambiguationConfidence,
        scorer: GildaTfIdfScorer,
        context_threshold_delta: float = 0.01,
    ):
        """

        :param confidence:
        :param scorer:
        :param context_threshold_delta: If the maximum delta between the top two :class:`.EquivalentIdSet`\\ s is below this
            value, assume disambiguation has failed
        """
        super().__init__(confidence)
        self.context_threshold_delta = context_threshold_delta
        self.scorer = scorer

    @functools.lru_cache(maxsize=int(getenv("KAZU_TFIDF_DISAMBIGUATION_DOCUMENT_CACHE_SIZE", 1)))
    def prepare(self, document: Document) -> None:
        """Build document representations by parser names here, and store in a dict.

        This method is cached so we don't need to call it multiple times per document.

        :param document:
        :return:
        """
        self.doc_vector = self.cacheable_build_document_representation(
            scorer=self.scorer, doc=document
        )

    @staticmethod
    @functools.lru_cache(maxsize=int(getenv("KAZU_TFIDF_DISAMBIGUATION_CACHE_SIZE", 20)))
    def cacheable_build_document_representation(
        scorer: GildaTfIdfScorer, doc: Document
    ) -> csr_matrix:
        """Static cached method, so we don't need to recalculate document representation
        between different instances of this class.

        :param scorer:
        :param doc:
        :return:
        """
        doc_string = " ".join(x.text for x in doc.sections)
        return scorer.vectorizer.transform([doc_string])

    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:

        idx_to_set: defaultdict[str, set[EquivalentIdSet]] = defaultdict(set)
        for id_set in id_sets:
            for idx in id_set.ids:
                idx_to_set[idx].add(id_set)

        best_score = 0.0
        best_set: Optional[set[EquivalentIdSet]] = None
        for idx, score in self.scorer(
            context_vec=self.doc_vector,
            parser_name=parser_name,
            id_sets=id_sets,
        ):
            # Note: we know that len(id_sets)>1 as it's a disambiguation strategy, and
            # it won't be called if len(id_sets)==1
            this_set = idx_to_set[idx]
            if best_set is None:
                best_set = this_set
                best_score = score
            elif best_set is this_set:
                # a lower scoring ID is associated with the same set[EquivalentIDSet]
                continue
            else:
                # we've gotten to the first ID that's actually different
                # to the one with the best score
                if (best_score - score) < self.context_threshold_delta:
                    # delta is insufficient, consider disambiguation to have failed
                    return set()
                else:
                    # the first ID's score is sufficiently above all others
                    return best_set
        # Code should never reach this, but it's required anyway
        return set()


class AnnotationLevelDisambiguationStrategy(DisambiguationStrategy):
    """Certain entities are often mentioned by some colloquial name, even if it's
    technically incorrect.

    In these cases, we may have an annotation_score field in the metadata_db, as a proxy
    of how widely studied the entity is. We use this annotation score as a proxy for
    'given a random mention of the entity, how likely is it that the author is referring
    to instance x vs instance y'. Naturally, this is a pretty unsophisticated
    disambiguation strategy, so should generally only be used as a last resort!
    """

    def prepare(self, document: Document) -> None:
        pass

    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:
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


class PreferDefaultLabelMatchDisambiguationStrategy(DisambiguationStrategy):
    """Prefer ids where the entity match string is the default label (after
    normalisation).

    .. note::
       This strategy is intended to be used with
       :class:`kazu.steps.linking.post_processing.mapping_strategies.strategies.ExactMatchMappingStrategy`
       with the ``disambiguation_essential`` argument
       set to ``True``.
    """

    def __init__(self, confidence: DisambiguationConfidence):
        super().__init__(confidence)
        self.metadata_db = MetadataDatabase()

    def prepare(self, document: Document) -> None:
        pass

    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:

        entity_class = self.metadata_db.parser_name_to_ent_class[parser_name]
        disambiguated_id_set = set()
        for equiv_id_set in id_sets:
            for idx, source in equiv_id_set.ids_and_source:
                default_label_norm = StringNormalizer.normalize(
                    self.metadata_db.get_by_idx(idx=idx, name=parser_name)[DEFAULT_LABEL],
                    entity_class=entity_class,
                )
                if default_label_norm == ent_match_norm:
                    disambiguated_id_set.add(
                        (
                            idx,
                            source,
                        )
                    )

        if len(disambiguated_id_set) == 0:
            return set()
        else:
            return {EquivalentIdSet(ids_and_source=frozenset(disambiguated_id_set))}


class PreferNearestEmbeddingToDefaultLabelDisambiguationStrategy(DisambiguationStrategy):
    """Prefer ids where the entity match string is nearest to the default label (as per
    the configured :class:`.StringSimilarityScorer`\\).

    In the case where multiple ID's share the same nearest embedding distance, multiple
    IDs will be returned. This can happen if there are two ids that share the same
    default label.
    """

    def __init__(
        self, complex_string_scorer: StringSimilarityScorer, confidence: DisambiguationConfidence
    ):
        super().__init__(confidence)
        self.complex_string_scorer = complex_string_scorer
        self.metadata_db = MetadataDatabase()

    def prepare(self, document: Document) -> None:
        pass

    def disambiguate(
        self,
        id_sets: set[EquivalentIdSet],
        document: Document,
        parser_name: str,
        ent_match: Optional[str] = None,
        ent_match_norm: Optional[str] = None,
    ) -> set[EquivalentIdSet]:

        assert (
            ent_match is not None
        ), "ent_match is None, but this strategy requires it to function."
        idx_and_scores: set[tuple[tuple[str, str], float]] = set()
        for equiv_id_set in id_sets:

            for idx, source in equiv_id_set.ids_and_source:
                default_label = self.metadata_db.get_by_idx(idx=idx, name=parser_name)[
                    DEFAULT_LABEL
                ]
                score = self.complex_string_scorer(ent_match, default_label)
                idx_and_scores.add(
                    (
                        (
                            idx,
                            source,
                        ),
                        score,
                    )
                )

        result: set[EquivalentIdSet] = set()
        for score, items in sort_then_group(
            idx_and_scores, key_func=lambda x: x[1], reverse=True
        ):  # negative operand for highest score first
            # return on first, as we're only interested in top hit
            result.add(EquivalentIdSet(ids_and_source=frozenset(item[0] for item in items)))
            # we only want the top score
            break

        return result
