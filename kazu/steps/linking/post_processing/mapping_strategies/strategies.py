import abc
import itertools
from abc import ABC
from collections.abc import Iterable
from typing import Optional

from kazu.data import (
    Document,
    Mapping,
    EquivalentIdSet,
    CandidatesToMetrics,
    StringMatchConfidence,
    DisambiguationConfidence,
)
from kazu.database.in_memory_db import MetadataDatabase, Metadata
from kazu.language.string_similarity_scorers import StringSimilarityScorer
from kazu.ontology_preprocessing.base import DEFAULT_LABEL
from kazu.steps.linking.post_processing.disambiguation.strategies import DisambiguationStrategy


class MappingFactory:
    """Factory class to produce mappings."""

    @staticmethod
    def create_mapping_from_id_sets(
        id_sets: set[EquivalentIdSet],
        parser_name: str,
        string_match_strategy: str,
        string_match_confidence: StringMatchConfidence,
        disambiguation_strategy: Optional[str],
        disambiguation_confidence: Optional[DisambiguationConfidence] = None,
        additional_metadata: Optional[dict] = None,
    ) -> Iterable[Mapping]:

        for id_set in id_sets:
            yield from MappingFactory.create_mapping_from_id_set(
                id_set=id_set,
                parser_name=parser_name,
                string_match_strategy=string_match_strategy,
                string_match_confidence=string_match_confidence,
                disambiguation_strategy=disambiguation_strategy,
                disambiguation_confidence=disambiguation_confidence,
                additional_metadata=additional_metadata,
            )

    @staticmethod
    def create_mapping_from_id_set(
        id_set: EquivalentIdSet,
        parser_name: str,
        string_match_strategy: str,
        string_match_confidence: StringMatchConfidence,
        disambiguation_strategy: Optional[str],
        disambiguation_confidence: Optional[DisambiguationConfidence] = None,
        additional_metadata: Optional[dict] = None,
    ) -> Iterable[Mapping]:
        for idx, source in id_set.ids_and_source:
            yield MappingFactory.create_mapping(
                parser_name=parser_name,
                source=source,
                idx=idx,
                string_match_strategy=string_match_strategy,
                string_match_confidence=string_match_confidence,
                disambiguation_strategy=disambiguation_strategy,
                disambiguation_confidence=disambiguation_confidence,
                additional_metadata=additional_metadata if additional_metadata is not None else {},
            )

    @staticmethod
    def _get_default_label_and_metadata_from_parser(
        parser_name: str, idx: str
    ) -> tuple[str, Metadata]:
        metadata = MetadataDatabase().get_by_idx(name=parser_name, idx=idx)
        default_label = metadata.pop(DEFAULT_LABEL)
        assert isinstance(default_label, str)
        return default_label, metadata

    @staticmethod
    def create_mapping(
        parser_name: str,
        source: str,
        idx: str,
        string_match_strategy: str,
        string_match_confidence: StringMatchConfidence,
        disambiguation_strategy: Optional[str] = None,
        disambiguation_confidence: Optional[DisambiguationConfidence] = None,
        additional_metadata: Optional[dict] = None,
        xref_source_parser_name: Optional[str] = None,
    ) -> Mapping:
        default_label, metadata = MappingFactory._get_default_label_and_metadata_from_parser(
            parser_name, idx
        )
        if additional_metadata:
            metadata.update(additional_metadata)
        return Mapping(
            default_label=default_label,
            idx=idx,
            source=source,
            string_match_strategy=string_match_strategy,
            string_match_confidence=string_match_confidence,
            disambiguation_strategy=disambiguation_strategy,
            disambiguation_confidence=disambiguation_confidence,
            parser_name=parser_name,
            metadata=metadata,
            xref_source_parser_name=xref_source_parser_name,
        )


class MappingStrategy(ABC):
    """A MappingStrategy is responsible for actualising instances of :class:`.Mapping`\\
    .

    This is performed in two steps:

    1. Filter the set of :data:`~.CandidatesToMetrics` associated with an :class:`.Entity` down
       to the most appropriate ones, (e.g. based on string similarity).

    2. If required, apply any configured :class:`.DisambiguationStrategy` to the filtered instances
       of :class:`.EquivalentIdSet`\\ .

    Selected instances of :class:`.EquivalentIdSet` are converted to :class:`.Mapping`\\ .
    """

    DISAMBIGUATION_NOT_REQUIRED = "disambiguation_not_required"

    def __init__(
        self,
        confidence: StringMatchConfidence,
        disambiguation_strategies: Optional[list[DisambiguationStrategy]] = None,
        disambiguation_essential: bool = False,
    ):
        """

        :param confidence: the level of confidence that should be assigned to this strategy. This is simply a label
            for human users, and has no bearing on the actual algorithm.
        :param disambiguation_strategies: after :meth:`filter_candidates` is called, these strategies are triggered if either
            multiple entries of :data:`~.CandidatesToMetrics` remain, and/or any of them are ambiguous.
        :param disambiguation_essential: disambiguation strategies MUST deliver a result, in order for this strategy to pass.
        """

        self.disambiguation_essential = disambiguation_essential
        self.confidence = confidence
        if disambiguation_essential and (
            disambiguation_strategies is None or len(disambiguation_strategies) == 0
        ):
            raise ValueError(
                "disambiguation strategies must be provided, as disambiguation_essential=True"
            )
        self.disambiguation_strategies = disambiguation_strategies

    def prepare(self, document: Document) -> None:
        """Perform any setup that needs to run once per document.

        Care should be taken if trying to cache this step, as the Document state is
        liable to change between executions.

        :param document:
        :return:
        """
        pass

    @abc.abstractmethod
    def filter_candidates(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        """Algorithms should override this method to return the "best"
        :data:`~.CandidatesToMetrics` for a given query string.

        Ideally, this will be a dict with a single element. However, it may not be possible to
        identify a single best match. In this scenario, the id sets of multiple
        :class:`.LinkingCandidate`\\ s will be carried forward for disambiguation
        (if configured).

        :param ent_match: the raw entity string.
        :param ent_match_norm: normalised version of the entity string.
        :param document: originating Document.
        :param candidates: candidates to filter.
        :param parser_name: parser name associated with these candidates.
        :return: a dict of filtered candidates
        """
        pass

    def disambiguate_if_required(
        self,
        filtered_candidates: CandidatesToMetrics,
        document: Document,
        parser_name: str,
        ent_match: str,
        ent_match_norm: str,
    ) -> tuple[set[EquivalentIdSet], Optional[str], Optional[DisambiguationConfidence]]:
        """Applies disambiguation strategies if configured, and either
        len(filtered_candidates) > 1 or any of the filtered_candidates are ambiguous. If
        ids are still ambiguous after all strategies have run, the disambiguation
        confidence will be :attr:`.DisambiguationConfidence.AMBIGUOUS`\\

        :param filtered_candidates: candidates to disambiguate
        :param document: originating Document
        :param parser_name: parser name associated with these candidates
        :param ent_match: string of entity to be disambiguated
        :param ent_match_norm: normalised string of entity to be disambiguated
        :return:
        """

        all_id_sets = set(
            id_set for candidates in filtered_candidates for id_set in candidates.associated_id_sets
        )

        if not self.disambiguation_essential and len(all_id_sets) == 1:
            # there's a single id set that isn't ambiguous, no need to disambiguate
            return all_id_sets, self.DISAMBIGUATION_NOT_REQUIRED, None
        elif not self.disambiguation_essential and (
            self.disambiguation_strategies is None or len(self.disambiguation_strategies) == 0
        ):
            return all_id_sets, None, DisambiguationConfidence.AMBIGUOUS
        else:
            assert self.disambiguation_strategies is not None
            for strategy in self.disambiguation_strategies:
                filtered_id_sets = strategy(
                    id_sets=all_id_sets,
                    document=document,
                    parser_name=parser_name,
                    ent_match=ent_match,
                    ent_match_norm=ent_match_norm,
                )
                if len(filtered_id_sets) == 1:
                    return filtered_id_sets, strategy.__class__.__name__, strategy.confidence
            if self.disambiguation_essential:
                return set(), None, DisambiguationConfidence.AMBIGUOUS
            else:
                return all_id_sets, None, DisambiguationConfidence.AMBIGUOUS

    def __call__(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
    ) -> Iterable[Mapping]:
        """

        :param ent_match: unnormalised NER string match (i.e. :attr:`.Entity.match`)
        :param ent_match_norm: normalised NER string match (i.e. :attr:`.Entity.match_norm`)
        :param document: originating document
        :param candidates: set of candidates to consider. Note, candidates from different parsers should not be mixed.
        :return:
        """
        parser_name = next(iter(candidates)).parser_name
        filtered_candidates = self.filter_candidates(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=document,
            candidates=candidates,
            parser_name=parser_name,
        )
        if filtered_candidates:
            (
                id_sets,
                successful_disambiguation_strategy,
                disambiguation_confidence,
            ) = self.disambiguate_if_required(
                filtered_candidates,
                document,
                parser_name,
                ent_match=ent_match,
                ent_match_norm=ent_match_norm,
            )
            yield from MappingFactory.create_mapping_from_id_sets(
                id_sets=id_sets,
                parser_name=parser_name,
                string_match_confidence=self.confidence,
                disambiguation_confidence=disambiguation_confidence,
                additional_metadata=None,
                string_match_strategy=self.__class__.__name__,
                disambiguation_strategy=successful_disambiguation_strategy,
            )


class ExactMatchMappingStrategy(MappingStrategy):
    """Returns any exact matches."""

    @staticmethod
    def filter_candidates(
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        return {k: v for k, v in candidates.items() if v.exact_match}


class SymbolMatchMappingStrategy(MappingStrategy):
    """Split both query and reference terms by whitespace.

    Select the term with the most splits as the 'query'. Check all of these tokens (and
    no more) are within the other term. Useful for symbol matching e.g. "MAP K8"
    (longest) vs "MAPK8" (shortest).
    """

    @staticmethod
    def match_symbols(s1: str, s2: str) -> bool:
        # the pattern should either be in both or neither
        reference_term_tokens = s1.split(" ")
        query_term_tokens = s2.split(" ")
        if len(reference_term_tokens) > len(query_term_tokens):
            longest = reference_term_tokens
            shortest = s2
        else:
            longest = query_term_tokens
            shortest = s1

        for tok in longest:
            if tok not in shortest:
                return False
            else:
                shortest = shortest.replace(tok, "", 1)

        return shortest.strip() == ""

    @classmethod
    def filter_candidates(
        cls,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        return {
            k: v for k, v in candidates.items() if cls.match_symbols(ent_match_norm, k.synonym_norm)
        }


class SynNormIsSubStringMappingStrategy(MappingStrategy):
    """For a :data:`~.CandidatesToMetrics`, see if any of their .synonym_norm are string
    matches of the match_norm tokens based on whitespace tokenisation.

    If exactly one element of :data:`~.CandidatesToMetrics` matches, prefer it.

    Works best on symbolic entities, e.g. "TESTIN gene" ->"TESTIN".
    """

    def __init__(
        self,
        confidence: StringMatchConfidence,
        disambiguation_strategies: Optional[list[DisambiguationStrategy]] = None,
        disambiguation_essential: bool = False,
        min_syn_norm_len_to_consider: int = 3,
    ):
        """

        :param confidence:
        :param disambiguation_strategies:
        :param disambiguation_essential:
        :param min_syn_norm_len_to_consider: only consider elements of
            :data:`~.CandidatesToMetrics` where the length of :attr:`~.LinkingCandidate.synonym_norm` is
            equal to or greater than this value.
        """
        super().__init__(
            confidence, disambiguation_strategies, disambiguation_essential=disambiguation_essential
        )
        self.min_syn_norm_len_to_consider = min_syn_norm_len_to_consider

    def filter_candidates(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        norm_tokens = set(ent_match_norm.split(" "))

        filtered_candidates_and_len = [
            (
                (
                    candidate,
                    metrics,
                ),
                len(candidate.synonym_norm),
            )
            for candidate, metrics in candidates.items()
            if candidate.synonym_norm in norm_tokens
            and len(candidate.synonym_norm) >= self.min_syn_norm_len_to_consider
        ]
        filtered_candidates_and_len.sort(key=lambda x: x[1], reverse=True)
        filtered_candidates_by_len_groups = itertools.groupby(
            filtered_candidates_and_len, key=lambda x: x[1]
        )
        for _, candidates_and_len_iter in filtered_candidates_by_len_groups:
            candidates_and_len_list = list(candidates_and_len_iter)
            if len(candidates_and_len_list) == 1:
                candidate, metrics = candidates_and_len_list[0][0]
                return {candidate: metrics}
        return {}


class StrongMatchMappingStrategy(MappingStrategy):
    """
    1. sort :data:`~.CandidatesToMetrics` by highest scoring search match to identify the
       highest scoring match.
    2. query remaining matches to see whether their scores are greater than this best score - the
       differential (i.e. there are many close string matches).
    """

    def __init__(
        self,
        confidence: StringMatchConfidence,
        disambiguation_strategies: Optional[list[DisambiguationStrategy]] = None,
        disambiguation_essential: bool = False,
        search_threshold: float = 80.0,
        symbolic_only: bool = False,
        differential: float = 2.0,
    ):
        """

        :param confidence:
        :param disambiguation_strategies:
        :param disambiguation_essential:
        :param search_threshold: only consider linking candidates above this search threshold
        :param symbolic_only: only consider candidates that are symbolic
        :param differential: only consider candidates with search scores equal or greater to the best match minus this value
        """
        super().__init__(
            confidence, disambiguation_strategies, disambiguation_essential=disambiguation_essential
        )
        self.differential = differential
        self.symbolic_only = symbolic_only
        self.search_threshold = search_threshold

    def filter_candidates(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        if self.symbolic_only:
            relevant_candidates_with_scores = [
                (
                    (
                        candidate,
                        metrics,
                    ),
                    score,
                )
                for candidate, metrics in candidates.items()
                if candidate.is_symbolic and (score := metrics.search_score) is not None
            ]
        else:
            relevant_candidates_with_scores = [
                (
                    (
                        candidate,
                        metrics,
                    ),
                    score,
                )
                for candidate, metrics in candidates.items()
                if (score := metrics.search_score) is not None
            ]

        if len(relevant_candidates_with_scores) == 0:
            return {}

        best_score = max((score for _, score in relevant_candidates_with_scores))

        return {
            candidates_and_metrics[0]: candidates_and_metrics[1]
            for candidates_and_metrics, score in relevant_candidates_with_scores
            if score >= self.search_threshold and best_score - score <= self.differential
        }


class StrongMatchWithEmbeddingConfirmationStringMatchingStrategy(StrongMatchMappingStrategy):
    """Same as parent class, but a complex string scorer with a predefined threshold is
    used to confirm that the ent_match is broadly similar to one of the candidates
    attached to the :data:`~.CandidatesToMetrics`\\ .

    Useful for refining non-symbolic close string matches (e.g. "Neck disease" and "Heck
    disease").
    """

    def __init__(
        self,
        confidence: StringMatchConfidence,
        complex_string_scorer: StringSimilarityScorer,
        disambiguation_strategies: Optional[list[DisambiguationStrategy]] = None,
        disambiguation_essential: bool = False,
        search_threshold: float = 80.0,
        embedding_threshold: float = 0.60,
        symbolic_only: bool = False,
        differential: float = 2.0,
    ):
        """
        :param confidence:
        :param complex_string_scorer: only consider linking candidates passing this string scorer call
        :param disambiguation_strategies:
        :param disambiguation_essential:
        :param search_threshold: only consider candidates above this search threshold
        :param embedding_threshold: the Entity.match and one of the LinkingCandidate.raw_synonyms must be
            above this threshold (according to the complex_string_scorer) for the candidate to be valid
        :param symbolic_only: only consider candidates that are symbolic
        :param differential: only consider candidates with search scores equal or greater to the best match minus this value
        """
        super().__init__(
            confidence=confidence,
            search_threshold=search_threshold,
            differential=differential,
            symbolic_only=symbolic_only,
            disambiguation_strategies=disambiguation_strategies,
            disambiguation_essential=disambiguation_essential,
        )
        self.embedding_threshold = embedding_threshold
        self.complex_string_scorer = complex_string_scorer

    def filter_candidates(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        candidates: CandidatesToMetrics,
        parser_name: str,
    ) -> CandidatesToMetrics:
        candidate_sorted_by_score = sorted(
            super()
            .filter_candidates(
                ent_match=ent_match,
                ent_match_norm=ent_match_norm,
                document=document,
                candidates=candidates,
                parser_name=parser_name,
            )
            .items(),
            key=lambda x: x[1].search_score,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        selected_id_sets = set()
        selected_candidates = {}
        for candidate, metrics in candidate_sorted_by_score:
            if candidate.associated_id_sets not in selected_id_sets:
                selected_id_sets.add(candidate.associated_id_sets)
                if any(
                    self.complex_string_scorer(ent_match, original_syn) >= self.embedding_threshold
                    for original_syn in candidate.raw_synonyms
                ):
                    selected_candidates[candidate] = metrics
        return selected_candidates
