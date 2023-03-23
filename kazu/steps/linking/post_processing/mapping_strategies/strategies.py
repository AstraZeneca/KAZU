import abc
import itertools
import urllib
from abc import ABC
from typing import List, Optional, Set, Iterable, Dict, FrozenSet, Tuple

from kazu.data.data import (
    Document,
    Mapping,
    EquivalentIdSet,
    SynonymTermWithMetrics,
    StringMatchConfidence,
    DisambiguationConfidence,
)
from kazu.modelling.database.in_memory_db import MetadataDatabase, Metadata
from kazu.modelling.language.string_similarity_scorers import StringSimilarityScorer
from kazu.modelling.ontology_preprocessing.base import DEFAULT_LABEL
from kazu.steps.linking.post_processing.disambiguation.strategies import DisambiguationStrategy


class MappingFactory:
    """
    factory class to produce mappings
    """

    @staticmethod
    def create_mapping_from_id_sets(
        id_sets: Set[EquivalentIdSet],
        parser_name: str,
        string_match_strategy: str,
        string_match_confidence: StringMatchConfidence,
        disambiguation_strategy: Optional[str],
        disambiguation_confidence: Optional[DisambiguationConfidence] = None,
        additional_metadata: Optional[Dict] = None,
        strip_url: bool = True,
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
                strip_url=strip_url,
            )

    @staticmethod
    def create_mapping_from_id_set(
        id_set: EquivalentIdSet,
        parser_name: str,
        string_match_strategy: str,
        string_match_confidence: StringMatchConfidence,
        disambiguation_strategy: Optional[str],
        disambiguation_confidence: Optional[DisambiguationConfidence] = None,
        additional_metadata: Optional[Dict] = None,
        strip_url: bool = True,
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
                strip_url=strip_url,
            )

    @staticmethod
    def _get_default_label_and_metadata_from_parser(
        parser_name: str, idx: str
    ) -> Tuple[str, Metadata]:
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
        additional_metadata: Optional[Dict] = None,
        strip_url: bool = True,
        xref_source_parser_name: Optional[str] = None,
    ) -> Mapping:
        default_label, metadata = MappingFactory._get_default_label_and_metadata_from_parser(
            parser_name, idx
        )
        if additional_metadata:
            metadata.update(additional_metadata)
        if strip_url:
            new_idx = MappingFactory._strip_url(idx)
        else:
            new_idx = idx
        return Mapping(
            default_label=default_label,
            idx=new_idx,
            source=source,
            string_match_strategy=string_match_strategy,
            string_match_confidence=string_match_confidence,
            disambiguation_strategy=disambiguation_strategy,
            disambiguation_confidence=disambiguation_confidence,
            parser_name=parser_name,
            metadata=metadata,
            xref_source_parser_name=xref_source_parser_name,
        )

    @staticmethod
    def _strip_url(idx):
        url = urllib.parse.urlparse(idx)
        if url.scheme == "":
            # not a url
            new_idx = idx
        else:
            new_idx = url.path.split("/")[-1]
        return new_idx


class MappingStrategy(ABC):
    """
    A MappingStrategy is responsible for actualising instances of :class:`.Mapping`\\ .

    This is performed in two steps:

    1. Filter the set of :class:`.SynonymTermWithMetrics` associated with an :class:`.Entity` down
       to the most appropriate ones, (e.g. based on string similarity).

    2. If required, apply any configured :class:`.DisambiguationStrategy` to the filtered instances
       of :class:`.EquivalentIdSet`\\ .

    Selected instances of :class:`.EquivalentIdSet` are converted to :class:`.Mapping`\\ .
    """

    DISAMBIGUATION_NOT_REQUIRED = "disambiguation_not_required"

    def __init__(
        self,
        confidence: StringMatchConfidence,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
    ):
        """

        :param confidence: the level of confidence that should be assigned to this strategy. This is simply a label
            for human users, and has no bearing on the actual algorithm.
        :param disambiguation_strategies: after :meth:`filter_terms` is called, these strategies are triggered if either
            multiple instances of :class:`.SynonymTermWithMetrics` remain, and/or any of them are ambiguous.
        """

        self.confidence = confidence
        self.disambiguation_strategies = disambiguation_strategies

    def prepare(self, document: Document):
        """
        perform any setup that needs to run once per document.

        Care should be taken if trying to cache this step, as the Document state is liable to
        change between executions.

        :param document:
        :return:
        """
        pass

    @abc.abstractmethod
    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        """
        Algorithms should override this method to return a set of the "best"
        :class:`.SynonymTermWithMetrics` for a given query string.

        Ideally, this will be a set with a single element. However, it may not be possible to
        identify a single best match. In this scenario, the id sets of multiple
        :class:`.SynonymTermWithMetrics` will be carried forward for disambiguation
        (if configured).

        :param ent_match: the raw entity string.
        :param ent_match_norm: normalised version of the entity string.
        :param document: originating Document.
        :param terms: terms to filter.
        :param parser_name: parser name associated with these terms.
        :return: a set of filtered terms
        """
        raise NotImplementedError()

    def disambiguate_if_required(
        self, filtered_terms: Set[SynonymTermWithMetrics], document: Document, parser_name: str
    ) -> Tuple[Set[EquivalentIdSet], Optional[str], Optional[DisambiguationConfidence]]:
        """
        applies disambiguation strategies if configured, and either len(filtered_terms) > 1 or any
        of the filtered_terms are ambiguous. If ids are still ambiguous after all strategies have run,
        the disambiguation confidence will be :attr:`.DisambiguationConfidence.AMBIGUOUS`\\

        :param filtered_terms: terms to disambiguate
        :param document: originating Document
        :param parser_name: parser name associated with these terms
        :return:
        """

        all_id_sets = set(id_set for term in filtered_terms for id_set in term.associated_id_sets)

        if len(all_id_sets) == 1:
            # there's a single id set that isn't ambiguous, no need to disambiguate
            return all_id_sets, self.DISAMBIGUATION_NOT_REQUIRED, None
        elif self.disambiguation_strategies is None:
            return all_id_sets, None, DisambiguationConfidence.AMBIGUOUS
        else:
            for strategy in self.disambiguation_strategies:
                filtered_id_sets = strategy(
                    id_sets=all_id_sets, document=document, parser_name=parser_name
                )
                if len(filtered_id_sets) == 1:
                    return filtered_id_sets, strategy.__class__.__name__, strategy.confidence

            return all_id_sets, None, DisambiguationConfidence.AMBIGUOUS

    def __call__(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
    ) -> Iterable[Mapping]:
        """

        :param ent_match: unnormalised NER string match (i.e. :attr:`.Entity.match`)
        :param ent_match_norm: normalised NER string match (i.e. :attr:`.Entity.match_norm`)
        :param document: originating document
        :param terms: set of terms to consider. Note, terms from different parsers should not be mixed.
        :return:
        """
        parser_name = next(iter(terms)).parser_name

        filtered_terms = self.filter_terms(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=document,
            terms=terms,
            parser_name=parser_name,
        )

        (
            id_sets,
            successful_disambiguation_strategy,
            disambiguation_confidence,
        ) = self.disambiguate_if_required(filtered_terms, document, parser_name)
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
    """
    returns any exact matches
    """

    @staticmethod
    def filter_terms(
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        return set(term for term in terms if term.exact_match)


class SymbolMatchMappingStrategy(MappingStrategy):
    """
    split both query and reference terms by whitespace. select the term with the most splits as the 'query'. Check
    all of these tokens (and no more) are within the other term. Useful for symbol matching
    e.g. "MAP K8" (longest) vs "MAPK8" (shortest)
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
    def filter_terms(
        cls,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        return set(term for term in terms if cls.match_symbols(ent_match_norm, term.term_norm))


class TermNormIsSubStringMappingStrategy(MappingStrategy):
    """
    for a set of :class:`.SynonymTermWithMetrics`, see if any of their .term_norm
    are string matches of the match_norm tokens based on whitespace tokenisation.

    If exactly one :class:`.SynonymTermWithMetrics` matches, prefer it.

    Works best on symbolic entities, e.g. "TESTIN gene" ->"TESTIN".
    """

    def __init__(
        self,
        confidence: StringMatchConfidence,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
        min_term_norm_len_to_consider: int = 3,
    ):
        """

        :param confidence:
        :param disambiguation_strategies:
        :param min_term_norm_len_to_consider: only consider instances of
            :class:`.SynonymTermWithMetrics` where the length of :attr:`~.SynonymTerm.term_norm` is
            equal to or greater than this value.
        """
        super().__init__(confidence, disambiguation_strategies)
        self.min_term_norm_len_to_consider = min_term_norm_len_to_consider

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        norm_tokens = set(ent_match_norm.split(" "))

        filtered_terms_and_len = [
            (
                term,
                len(term.term_norm),
            )
            for term in terms
            if term.term_norm in norm_tokens
            and len(term.term_norm) >= self.min_term_norm_len_to_consider
        ]
        filtered_terms_and_len.sort(key=lambda x: x[1], reverse=True)
        filtered_terms_by_len_groups = itertools.groupby(filtered_terms_and_len, key=lambda x: x[1])
        for _, terms_and_len_iter in filtered_terms_by_len_groups:
            terms_and_len_list = list(terms_and_len_iter)
            if len(terms_and_len_list) == 1:
                return {terms_and_len_list[0][0]}
        return set()


class StrongMatchMappingStrategy(MappingStrategy):
    """
    1. sort :class:`.SynonymTermWithMetrics` by highest scoring search match to identify the
       highest scoring match.
    2. query remaining matches to see whether their scores are greater than this best score - the
       differential (i.e. there are many close string matches).
    """

    def __init__(
        self,
        confidence: StringMatchConfidence,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
        search_threshold=80.0,
        symbolic_only: bool = False,
        differential: float = 2.0,
    ):
        """

        :param confidence:
        :param disambiguation_strategies:
        :param search_threshold: only consider synonym terms above this search threshold
        :param symbolic_only: only consider terms that are symbolic
        :param differential: only consider terms with search scores equal or greater to the best match minus this value
        """
        super().__init__(confidence, disambiguation_strategies)
        self.differential = differential
        self.symbolic_only = symbolic_only
        self.search_threshold = search_threshold

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        if self.symbolic_only:
            relevant_terms_with_scores = [
                (term, score)
                for term in terms
                if term.is_symbolic and (score := term.search_score) is not None
            ]
        else:
            relevant_terms_with_scores = [
                (term, score) for term in terms if (score := term.search_score) is not None
            ]

        if len(relevant_terms_with_scores) == 0:
            return set()

        best_score = max((score for _, score in relevant_terms_with_scores))

        return set(
            term
            for term, score in relevant_terms_with_scores
            if score >= self.search_threshold and best_score - score <= self.differential
        )


class StrongMatchWithEmbeddingConfirmationStringMatchingStrategy(StrongMatchMappingStrategy):
    """
    Same as parent class, but a complex string scorer with a predefined threshold is used to confirm that the
    ent_match is broadly similar to one of the terms attached to the
    :class:`.SynonymTermWithMetrics`\\ .

    Useful for refining non-symbolic close string matches (e.g. "Neck disease" and "Heck disease").
    """

    def __init__(
        self,
        confidence: StringMatchConfidence,
        complex_string_scorer: StringSimilarityScorer,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
        search_threshold: float = 80.0,
        embedding_threshold: float = 0.60,
        symbolic_only: bool = False,
        differential: float = 2.0,
    ):
        """
        :param confidence:
        :param complex_string_scorer: only consider synonym terms passing this string scorer call
        :param disambiguation_strategies:
        :param search_threshold: only consider synonym terms above this search threshold
        :param embedding_threshold: the Entity.match and one of the SynonymTermWithMetrics.terms must be
            above this threshold (according to the complex_string_scorer) for the term to be valid
        :param symbolic_only: only consider terms that are symbolic
        :param differential: only consider terms with search scores equal or greater to the best match minus this value
        """
        super().__init__(
            confidence=confidence,
            search_threshold=search_threshold,
            differential=differential,
            symbolic_only=symbolic_only,
            disambiguation_strategies=disambiguation_strategies,
        )
        self.embedding_threshold = embedding_threshold
        self.complex_string_scorer = complex_string_scorer

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        synonym_term_sorted_by_score = sorted(
            super().filter_terms(
                ent_match=ent_match,
                ent_match_norm=ent_match_norm,
                document=document,
                terms=terms,
                parser_name=parser_name,
            ),
            key=lambda x: x.search_score,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        selected_id_sets = set()
        selected_terms = set()
        for term in synonym_term_sorted_by_score:
            if term.associated_id_sets not in selected_id_sets:
                selected_id_sets.add(term.associated_id_sets)
                if any(
                    self.complex_string_scorer(ent_match, original_term) >= self.embedding_threshold
                    for original_term in term.terms
                ):
                    selected_terms.add(term)
        return selected_terms


class NoopMappingStrategy(MappingStrategy):
    """
    a No-op strategy in case you want to pass all terms straight through for disambiguation
    """

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        return set(terms)
