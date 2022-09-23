import itertools
import logging
import urllib
from typing import List, Optional, Set, Iterable, Dict, FrozenSet, Tuple

from kazu.data.data import (
    Document,
    Mapping,
    EquivalentIdSet,
    SynonymTermWithMetrics,
)
from kazu.data.data import LinkRanks
from kazu.modelling.database.in_memory_db import MetadataDatabase
from kazu.modelling.language.string_similarity_scorers import (
    BooleanStringSimilarityScorer,
)
from kazu.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
)
from kazu.steps.linking.post_processing.disambiguation.strategies import DisambiguationStrategy



class MappingFactory:
    """
    factory class to produce mappings
    """

    metadata_db = MetadataDatabase()

    @staticmethod
    def create_mapping_from_id_set(
        id_set: EquivalentIdSet,
        parser_name: str,
        mapping_strategy: str,
        disambiguation_strategy: Optional[str],
        confidence: LinkRanks,
        additional_metadata: Optional[Dict] = None,
        strip_url: bool = True,
    ) -> Iterable[Mapping]:
        for idx in id_set.ids:
            source = id_set.ids_to_source[idx]
            yield MappingFactory.create_mapping(
                parser_name=parser_name,
                source=source,
                idx=idx,
                mapping_strategy=mapping_strategy,
                disambiguation_strategy=disambiguation_strategy,
                confidence=confidence,
                additional_metadata=additional_metadata if additional_metadata is not None else {},
                strip_url=strip_url,
            )

    @staticmethod
    def create_mapping(
        parser_name: str,
        source: str,
        idx: str,
        mapping_strategy: str,
        disambiguation_strategy: Optional[str],
        confidence: LinkRanks,
        additional_metadata: Optional[Dict],
        strip_url: bool = True,
    ) -> Mapping:
        metadata = MappingFactory.metadata_db.get_by_idx(name=parser_name, idx=idx)
        if additional_metadata:
            metadata.update(additional_metadata)
        if strip_url:
            url = urllib.parse.urlparse(idx)
            if url.scheme == "":
                # not a url
                new_idx = idx
            else:
                new_idx = url.path.split("/")[-1]
        else:
            new_idx = idx
        default_label = metadata.pop(DEFAULT_LABEL)
        assert isinstance(default_label, str)
        return Mapping(
            default_label=default_label,
            idx=new_idx,
            source=source,
            mapping_strategy=mapping_strategy,
            disambiguation_strategy=disambiguation_strategy,
            confidence=confidence,
            parser_name=parser_name,
            metadata=metadata,
        )


class MappingStrategy:
    """
    the purpose of a MappingStrategy is two fold:

    1) Filter the set of SynonymTermWithMetrics associated with an Entity down to the most appropriate ones,
        (e.g. based on string similarity).

    2) If required, apply any configured DisambiguationStrategy to the filtered instances of EquivalentIdSet



    """

    def __init__(
        self,
        confidence: LinkRanks,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
    ):
        """

        :param confidence: the level of confidence that should be assigned to this strategy. This is simply a label
            for human users, and has no bearing on the actual algorithm. Note, if after term filtering and (optional)
            disambiguation, if multiple EquivalentIdSets still remain, they will all receive the confidence label of
            LinkRanks.AMBIGUOUS
        :param disambiguation_strategies: Optional[List[DisambiguationStrategy]] which are triggered if any of the
            filtered SynonymTermWithMetrics are .ambiguous, or multiple SynonymTermWithMetrics remain after the
            filter_terms method is called
        """

        self.confidence = confidence
        self.disambiguation_strategies = disambiguation_strategies

    def prepare(self, document: Document):
        """
        perform any setup that needs to run once per document. Care should be taken if trying to cache this step,
        as the Document state is liable to change between executions.
        :param document:
        :return:
        """
        pass

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        """

        Algorithms should override this method to return a set of the 'best' SynonymTermWithMetrics
        for a given query string. Ideally, this will be a set with a single element. However, it may not be possible to
        identify a single best match. In this scenario, the id sets of multiple SynonymTermWithMetrics will be carried
        forward for disambiguation (if configured)

        :param ent_match: Entity.match
        :param ent_match_norm: normalised version of Entity.match
        :param document: originating Document
        :param terms: terms to filter
        :param parser_name: parser name associated with these terms
        :return: defaults to set(terms) (i.e. no filtering)
        """
        return set(terms)

    def disambiguate_if_required(
        self, filtered_terms: Set[SynonymTermWithMetrics], document: Document, parser_name: str
    ) -> Tuple[Set[EquivalentIdSet], Optional[str]]:
        """
        applies disambiguation strategies if configured, and either len(filtered_terms) > 1 or any
        of the SynonymTermWithMetrics are ambiguous
        :param filtered_terms: terms to disambiguate
        :param document: originating Document
        :param parser_name: parser name associated with these terms
        :return:
        """

        all_id_sets: Set[EquivalentIdSet] = set()
        disambiguation_required = len(filtered_terms) > 1
        for term in filtered_terms:
            all_id_sets.update(term.associated_id_sets)
            if term.is_ambiguous:
                disambiguation_required = True

        if disambiguation_required and self.disambiguation_strategies is not None:
            for strategy in self.disambiguation_strategies:
                filtered_id_sets = strategy(
                    id_sets=all_id_sets, document=document, parser_name=parser_name
                )
                if len(filtered_id_sets) == 1:
                    return filtered_id_sets, strategy.__class__.__name__
            else:
                return all_id_sets, None
        else:
            return all_id_sets, None

    def __call__(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
    ) -> Iterable[Mapping]:
        """

        :param ent_match: unnormalised NER string match (i.e. Entity.match)
        :param ent_match_norm: normalised NER string match (i.e. Entity.match_norm)
        :param document: originating document
        :param terms: set of terms to consider
        :return:
        """
        parser_name = next(iter(terms)).parser_name

        filtered_terms: Set[SynonymTermWithMetrics] = self.filter_terms(
            ent_match=ent_match,
            ent_match_norm=ent_match_norm,
            document=document,
            terms=terms,
            parser_name=parser_name,
        )

        id_sets, successful_disambiguation_strategy = self.disambiguate_if_required(
            filtered_terms, document, parser_name
        )
        for id_set in id_sets:
            for idx in id_set.ids:
                yield MappingFactory.create_mapping(
                    source=id_set.ids_to_source[idx],
                    parser_name=parser_name,
                    idx=idx,
                    confidence=self.confidence if len(id_sets) == 1 else LinkRanks.AMBIGUOUS,
                    additional_metadata=None,
                    mapping_strategy=self.__class__.__name__,
                    disambiguation_strategy=successful_disambiguation_strategy,
                )


class ExactMatchMappingStrategy(MappingStrategy):
    """
    returns any exact matches
    """

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        return set(filter(lambda x: x.exact_match, terms))


class SymbolMatchMappingStrategy(MappingStrategy):
    """
    split both query and reference terms by whitespace. select the term with the most splits as the 'query'. Check
    all of these tokens (and no more) are within the other term. Useful for symbol matching
    e.g.

    MAP K8 (longest)
    vs
    MAPK8 (shortest)

    """

    def match_symbols(self, s1: str, s2: str) -> bool:
        # the pattern should either be in both or neither
        reference_term_tokens = s1.split(" ")
        query_term_tokens = s2.split(" ")
        longest, shortest = (
            (
                reference_term_tokens,
                s2,
            )
            if len(reference_term_tokens) > len(query_term_tokens)
            else (
                query_term_tokens,
                s1,
            )
        )

        for tok in longest:
            if tok not in shortest:
                return False
            else:
                shortest = shortest.replace(tok, "", 1)

        return shortest.strip() == ""

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        return set(term for term in terms if self.match_symbols(ent_match_norm, term.term_norm))


class TermNormIsSubStringMappingStrategy(MappingStrategy):
    """
    see if any terms are substrings of the match norm. If exactly one is returned, prefer it. Works best on symbolic
    entities, e.g. "TESTIN gene" ->"TESTIN"
    """

    def __init__(
        self,
        confidence: LinkRanks,
        min_term_norm_len_to_consider: int = 3,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
    ):
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
        norm_tokens = ent_match_norm.split(" ")

        filtered_terms_and_len = [
            (
                term,
                len(term.term_norm),
            )
            for term in terms
            if any(term.term_norm == norm_token for norm_token in norm_tokens)
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
    1) sort SynonymTermWithMetrics by highest scoring search match t identify the highest scoring match
    2) query remaining matches to see whether their scores are greater than this best score - the differential
        (i.e. there are many close string matches)
    """

    def __init__(
        self,
        confidence: LinkRanks,
        search_threshold=80.0,
        symbolic_only: bool = False,
        differential: float = 2.0,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
    ):
        """

        :param confidence:
        :param search_threshold: only consider synonym terms above this search threshold
        :param symbolic_only: only consider terms that are symbolic
        :param differential: only consider terms with search scores equal or greater to the best match minus this value
        :param disambiguation_strategies:
        """
        super().__init__(confidence, disambiguation_strategies)
        self.differential = differential
        self.symbolic_only = symbolic_only
        self.search_threshold = search_threshold

    def filter_and_sort_by_differential(
        self, terms: FrozenSet[SynonymTermWithMetrics]
    ) -> List[SynonymTermWithMetrics]:
        filtered_terms_list: List[SynonymTermWithMetrics] = []

        if self.symbolic_only:
            all_scores = [
                x.search_score for x in terms if x.search_score is not None and x.is_symbolic
            ]
            # note, we need this check as max will throw an exception if sequence is empty
            if len(all_scores) == 0:
                return filtered_terms_list
            else:
                best_symbolic: float = max(all_scores)
                filtered_terms_list.extend(
                    sorted(
                        filter(
                            lambda x: x.search_score >= self.search_threshold  # type: ignore[arg-type,return-value,union-attr,operator]
                            and best_symbolic - x.search_score <= self.differential  # type: ignore[arg-type,return-value,union-attr,operator]
                            and x.is_symbolic,  # type: ignore[arg-type,return-value,union-attr,operator]
                            terms,
                        ),
                        key=lambda x: x.search_score,  # type: ignore[arg-type,return-value,union-attr,operator]
                        reverse=True,
                    )
                )  # type: ignore[arg-type]
        else:
            best_all: float = max(x.search_score for x in terms)  # type: ignore[arg-type,return-value,union-attr,operator,type-var,assignment]
            filtered_terms_list.extend(
                sorted(
                    filter(
                        lambda x: x.search_score >= self.search_threshold  # type: ignore[arg-type,return-value,union-attr,operator,type-var,assignment]
                        and best_all - x.search_score <= self.differential,  # type: ignore[arg-type,return-value,union-attr,operator]
                        terms,
                    ),
                    key=lambda x: x.search_score,  # type: ignore[arg-type,return-value,union-attr,operator]
                    reverse=True,
                )
            )  # type: ignore[arg-type]  # type: ignore[arg-type]

        return filtered_terms_list

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:

        selected_terms = self.filter_and_sort_by_differential(terms)
        return set(selected_terms)


class StrongMatchWithEmbeddingConfirmationStringMatchingStrategy(StrongMatchMappingStrategy):
    """
    1) same as parent class, but a complex string scorer with a predefined threshold is used to confirm that the
        ent_match is broadly similar to one of the terms attached to the SynonymTermWithMetrics. useful for refining
        non symbolic close string matches (e.g. Heck disease and Heck disease)
    """

    def __init__(
        self,
        confidence: LinkRanks,
        complex_string_scorer: BooleanStringSimilarityScorer,
        search_threshold: float = 80.0,
        symbolic_only: bool = False,
        differential: float = 2.0,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
    ):
        """

        :param confidence:
        :param complex_string_scorer: only consider synonym terms passing this string scorer call
        :param search_threshold: only consider synonym terms above this search threshold
        :param symbolic_only: only consider terms that are symbolic
        :param differential: only consider terms with search scores equal or greater to the best match minus this value
        :param disambiguation_strategies:
        """

        super().__init__(
            confidence=confidence,
            search_threshold=search_threshold,
            differential=differential,
            symbolic_only=symbolic_only,
            disambiguation_strategies=disambiguation_strategies,
        )
        self.complex_string_scorer = complex_string_scorer

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        synonym_term_sorted_by_score = self.filter_and_sort_by_differential(terms)
        selected_id_sets = set()
        selected_terms = set()
        for term in synonym_term_sorted_by_score:
            if term.associated_id_sets not in selected_id_sets:
                selected_id_sets.add(term.associated_id_sets)
                if self.complex_string_scorer(ent_match, list(term.terms)[0]):
                    selected_terms.add(term)
        return selected_terms


class DefinedElsewhereInDocumentMappingStrategy(MappingStrategy):
    """
    1) look for entities on the document that have mappings
    2) see if any of these mappings correspond to any ids in the EquivalentIdSets on each SynonymTermWithMetrics
    3) filter the synonym terms according to detected mappings
    """

    def __init__(
        self,
        confidence: LinkRanks,
        disambiguation_strategies: Optional[List[DisambiguationStrategy]] = None,
    ):

        super().__init__(confidence, disambiguation_strategies)
        self.found_equivalent_ids: Set[Tuple[str, str, str]] = set()

    def prepare(self, document: Document):
        """
        can't be cached: document state may change between executions
        :param document:
        :return:
        """
        self.found_equivalent_ids.clear()
        entities = document.get_entities()
        self.found_equivalent_ids.update(
            (
                mapping.parser_name,
                mapping.source,
                mapping.idx,
            )
            for ent in entities
            for mapping in ent.mappings
        )

    def filter_terms(
        self,
        ent_match: str,
        ent_match_norm: str,
        document: Document,
        terms: FrozenSet[SynonymTermWithMetrics],
        parser_name: str,
    ) -> Set[SynonymTermWithMetrics]:
        found_terms = set()
        for term in terms:
            for id_set in term.associated_id_sets:
                for idx in id_set.ids:
                    if (
                        term.parser_name,
                        id_set.ids_to_source[idx],
                        idx,
                    ) in self.found_equivalent_ids:
                        found_terms.add(term)
                        break
        return found_terms
