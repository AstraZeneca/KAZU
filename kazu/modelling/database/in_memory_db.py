import dataclasses
import logging
from collections import defaultdict
from copy import deepcopy
from enum import auto
from typing import Optional, Dict, List, Tuple, Set, Iterable, Literal

from kazu.data.data import (
    SynonymTerm,
    SimpleValue,
    EquivalentIdAggregationStrategy,
    AutoNameEnum,
    EquivalentIdSet,
    AssociatedIdSets,
)
from kazu.utils.utils import Singleton

logger = logging.getLogger(__name__)

# type aliases to make function signatures more explanatory,
# reduce need for comments.
ParserName = str
Idx = str
NormalisedSynonymStr = str
Metadata = Dict[str, SimpleValue]


class MetadataDatabase(metaclass=Singleton):
    """
    Singleton of Ontology metadata database. Purpose: metadata needs to be looked up in different linking processes,
    and this singleton allows us to load it once/reduce memory usage
    """

    def __init__(self):
        self._database: Dict[ParserName, Dict[Idx, Metadata]] = {}
        self._keys_lst: Dict[ParserName, List[Idx]] = {}
        self.loaded_parsers: Set[str] = set()

    def add_parser(self, name: ParserName, metadata: Dict[Idx, Metadata]):
        """
        add metadata to the ontology. Note, metadata is assumed to be static, and global. Calling this function will
        override any existing entries with associated with the keys in the metadata dict

        :param name: name of ontology to add to
        :param metadata: dict in format {idx:metadata}
        :return:
        """
        self.loaded_parsers.add(name)
        if name in self._database:
            logger.info(
                f"parser {name} already present in metadata database - will override existing parser data."
            )
        self._database[name] = metadata
        self._keys_lst[name] = list(self._database[name].keys())

    def get_by_idx(self, name: ParserName, idx: Idx) -> Metadata:
        """
        get the metadata associated with an ontology and id

        :param name: name of ontology to query
        :param idx: idx to query
        :return:
        """
        return deepcopy(self._database[name][idx])

    def get_by_index(self, name: ParserName, i: int) -> Tuple[Idx, Metadata]:
        idx = self._keys_lst[name][i]
        return deepcopy(
            (
                idx,
                self._database[name][idx],
            )
        )

    def get_all(self, name: ParserName) -> Dict[Idx, Metadata]:
        """
        get all metadata associated with an ontology

        :param name: name of ontology
        :return:
        """
        return self._database[name]


class DBModificationResult(AutoNameEnum):
    ID_SET_MODIFIED = auto()
    SYNONYM_TERM_ADDED = auto()
    SYNONYM_TERM_DROPPED = auto()
    NO_ACTION = auto()


class SynonymDatabase(metaclass=Singleton):
    """
    Singleton of a database of synonyms.
    """

    def __init__(self):
        self._syns_database_by_syn: Dict[ParserName, Dict[NormalisedSynonymStr, SynonymTerm]] = {}
        self._syns_by_aggregation_strategy: Dict[
            ParserName, Dict[EquivalentIdAggregationStrategy, Dict[Idx, Set[NormalisedSynonymStr]]]
        ] = {}
        self._associated_id_sets_by_id: Dict[ParserName, Dict[str, Set[AssociatedIdSets]]] = {}

        self.loaded_parsers: Set[ParserName] = set()

    def add(
        self, name: ParserName, synonyms: Iterable[SynonymTerm]
    ) -> Literal[DBModificationResult.SYNONYM_TERM_ADDED, DBModificationResult.NO_ACTION]:
        """
        add synonyms to the database.

        :param name: name of ontology to add to
        :param synonyms: iterable of SynonymTerms to add
        :return:
        """
        self.loaded_parsers.add(name)
        result: Literal[
            DBModificationResult.SYNONYM_TERM_ADDED, DBModificationResult.NO_ACTION
        ] = DBModificationResult.NO_ACTION
        if name not in self._syns_database_by_syn:
            self._syns_database_by_syn[name] = {}
            self._associated_id_sets_by_id[name] = defaultdict(set)
        for synonym in synonyms:
            self._syns_database_by_syn[name][synonym.term_norm] = synonym
            for equiv_ids in synonym.associated_id_sets:
                for idx in equiv_ids.ids:
                    dict_for_this_parser = self._syns_by_aggregation_strategy.setdefault(name, {})
                    dict_for_this_aggregation_strategy = dict_for_this_parser.setdefault(
                        synonym.aggregated_by, {}
                    )
                    syn_set_for_this_id = dict_for_this_aggregation_strategy.setdefault(idx, set())
                    syn_set_for_this_id.add(synonym.term_norm)
                    self._associated_id_sets_by_id[name][idx].add(synonym.associated_id_sets)

                    result = DBModificationResult.SYNONYM_TERM_ADDED
        return result

    def drop_synonym_term(
        self, name: ParserName, synonym: NormalisedSynonymStr
    ) -> Literal[DBModificationResult.SYNONYM_TERM_DROPPED]:
        """
        remove a synonym term from the database


        :param name:
        :param synonym:
        :return:
        """
        self._syns_database_by_syn[name].pop(synonym)
        return DBModificationResult.SYNONYM_TERM_DROPPED

    def drop_id_from_all_synonym_terms(
        self, name: ParserName, idx: str
    ) -> Literal[DBModificationResult.ID_SET_MODIFIED, DBModificationResult.NO_ACTION]:
        """
        remove a given id from all synonym terms. If no other ids remain, drop the synonym term all together


        :param name:
        :param idx:
        :return:
        """
        result: Literal[
            DBModificationResult.ID_SET_MODIFIED, DBModificationResult.NO_ACTION
        ] = DBModificationResult.NO_ACTION
        set_of_associated_id_set = self.get_associated_id_sets_for_id(name, idx)
        for term in list(self._syns_database_by_syn[name].values()):
            if term.associated_id_sets in set_of_associated_id_set:
                new_assoc_id_set = set()
                for equiv_id_set in term.associated_id_sets:
                    if idx in equiv_id_set.ids:
                        updated_id_set = EquivalentIdSet(
                            frozenset(
                                id_tup for id_tup in equiv_id_set.ids_and_source if id_tup[0] != idx
                            )
                        )
                        if len(updated_id_set.ids_and_source) > 0:
                            new_assoc_id_set.add(updated_id_set)
                    else:
                        new_assoc_id_set.add(equiv_id_set)
                self._modify_or_drop_synonym_term_after_id_set_change(
                    id_sets=new_assoc_id_set, name=name, synonym_term=term
                )
                result = DBModificationResult.ID_SET_MODIFIED

        return result

    def drop_equivalent_id_set_from_synonym_term(
        self, name: ParserName, synonym: NormalisedSynonymStr, id_set_to_drop: EquivalentIdSet
    ) -> Literal[DBModificationResult.ID_SET_MODIFIED, DBModificationResult.SYNONYM_TERM_DROPPED]:
        """
        remove an EquivalentIdSet from a synonym term, dropping the term all together if
        no others remain

        :param name:
        :param synonym:
        :param id_set_to_drop:
        :return:
        """

        synonym_term = self._syns_database_by_syn[name][synonym]
        id_sets = set(synonym_term.associated_id_sets)
        id_sets.discard(id_set_to_drop)
        result = self._modify_or_drop_synonym_term_after_id_set_change(id_sets, name, synonym_term)
        return result

    def _modify_or_drop_synonym_term_after_id_set_change(
        self, id_sets: Set[EquivalentIdSet], name: ParserName, synonym_term: SynonymTerm
    ) -> Literal[DBModificationResult.ID_SET_MODIFIED, DBModificationResult.SYNONYM_TERM_DROPPED]:
        result: Literal[
            DBModificationResult.ID_SET_MODIFIED, DBModificationResult.SYNONYM_TERM_DROPPED
        ]
        if len(id_sets) > 0:
            new_term = dataclasses.replace(
                synonym_term,
                associated_id_sets=frozenset(id_sets),
                aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
            )
            add_result = self.add(name, (new_term,))
            assert add_result == DBModificationResult.SYNONYM_TERM_ADDED
            result = DBModificationResult.ID_SET_MODIFIED
        else:
            # if there are no longer any id sets associated with the record, remove it completely
            self.drop_synonym_term(name, synonym_term.term_norm)
            result = DBModificationResult.SYNONYM_TERM_DROPPED
        return result

    def drop_equivalent_id_set_containing_id_from_all_synonym_terms(
        self, name: ParserName, id_to_drop: Idx
    ) -> Tuple[int, int]:
        """
        remove all EquivalentIdSet's that contain this id from all SynonymTerms
        in the database

        :param name:
        :param id_to_drop:
        :return:
        """

        terms_modified = 0
        terms_dropped = 0

        for assoc_id_set in self.get_associated_id_sets_for_id(name, id_to_drop):
            for equiv_id_set in assoc_id_set:
                if id_to_drop in equiv_id_set.ids:
                    for term_norm in list(self.get_all(name)):
                        result = self.drop_equivalent_id_set_from_synonym_term(
                            name, term_norm, equiv_id_set
                        )
                        if result == DBModificationResult.ID_SET_MODIFIED:
                            terms_modified += 1
                        elif result == DBModificationResult.SYNONYM_TERM_DROPPED:
                            terms_dropped += 1
        return terms_modified, terms_dropped

    def get(self, name: ParserName, synonym: NormalisedSynonymStr) -> SynonymTerm:
        """
        get a set of EquivalentIdSets associated with an ontology and synonym string

        :param name: name of ontology to query
        :param synonym: idx to query
        :return:
        """
        return self._syns_database_by_syn[name][synonym]

    def get_syns_for_id(
        self,
        name: ParserName,
        idx: Idx,
        strategy_filters: Optional[Set[EquivalentIdAggregationStrategy]] = None,
    ) -> Set[NormalisedSynonymStr]:
        result = set()
        if strategy_filters is None:
            for syn_dict in self._syns_by_aggregation_strategy[name].values():
                result.update(syn_dict.get(idx, set()))
        else:
            for agg_strategy in strategy_filters:
                result.update(
                    self._syns_by_aggregation_strategy[name]
                    .get(agg_strategy, dict())
                    .get(idx, set())
                )
        return result

    def get_syns_sharing_id(
        self,
        name: ParserName,
        synonym: NormalisedSynonymStr,
        strategy_filters: Optional[Set[EquivalentIdAggregationStrategy]] = None,
    ) -> Set[NormalisedSynonymStr]:
        """
        get all other syns for a synonym in a kb

        :param name: parser name
        :param synonym: synonym
        :param strategy_filters: Optional set of EquivalentIdAggregationStrategy. If provided, only syns aggregated
            via these strategies will be returned. If None (the default), all syns will be returned
        :return:
        """
        result: Set[str] = set()
        synonym_term = self.get(name, synonym)
        if strategy_filters is not None and synonym_term.aggregated_by not in strategy_filters:
            return result
        for equiv_id_set in synonym_term.associated_id_sets:
            for idx in equiv_id_set.ids:
                syns = self.get_syns_for_id(
                    name,
                    idx,
                    strategy_filters,
                )
                result.update(syns)
        return result

    def get_all(self, name: ParserName) -> Dict[NormalisedSynonymStr, SynonymTerm]:
        """
        get all synonyms associated with an ontology

        :param name: name of ontology
        :return:
        """
        return self._syns_database_by_syn[name]

    def get_associated_id_sets_for_id(self, name: ParserName, idx: Idx) -> Set[AssociatedIdSets]:
        return self._associated_id_sets_by_id[name][idx]
