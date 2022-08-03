import copy
import logging
from collections import defaultdict
from typing import Optional, DefaultDict, Dict, List, Tuple, Set, Iterable

from kazu.data.data import (
    SynonymTerm,
    UNAMBIGUOUS_SYNONYM_MERGE_STRATEGIES,
    SimpleValue,
    EquivalentIdSet,
)

logger = logging.getLogger(__name__)


class MetadataDatabase:
    """
    Singleton of Ontology metadata database. Purpose: metadata needs to be looked up in different linking processes,
    and this singleton allows us to load it once/reduce memory usage
    """

    instance: Optional["__MetadataDatabase"] = None

    class __MetadataDatabase:
        # key: parser_name, value: {idx:<generic metadata - dict of strings to simple values>}
        database: DefaultDict[str, Dict[str, Dict[str, SimpleValue]]] = defaultdict(dict)
        # key: parser_name,value: List[IDX]
        keys_lst: DefaultDict[str, List[str]] = defaultdict(list)

        def add(self, name: str, metadata: Dict[str, Dict[str, SimpleValue]]):
            self.database[name].update(metadata)
            self.keys_lst[name] = list(self.database[name].keys())

        def get_by_idx(self, name: str, idx: str) -> Dict[str, SimpleValue]:
            return self.database[name][idx]

        def get_by_index(self, name: str, i: int) -> Tuple[str, Dict[str, SimpleValue]]:
            idx = self.keys_lst[name][i]
            return idx, self.database[name][idx]

    def __init__(self):
        if not MetadataDatabase.instance:
            MetadataDatabase.instance = MetadataDatabase.__MetadataDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get_by_idx(self, name: str, idx: str) -> Dict[str, SimpleValue]:
        """
        get the metadata associated with an ontology and id
        :param name: name of ontology to query
        :param idx: idx to query
        :return:
        """
        assert self.instance is not None
        return copy.deepcopy(self.instance.get_by_idx(name, idx))

    def get_by_index(self, name: str, i: int) -> Tuple[str, Dict[str, SimpleValue]]:
        assert self.instance is not None
        return copy.deepcopy(self.instance.get_by_index(name, i))

    def get_all(self, name: str) -> Dict[str, Dict[str, SimpleValue]]:
        """
        get all metadata associated with an ontology
        :param name: name of ontology
        :return:
        """
        assert self.instance is not None
        return self.instance.database[name]

    def get_loaded_parsers(self) -> Set[str]:
        """
        get the names of all loaded parsers
        :return:
        """
        assert self.instance is not None
        return set(self.instance.database.keys())

    def add(self, name: str, metadata: Dict[str, Dict[str, SimpleValue]]):
        """
        add metadata to the ontology. Note, metadata is assumed to be static, and global. Calling this function will
        override any existing entries with associated with the keys in the metadata dict
        :param name: name of ontology to add to
        :param metadata: dict in format {idx:metadata}
        :return:
        """
        self.instance.add(name, metadata)  # type: ignore


class SynonymDatabase:
    """
    Singleton of a database of synonyms.
    """

    instance: Optional["__SynonymDatabase"] = None

    class __SynonymDatabase:
        syns_database_by_syn: DefaultDict[str, Dict[str, SynonymTerm]] = defaultdict(dict)
        ambig_syns_database_by_idx: DefaultDict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        unambig_syns_database_by_idx: DefaultDict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        loaded_parsers: Set[str] = set()

        def add(self, name: str, synonyms: Iterable[SynonymTerm]):
            self.loaded_parsers.add(name)
            for synonym in synonyms:
                self.syns_database_by_syn[name][synonym.term_norm] = synonym
                for equiv_ids in synonym.associated_id_sets:
                    for idx in equiv_ids.ids:
                        if synonym.aggregated_by in UNAMBIGUOUS_SYNONYM_MERGE_STRATEGIES:
                            self.unambig_syns_database_by_idx[name][idx].add(synonym.term_norm)
                        else:
                            self.ambig_syns_database_by_idx[name][idx].add(synonym.term_norm)

        def get(self, name: str, synonym: str) -> SynonymTerm:
            return self.syns_database_by_syn[name][synonym]

        def get_syns_for_id(self, name: str, idx: str, ignore_ambiguous: bool) -> Set[str]:
            if ignore_ambiguous:
                return self.unambig_syns_database_by_idx[name][idx]
            else:
                result = set()
                result.update(self.unambig_syns_database_by_idx[name][idx])
                result.update(self.ambig_syns_database_by_idx[name][idx])
                return result

    def __init__(self):
        if not SynonymDatabase.instance:
            SynonymDatabase.instance = SynonymDatabase.__SynonymDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get(self, name: str, synonym: str) -> SynonymTerm:
        """
        get a set of EquivalentIdSets associated with an ontology and synonym string
        :param name: name of ontology to query
        :param synonym: idx to query
        :return:
        """
        assert self.instance is not None
        return self.instance.get(name, synonym)

    def get_syns_for_id(self, name: str, idx: str, ignore_ambiguous: bool) -> Set[str]:
        return self.instance.get_syns_for_id(name, idx, ignore_ambiguous)  # type: ignore

    def get_syns_for_synonym(
        self, name: str, synonym: str, ignore_ambiguous: bool
    ) -> List[Tuple[EquivalentIdSet, Set[str]]]:
        """
        get all other syns for a synonym in a kb
        :param name: parser name
        :param synonym: synonym
        :return:
        """
        result = []
        synonym_term = self.get(name, synonym)
        for equiv_id_set in synonym_term.associated_id_sets:
            if (
                ignore_ambiguous
                and synonym_term.aggregated_by not in UNAMBIGUOUS_SYNONYM_MERGE_STRATEGIES
            ):
                continue

            for idx in equiv_id_set.ids:
                result.append(
                    (
                        equiv_id_set,
                        self.get_syns_for_id(
                            name,
                            idx,
                            ignore_ambiguous,
                        ),
                    )
                )
        return result

    def get_all(self, name: str) -> Dict[str, SynonymTerm]:
        """
        get all synonyms associated with an ontology
        :param name: name of ontology
        :return:
        """
        assert self.instance is not None
        return self.instance.syns_database_by_syn[name]

    def get_database(self) -> DefaultDict[str, Dict[str, SynonymTerm]]:
        return self.instance.syns_database_by_syn  # type: ignore

    def add(self, name: str, synonyms: Iterable[SynonymTerm]):
        """
        add synonyms to the database.
        :param name: name of ontology to add to
        :param synonyms: dict in format {synonym string: Set[EquivalentIdSet]}
        :return:
        """
        assert self.instance is not None
        self.instance.add(name, synonyms)

    def get_loaded_parsers(self) -> Set[str]:
        assert self.instance is not None
        return self.instance.loaded_parsers

    def validate_integrity(self):
        failed_terms = []
        for parser_name in self.get_loaded_parsers():
            for term in self.get_all(parser_name).values():
                if term.is_ambiguous and term.aggregated_by in UNAMBIGUOUS_SYNONYM_MERGE_STRATEGIES:
                    # if more than one id_set associated with a term, it can't be unambiguous
                    failed_terms.append(term)

        for term in failed_terms:
            logger.error(f"{term} failed integrity check")
