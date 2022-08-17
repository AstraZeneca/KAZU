import copy
import logging
from collections import defaultdict
from typing import Optional, DefaultDict, Dict, List, Tuple, Set, Iterable

from kazu.data.data import (
    SynonymTerm,
    SimpleValue,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
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
        # key: parser_name, value: List[IDX]
        keys_lst: DefaultDict[str, List[str]] = defaultdict(list)

        def add_parser(self, name: str, metadata: Dict[str, Dict[str, SimpleValue]]):
            if name in self.database:
                logger.info(
                    f"parser {name} already present in metadata database - will override existing parser data."
                )
            self.database[name] = metadata
            self.keys_lst[name] = list(self.database[name].keys())

        def get_by_idx(self, name: str, idx: str) -> Dict[str, SimpleValue]:
            return self.database[name][idx]

        def get_by_index(self, name: str, i: int) -> Tuple[str, Dict[str, SimpleValue]]:
            idx = self.keys_lst[name][i]
            return idx, self.database[name][idx]

    def __init__(self):
        if not MetadataDatabase.instance:
            MetadataDatabase.instance = MetadataDatabase.__MetadataDatabase()

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

    def add_parser(self, name: str, metadata: Dict[str, Dict[str, SimpleValue]]):
        """
        add metadata to the ontology. Note, metadata is assumed to be static, and global. Calling this function will
        override any existing entries with associated with the keys in the metadata dict
        :param name: name of ontology to add to
        :param metadata: dict in format {idx:metadata}
        :return:
        """
        assert self.instance is not None
        self.instance.add_parser(name, metadata)


class SynonymDatabase:
    """
    Singleton of a database of synonyms.
    """

    instance: Optional["__SynonymDatabase"] = None

    class __SynonymDatabase:
        syns_database_by_syn: DefaultDict[str, Dict[str, SynonymTerm]] = defaultdict(dict)
        syns_by_aggregation_strategy: DefaultDict[
            str, DefaultDict[EquivalentIdAggregationStrategy, DefaultDict[str, Set[str]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        loaded_parsers: Set[str] = set()

        def add(self, name: str, synonyms: Iterable[SynonymTerm]):
            self.loaded_parsers.add(name)
            for synonym in synonyms:
                self.syns_database_by_syn[name][synonym.term_norm] = synonym
                for equiv_ids in synonym.associated_id_sets:
                    for idx in equiv_ids.ids:
                        self.syns_by_aggregation_strategy[name][synonym.aggregated_by][idx].add(
                            synonym.term_norm
                        )

        def get(self, name: str, synonym: str) -> SynonymTerm:
            return self.syns_database_by_syn[name][synonym]

        def get_syns_for_id(
            self,
            name: str,
            idx: str,
            strategy_filters: Optional[Set[EquivalentIdAggregationStrategy]] = None,
        ) -> Set[str]:
            result = set()
            if strategy_filters is None:
                for syn_dict in self.syns_by_aggregation_strategy[name].values():
                    result.update(syn_dict.get(idx, set()))
            else:
                for agg_strategy in strategy_filters:
                    result.update(self.syns_by_aggregation_strategy[name][agg_strategy][idx])
            return result

    def __init__(self):
        if not SynonymDatabase.instance:
            SynonymDatabase.instance = SynonymDatabase.__SynonymDatabase()

    def get(self, name: str, synonym: str) -> SynonymTerm:
        """
        get a set of EquivalentIdSets associated with an ontology and synonym string
        :param name: name of ontology to query
        :param synonym: idx to query
        :return:
        """
        assert self.instance is not None
        return self.instance.get(name, synonym)

    def get_syns_for_id(
        self,
        name: str,
        idx: str,
        strategy_filters: Optional[Set[EquivalentIdAggregationStrategy]] = None,
    ) -> Set[str]:
        assert self.instance is not None
        return self.instance.get_syns_for_id(name, idx, strategy_filters)

    def get_syns_for_synonym(
        self,
        name: str,
        synonym: str,
        strategy_filters: Optional[Set[EquivalentIdAggregationStrategy]] = None,
    ) -> List[Tuple[EquivalentIdSet, Set[str]]]:
        """
        get all other syns for a synonym in a kb
        :param name: parser name
        :param synonym: synonym
        :param strategy_filters: Optional set of EquivalentIdAggregationStrategy. If provided, only syns aggregated
            via these strategies will be returned. If None (the default), all syns will be returned
        :return:
        """
        result = []
        synonym_term = self.get(name, synonym)
        for equiv_id_set in synonym_term.associated_id_sets:
            if strategy_filters is not None and synonym_term.aggregated_by not in strategy_filters:
                continue

            for idx in equiv_id_set.ids:
                result.append(
                    (
                        equiv_id_set,
                        self.get_syns_for_id(
                            name,
                            idx,
                            strategy_filters,
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
        assert self.instance is not None
        return self.instance.syns_database_by_syn

    def add(self, name: str, synonyms: Iterable[SynonymTerm]):
        """
        add synonyms to the database.
        :param name: name of ontology to add to
        :param synonyms: iterable of SynonymTerms to add
        :return:
        """
        assert self.instance is not None
        self.instance.add(name, synonyms)

    def get_loaded_parsers(self) -> Set[str]:
        assert self.instance is not None
        return self.instance.loaded_parsers
