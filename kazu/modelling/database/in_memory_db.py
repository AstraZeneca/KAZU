import copy
import logging
from typing import Optional, Dict, List, Tuple, Set, Iterable

from kazu.data.data import SynonymTerm, SimpleValue, EquivalentIdAggregationStrategy
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

    _database: Dict[ParserName, Dict[Idx, Metadata]] = {}
    _keys_lst: Dict[ParserName, List[Idx]] = {}
    loaded_parsers: Set[str] = set()

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
        return copy.deepcopy(self._database[name][idx])

    def get_by_index(self, name: ParserName, i: int) -> Tuple[Idx, Metadata]:
        idx = self._keys_lst[name][i]
        return copy.deepcopy((idx, self._database[name][idx],))

    def get_all(self, name: ParserName) -> Dict[Idx, Metadata]:
        """
        get all metadata associated with an ontology

        :param name: name of ontology
        :return:
        """
        return self._database[name]


class SynonymDatabase(metaclass=Singleton):
    """
    Singleton of a database of synonyms.
    """

    _syns_database_by_syn: Dict[ParserName, Dict[NormalisedSynonymStr, SynonymTerm]] = {}
    _syns_by_aggregation_strategy: Dict[
        ParserName, Dict[EquivalentIdAggregationStrategy, Dict[Idx, Set[NormalisedSynonymStr]]]
    ] = {}
    loaded_parsers: Set[ParserName] = set()

    def add(self, name: ParserName, synonyms: Iterable[SynonymTerm]):
        """
        add synonyms to the database.

        :param name: name of ontology to add to
        :param synonyms: iterable of SynonymTerms to add
        :return:
        """
        self.loaded_parsers.add(name)
        self._syns_database_by_syn[name] = {}
        for synonym in synonyms:
            self._syns_database_by_syn[name][synonym.term_norm] = synonym
            for equiv_ids in synonym.associated_id_sets:
                for idx in equiv_ids.ids:
                    dict_for_this_parser = self._syns_by_aggregation_strategy.setdefault(
                        name, {}
                    )
                    dict_for_this_aggregation_strategy = dict_for_this_parser.setdefault(
                        synonym.aggregated_by, {}
                    )
                    syn_set_for_this_id = dict_for_this_aggregation_strategy.setdefault(
                        idx, set()
                    )
                    syn_set_for_this_id.add(synonym.term_norm)

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

    def get_database(self) -> Dict[ParserName, Dict[NormalisedSynonymStr, SynonymTerm]]:
        return self._syns_database_by_syn
