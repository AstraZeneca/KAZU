import logging
from copy import deepcopy
from typing import Optional, Any
from collections.abc import Iterable

from kazu.data import (
    LinkingCandidate,
    EquivalentIdAggregationStrategy,
    AssociatedIdSets,
)
from kazu.utils.utils import Singleton

logger = logging.getLogger(__name__)

# type aliases to make function signatures more explanatory,
# reduce need for comments.
ParserName = str
Idx = str
NormalisedSynonymStr = str
Metadata = dict[str, Any]


class MetadataDatabase(metaclass=Singleton):
    """Singleton of Ontology metadata database.

    Purpose: metadata needs to be looked up in different linking processes,
    and this singleton allows us to load it once/reduce memory usage
    """

    def __init__(self):
        self._database: dict[ParserName, dict[Idx, Metadata]] = {}
        self._keys_lst: dict[ParserName, list[Idx]] = {}
        self.parser_name_to_ent_class: dict[str, str] = {}

    @property
    def loaded_parsers(self) -> set[ParserName]:
        return set(self.parser_name_to_ent_class.keys())

    def add_parser(
        self, name: ParserName, entity_class: str, metadata: dict[Idx, Metadata]
    ) -> None:
        """Add metadata to the ontology. Note, metadata is assumed to be static, and
        global. Calling this function will override any existing entries with associated
        with the keys in the metadata dict.

        :param name: name of ontology to add to
        :param entity_class: entity class this parser maps to
        :param metadata: dict in format {idx:metadata}
        :return:
        """
        self.parser_name_to_ent_class[name] = entity_class
        if name in self._database:
            logger.info(
                f"parser {name} already present in metadata database - will override existing parser data."
            )
        self._database[name] = metadata
        self._keys_lst[name] = list(self._database[name].keys())

    def get_by_idx(self, name: ParserName, idx: Idx) -> Metadata:
        """Get the metadata associated with an ontology and id.

        :param name: name of ontology to query
        :param idx: idx to query
        :return:
        """
        return deepcopy(self._database[name][idx])

    def get_all(self, name: ParserName) -> dict[Idx, Metadata]:
        """Get all metadata associated with an ontology.

        :param name: name of ontology
        :return:
        """
        return self._database[name]


class SynonymDatabase(metaclass=Singleton):
    """Singleton of a database of synonyms."""

    def __init__(self):
        self._syns_database_by_syn: dict[
            ParserName, dict[NormalisedSynonymStr, LinkingCandidate]
        ] = {}
        self._syns_by_aggregation_strategy: dict[
            ParserName, dict[EquivalentIdAggregationStrategy, dict[Idx, set[NormalisedSynonymStr]]]
        ] = {}
        self._associated_id_sets_by_id: dict[ParserName, dict[str, set[AssociatedIdSets]]] = {}
        self.loaded_parsers: set[ParserName] = set()

    def add_parser(self, name: ParserName, synonyms: Iterable[LinkingCandidate]) -> None:
        """Add synonyms to the database.

        :param name: name of ontology to add to
        :param synonyms: iterable of LinkingCandidates to add
        :return:
        """
        self.loaded_parsers.add(name)
        if name not in self._syns_database_by_syn:
            self._syns_database_by_syn[name] = {}
            self._associated_id_sets_by_id[name] = {}
        for synonym in synonyms:
            self._syns_database_by_syn[name][synonym.synonym_norm] = synonym
            for equiv_ids in synonym.associated_id_sets:
                for idx in equiv_ids.ids:
                    dict_for_this_parser = self._syns_by_aggregation_strategy.setdefault(name, {})
                    dict_for_this_aggregation_strategy = dict_for_this_parser.setdefault(
                        synonym.aggregated_by, {}
                    )
                    syn_set_for_this_id = dict_for_this_aggregation_strategy.setdefault(idx, set())
                    syn_set_for_this_id.add(synonym.synonym_norm)
                    self._associated_id_sets_by_id[name].setdefault(idx, set()).add(
                        synonym.associated_id_sets
                    )

    def get(self, name: ParserName, synonym: NormalisedSynonymStr) -> LinkingCandidate:
        """Get a set of EquivalentIdSets associated with an ontology and synonym string.

        :param name: name of ontology to query
        :param synonym: idx to query
        :return:
        """
        return self._syns_database_by_syn[name][synonym]

    def get_syns_for_id(
        self,
        name: ParserName,
        idx: Idx,
        strategy_filters: Optional[set[EquivalentIdAggregationStrategy]] = None,
    ) -> set[NormalisedSynonymStr]:
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

    def get_all(self, name: ParserName) -> dict[NormalisedSynonymStr, LinkingCandidate]:
        """Get all synonyms associated with an ontology.

        :param name: name of ontology
        :return:
        """
        return self._syns_database_by_syn[name]
