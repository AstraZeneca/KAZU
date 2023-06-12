import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from enum import auto
from typing import (
    cast,
    List,
    Tuple,
    Dict,
    Iterable,
    Set,
    Optional,
    FrozenSet,
    DefaultDict,
    Literal,
    Any,
)

import pandas as pd
from kazu.data.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTerm,
    SimpleValue,
    CuratedTerm,
    ParserBehaviour,
    CuratedTermBehaviour,
    AssociatedIdSets,
    GlobalParserActions,
    AutoNameEnum,
    MentionConfidence,
)
from kazu.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
    NormalisedSynonymStr,
    Idx,
)
from kazu.language.string_similarity_scorers import StringSimilarityScorer
from kazu.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import PathLike, as_path

# dataframe column keys
DEFAULT_LABEL = "default_label"
IDX = "idx"
SYN = "syn"
MAPPING_TYPE = "mapping_type"
SOURCE = "source"
DATA_ORIGIN = "data_origin"
IdsAndSource = Set[Tuple[str, str]]

logger = logging.getLogger(__name__)


class CurationException(Exception):
    pass


def load_curated_terms(
    path: PathLike,
) -> List[CuratedTerm]:
    """
    Load :class:`kazu.data.data.CuratedTerm`\\ s from a file path.

    :param path: path to json lines file that map to :class:`kazu.data.data.CuratedTerm`
    :return:
    """
    curations_path = as_path(path)
    if curations_path.exists():
        with curations_path.open(mode="r") as jsonlf:
            curations = [CuratedTerm.from_json(line) for line in jsonlf]
    else:
        raise ValueError(f"curations do not exist at {path}")
    return curations


def load_global_actions(
    path: PathLike,
) -> GlobalParserActions:
    """
    Load an instance of :class:`.GlobalParserActions` from a file path.

    :param path: path to a json serialised GlobalParserActions
    :return:
    """
    global_actions_path = as_path(path)
    if global_actions_path.exists():
        with global_actions_path.open(mode="r") as jsonlf:
            global_actions = GlobalParserActions.from_json(json.load(jsonlf))
    else:
        raise ValueError(f"global actions do not exist at {path}")
    return global_actions


class CurationModificationResult(AutoNameEnum):
    ID_SET_MODIFIED = auto()
    SYNONYM_TERM_ADDED = auto()
    SYNONYM_TERM_DROPPED = auto()
    NO_ACTION = auto()


class CurationProcessor:
    """A CurationProcessor is responsible for modifying the set of :class:`.SynonymTerm`\\s produced
    by an :class:`kazu.ontology_preprocessing.base.OntologyParser` with any relevant :class:`.GlobalParserActions` and/or
    :class:`.CuratedTerm` associated with the parser.

    That is to say, this class modifies the raw data produced by a parser with any a posteriori
    observations about the data (such as bad synonyms, mis-mapped terms etc). Is also identifies
    curations that should be used for dictionary based NER.

    This class should be used before instances of :class:`.SynonymTerm`\\s are loaded into the
    internal database representation.
    """

    # curations are applied in the following order
    CURATION_APPLY_ORDER = (
        CuratedTermBehaviour.IGNORE,
        CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
        CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
        CuratedTermBehaviour.INHERIT_FROM_SOURCE_TERM,
    )
    _BEHAVIOUR_TO_ORDER_INDEX = {behav: i for i, behav in enumerate(CURATION_APPLY_ORDER)}

    def __init__(
        self,
        parser_name: str,
        entity_class: str,
        global_actions: Optional[GlobalParserActions],
        curations: List[CuratedTerm],
        synonym_terms: Set[SynonymTerm],
    ):
        """

        :param parser_name: name of parser to process
        :param entity_class: name of parser entity_class to process (typically as passed to :class:`kazu.ontology_preprocessing.base.OntologyParser`\\ )
        :param global_actions:
        :param curations:
        :param synonym_terms:
        """
        self.global_actions = global_actions
        self.entity_class = entity_class
        self.parser_name = parser_name
        self._terms_by_term_norm: Dict[NormalisedSynonymStr, SynonymTerm] = {}
        self._terms_by_id: DefaultDict[Idx, Set[SynonymTerm]] = defaultdict(set)
        # used by inherited curations to decide behaviour
        self._curations_by_syn: DefaultDict[str, Set[CuratedTerm]] = defaultdict(set)
        for term in synonym_terms:
            self._update_term_lookups(term, False)
        self.curations = set(curations)

    @classmethod
    def curation_sort_key(cls, curated_term: CuratedTerm) -> Tuple[int, bool, str]:
        """Determines the order curations are processed in.

        We use associated_id_sets as a key, so that any overrides will be
        processed after any original behaviours
        """
        return (
            cls._BEHAVIOUR_TO_ORDER_INDEX[curated_term.behaviour],
            curated_term.associated_id_sets is not None,
            curated_term.curated_synonym,
        )

    def _update_term_lookups(
        self, term: SynonymTerm, override: bool
    ) -> Literal[
        CurationModificationResult.SYNONYM_TERM_ADDED, CurationModificationResult.NO_ACTION
    ]:
        assert term.original_term is None

        safe_to_add = False
        maybe_existing_term = self._terms_by_term_norm.get(term.term_norm)
        if maybe_existing_term is None:
            logger.debug("adding new term %s", term)
            safe_to_add = True
        elif override:
            safe_to_add = True
            logger.debug("overriding existing term %s", maybe_existing_term)
        elif (
            len(
                term.associated_id_sets.symmetric_difference(maybe_existing_term.associated_id_sets)
            )
            > 0
        ):
            logger.warning(
                "conflict on term norms \n%s\n%s\nthe latter will be ignored",
                maybe_existing_term,
                term,
            )
        if safe_to_add:
            self._terms_by_term_norm[term.term_norm] = term
            for equiv_ids in term.associated_id_sets:
                for idx in equiv_ids.ids:
                    self._terms_by_id[idx].add(term)
            return CurationModificationResult.SYNONYM_TERM_ADDED
        else:
            return CurationModificationResult.NO_ACTION

    def _drop_synonym_term(self, synonym: NormalisedSynonymStr) -> None:
        """Remove a synonym term from the database, so that it cannot be used as a linking target.

        :param synonym:
        :return:
        """
        try:
            term_to_remove = self._terms_by_term_norm.pop(synonym)
            for equiv_id_set in term_to_remove.associated_id_sets:
                for idx in equiv_id_set.ids:
                    terms_by_id = self._terms_by_id.get(idx)
                    if terms_by_id is not None:
                        terms_by_id.remove(term_to_remove)
            logger.debug(
                "successfully dropped %s from database for %s",
                synonym,
                self.entity_class,
            )
        except KeyError:
            logger.warning(
                "tried to drop %s from database, but key doesn't exist for %s",
                synonym,
                self.parser_name,
            )

    def _drop_id_from_all_synonym_terms(self, id_to_drop: Idx) -> Counter:
        """Remove a given id from all :class:`.SynonymTerm`\\ s.

        Drop any :class:`.SynonymTerm`\\ s with no remaining ID after removal.

        :param id_to_drop:
        :return: counter of :class:`.CurationModificationResult`
        """

        terms_to_modify = self._terms_by_id.get(id_to_drop, set())
        counter = Counter(
            self._drop_id_from_synonym_term(id_to_drop=id_to_drop, term_to_modify=term_to_modify)
            for term_to_modify in set(terms_to_modify)
        )

        return counter

    def _drop_id_from_synonym_term(
        self, id_to_drop: Idx, term_to_modify: SynonymTerm
    ) -> Literal[
        CurationModificationResult.ID_SET_MODIFIED,
        CurationModificationResult.SYNONYM_TERM_DROPPED,
        CurationModificationResult.NO_ACTION,
    ]:
        """Remove an id from a given :class:`.SynonymTerm`\\ .

        :param id_to_drop:
        :param term_to_modify:
        :return:
        """
        new_assoc_id_frozenset = self._drop_id_from_associated_id_sets(
            id_to_drop, term_to_modify.associated_id_sets
        )
        if len(new_assoc_id_frozenset.symmetric_difference(term_to_modify.associated_id_sets)) == 0:
            return CurationModificationResult.NO_ACTION
        else:
            return self._modify_or_drop_synonym_term_after_id_set_change(
                new_associated_id_sets=new_assoc_id_frozenset, synonym_term=term_to_modify
            )

    def _drop_id_from_associated_id_sets(
        self, id_to_drop: Idx, associated_id_sets: AssociatedIdSets
    ) -> AssociatedIdSets:
        """Remove an id from a :class:`.AssociatedIdSets`\\ .

        :param id_to_drop:
        :param associated_id_sets:
        :return:
        """
        new_assoc_id_set = set()
        for equiv_id_set in associated_id_sets:
            if id_to_drop in equiv_id_set.ids:
                updated_ids_and_source = frozenset(
                    id_tup for id_tup in equiv_id_set.ids_and_source if id_tup[0] != id_to_drop
                )
                if len(updated_ids_and_source) > 0:
                    updated_equiv_id_set = EquivalentIdSet(updated_ids_and_source)
                    new_assoc_id_set.add(updated_equiv_id_set)
            else:
                new_assoc_id_set.add(equiv_id_set)
        new_assoc_id_frozenset = frozenset(new_assoc_id_set)
        return new_assoc_id_frozenset

    def _modify_or_drop_synonym_term_after_id_set_change(
        self, new_associated_id_sets: AssociatedIdSets, synonym_term: SynonymTerm
    ) -> Literal[
        CurationModificationResult.ID_SET_MODIFIED, CurationModificationResult.SYNONYM_TERM_DROPPED
    ]:
        """Modifies or drops a :class:`.SynonymTerm` after a :class:`.AssociatedIdSets` has
        changed.

        :param new_associated_id_sets:
        :param synonym_term:
        :return:
        """
        result: Literal[
            CurationModificationResult.ID_SET_MODIFIED,
            CurationModificationResult.SYNONYM_TERM_DROPPED,
        ]
        if len(new_associated_id_sets) > 0:
            if new_associated_id_sets == synonym_term.associated_id_sets:
                raise ValueError(
                    "function called inappropriately where the id sets haven't changed. This"
                    " has failed as it will otherwise modify the value of aggregated_by, when"
                    " nothing has changed"
                )
            new_term = dataclasses.replace(
                synonym_term,
                associated_id_sets=new_associated_id_sets,
                aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
            )
            add_result = self._update_term_lookups(new_term, True)
            assert add_result is CurationModificationResult.SYNONYM_TERM_ADDED
            result = CurationModificationResult.ID_SET_MODIFIED
        else:
            # if there are no longer any id sets associated with the record, remove it completely
            self._drop_synonym_term(synonym_term.term_norm)
            result = CurationModificationResult.SYNONYM_TERM_DROPPED
        return result

    def export_curations_and_final_terms(
        self,
    ) -> Tuple[List[CuratedTerm], Set[SynonymTerm]]:
        """Perform any updates required to the synonym terms as specified in the curations/global
        actions.

        The returned :class:`.CuratedTerm`\\s can be used for Dictionary based NER, whereas the
        returned :class:`.SynonymTerm`\\s can be loaded into the internal database for linking.

        :return:
        """
        self._process_global_actions()
        return list(self._process_curations()), set(self._terms_by_term_norm.values())

    def _process_curations(self) -> Iterable[CuratedTerm]:
        safe_curations = self.fix_conflicts_in_curations(self.curations)
        for curation in sorted(safe_curations, key=self.curation_sort_key):
            curation = self._process_curation_action(curation)
            yield curation

    def fix_conflicts_in_curations(self, curations: Set[CuratedTerm]) -> Set[CuratedTerm]:
        """Check to see if a list of curations contain conflicts.

        Conflicts can occur for the following reasons:

        1) If two or more curations normalise to the same NormalisedSynonymStr,
           but have different behaviours.

        2) If two or more curations normalise to the same NormalisedSynonymStr,
           but have different associated ID sets specified, such that one would
           override the other.

        3) If two or more curations have conflicting values for case sensitivity and
           mention_confidence. E.g. A case-insensitive curation cannot have a higher
           mention confidence value than a case-sensitive one for the same synonym.


        :param curations:
        :return: safe curations set
        """

        curations_by_term_norm = defaultdict(set)
        curations_by_syn_lower = defaultdict(set)
        for curation in curations:
            curations_by_term_norm[curation.term_norm_for_linking(self.entity_class)].add(curation)
            curations_by_syn_lower[curation.curated_synonym.lower()].add(curation)

        all_remove = set()
        for potential_conflict_set in curations_by_term_norm.values():
            if len(potential_conflict_set) > 1:
                to_add, to_remove = self.resolve_behaviour_conflicts(potential_conflict_set)
                all_remove.update(to_remove)
                curations.update(to_add)

        for potential_conflict_set in curations_by_syn_lower.values():
            potential_conflict_set.difference_update(all_remove)
            if len(potential_conflict_set) > 1:
                to_add, to_remove = self.resolve_case_conflicts(potential_conflict_set)
                all_remove.update(to_remove)
                curations.update(to_add)

        curations.difference_update(all_remove)
        return curations

    def resolve_behaviour_conflicts(
        self, curations: Set[CuratedTerm]
    ) -> Tuple[Set[CuratedTerm], Set[CuratedTerm]]:

        curations_by_syn_lower = defaultdict(set)
        potentially_conflicting_behaviours = set()
        potentially_conflicting_id_sets = set()
        source_curations = set()
        inherited_curations = set()
        for curation in curations:
            if curation.source_term is None:
                source_curations.add(curation)
                if curation.behaviour in {
                    CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
                    CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
                    CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
                }:
                    potentially_conflicting_behaviours.add(curation.behaviour)
                if curation.associated_id_sets is not None:
                    potentially_conflicting_id_sets.add(curation.associated_id_sets)
            else:
                inherited_curations.add(curation)
            curations_by_syn_lower[curation.curated_synonym.lower()].add(curation)

        if len(potentially_conflicting_behaviours) > 1:
            resolved_behaviour = (
                CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING
                if CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING
                in potentially_conflicting_behaviours
                else CuratedTermBehaviour.IGNORE
            )
            logger.warning(
                "conflicting behaviours detected. The following source curations will be %s\n%s",
                resolved_behaviour.name,
                "\n\n".join(curation.to_json() for curation in source_curations) + "\n\n",
            )
            if len(inherited_curations) > 0:
                logger.warning(
                    "conflicting behaviours detected. The following inherited curations will be %s\n%s",
                    CuratedTermBehaviour.IGNORE.name,
                    "\n\n".join(curation.to_json() for curation in inherited_curations) + "\n\n",
                )
            return (
                set(
                    dataclasses.replace(curation, behaviour=resolved_behaviour)
                    for curation in source_curations
                ).union(
                    dataclasses.replace(curation, behaviour=CuratedTermBehaviour.IGNORE)
                    for curation in inherited_curations
                ),
                curations,
            )
        if len(potentially_conflicting_id_sets) > 1:
            raise CurationException(
                "conflicting id sets detected in curations. Please fix the below curations\n%s",
                "\n\n".join(curation.to_json() for curation in source_curations)
                + "\n\n"
                + "\n\n".join(curation.to_json() for curation in inherited_curations)
                + "\n\n",
            )
        return set(), set()

    def resolve_case_conflicts(
        self, curations: Set[CuratedTerm]
    ) -> Tuple[Set[CuratedTerm], Set[CuratedTerm]]:

        to_add: Set[CuratedTerm] = set()
        to_remove: Set[CuratedTerm] = set()
        cs_conf = set()
        ci_conf = set()
        cs_lookup: DefaultDict[str, Tuple[Set[CuratedTerm], Set[MentionConfidence]]] = defaultdict(
            lambda: (set(), set())
        )
        for curation in curations:
            if curation.behaviour in {
                CuratedTermBehaviour.INHERIT_FROM_SOURCE_TERM,
                CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
            }:
                if curation.case_sensitive:
                    cs_conf.add(curation.mention_confidence)
                    cs_lookup[curation.curated_synonym][0].add(curation)
                    cs_lookup[curation.curated_synonym][1].add(curation.mention_confidence)
                else:
                    ci_conf.add(curation.mention_confidence)

        if len(ci_conf.union(cs_conf)) == 1:
            logger.debug("curations OK \n %s", curations)
        elif len(ci_conf) > 1 or (
            len(ci_conf) > 0 and len(cs_conf) > 0 and min(ci_conf) < min(cs_conf)
        ):
            # set all to CI min
            target_conf = min(ci_conf)
            logger.warning(
                "conflict detected in case sensitivity/confidence combination for the following curations. Affected curations will adopt the conservative confidence value %s,%s",
                target_conf.name,
                "\n".join(curation.to_json() for curation in curations) + "\n\n",
            )
            to_add.update(
                dataclasses.replace(curation, mention_confidence=target_conf)
                for curation in curations
            )
            to_remove.update(curations)
        elif len(cs_conf) > 1:
            for curations_cs, cs_confs_by_exact_match in cs_lookup.values():
                if len(cs_confs_by_exact_match) > 1:
                    target_conf = min(cs_confs_by_exact_match)
                    logger.warning(
                        "conflict detected in case sensitivity/confidence combination for the following curations. Affected curations will adopt the conservative confidence value %s,%s",
                        target_conf.name,
                        "\n".join(curation.to_json() for curation in curations_cs) + "\n\n",
                    )
                    to_add.update(
                        dataclasses.replace(curation, mention_confidence=target_conf)
                        for curation in curations_cs
                    )
                    to_remove.update(curations_cs)

        return to_add, to_remove

    def _process_curation_action(self, curation: CuratedTerm) -> CuratedTerm:

        if curation.source_term is None:
            self._curations_by_syn[curation.curated_synonym].add(curation)
        term_norm = curation.term_norm_for_linking(self.entity_class)
        if curation.behaviour is CuratedTermBehaviour.IGNORE:
            logger.debug("curation ignored: %s for %s", curation, self.parser_name)
        elif curation.behaviour is CuratedTermBehaviour.INHERIT_FROM_SOURCE_TERM:
            assert curation.source_term is not None, curation
            logger.debug(
                "curation inherits linking behaviour from %s for %s",
                curation.source_term,
                self.parser_name,
            )
            source_curations = self._curations_by_syn.get(curation.source_term)
            if source_curations is None:
                logger.warning(
                    "no source curation found for %s for %s. The curation will be ignored",
                    curation.source_term,
                    self.parser_name,
                )
                return dataclasses.replace(curation, behaviour=CuratedTermBehaviour.IGNORE)
            elif len(source_curations) > 1:
                logger.warning(
                    "multiple source curations found for %s for %s. The curation will be ignored",
                    curation.source_term,
                    self.parser_name,
                )
                return dataclasses.replace(curation, behaviour=CuratedTermBehaviour.IGNORE)
        elif curation.behaviour is CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING:
            self._drop_synonym_term(term_norm)
        elif curation.behaviour is CuratedTermBehaviour.ADD_FOR_LINKING_ONLY:
            self._attempt_to_add_database_entry_for_curation(
                curation_associated_id_set=curation.associated_id_sets,
                curated_synonym=curation.curated_synonym,
                curation_term_norm=term_norm,
            )

        elif curation.behaviour is CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING:
            self._attempt_to_add_database_entry_for_curation(
                curation_associated_id_set=curation.associated_id_sets,
                curated_synonym=curation.curated_synonym,
                curation_term_norm=term_norm,
            )
            term_for_this_curation = self._terms_by_term_norm.get(term_norm)
            if term_for_this_curation is None:
                logger.warning(
                    "CuratedTerm %s is invalid: "
                    "requires an identifier but none was found. It may have been removed by another curation, or not exist in the underlying data sourcee.",
                    curation,
                )
                return dataclasses.replace(curation, behaviour=CuratedTermBehaviour.IGNORE)
        else:
            raise ValueError(f"unknown behaviour for parser {self.parser_name}, {curation}")
        return curation

    def _process_global_actions(self) -> None:
        if self.global_actions is None:
            return None

        override_curations_by_id = defaultdict(set)
        for curation in self.curations:
            if curation.associated_id_sets is not None:
                for equiv_id_set in curation.associated_id_sets:
                    for idx in equiv_id_set.ids:
                        override_curations_by_id[idx].add(curation)

        for action in self.global_actions.parser_behaviour(self.parser_name):
            if action.behaviour is ParserBehaviour.DROP_IDS_FROM_PARSER:
                ids = action.parser_to_target_id_mappings[self.parser_name]
                for idx in ids:
                    counter_this_idx = self._drop_id_from_all_synonym_terms(idx)
                    if (
                        counter_this_idx[CurationModificationResult.ID_SET_MODIFIED]
                        + counter_this_idx[CurationModificationResult.SYNONYM_TERM_DROPPED]
                        == 0
                    ):
                        logger.warning("failed to drop %s from %s", idx, self.parser_name)
                    else:
                        logger.debug(
                            "dropped ID %s from %s. SynonymTerm modified count: %s, SynonymTerm dropped count: %s",
                            idx,
                            self.parser_name,
                            counter_this_idx[CurationModificationResult.ID_SET_MODIFIED],
                            counter_this_idx[CurationModificationResult.SYNONYM_TERM_DROPPED],
                        )

                        for override_curation_to_modify in set(
                            override_curations_by_id.get(idx, set())
                        ):
                            assert override_curation_to_modify.associated_id_sets is not None
                            new_associated_id_sets = self._drop_id_from_associated_id_sets(
                                idx, override_curation_to_modify.associated_id_sets
                            )
                            if len(new_associated_id_sets) == 0:

                                self.curations.remove(override_curation_to_modify)
                                override_curations_by_id[idx].remove(override_curation_to_modify)
                                logger.debug(
                                    "removed curation %s because of global action",
                                    override_curation_to_modify,
                                )
                            elif (
                                new_associated_id_sets
                                != override_curation_to_modify.associated_id_sets
                            ):
                                self.curations.remove(override_curation_to_modify)
                                override_curations_by_id[idx].remove(override_curation_to_modify)
                                mod_curation = dataclasses.replace(
                                    override_curation_to_modify,
                                    associated_id_sets=new_associated_id_sets,
                                )
                                self.curations.add(mod_curation)
                                override_curations_by_id[idx].add(mod_curation)
                                logger.debug(
                                    "modified curation %s to %s because of global action",
                                    override_curation_to_modify,
                                    mod_curation,
                                )

            else:
                raise ValueError(f"unknown behaviour for parser {self.parser_name}, {action}")
        return None

    def _attempt_to_add_database_entry_for_curation(
        self,
        curation_term_norm: NormalisedSynonymStr,
        curation_associated_id_set: Optional[AssociatedIdSets],
        curated_synonym: str,
    ) -> Literal[
        CurationModificationResult.SYNONYM_TERM_ADDED, CurationModificationResult.NO_ACTION
    ]:
        """
        Create a new :class:`~kazu.data.data.SynonymTerm` for the database, or return an existing
        matching one if already present.

        Notes:

        If a term_norm already exists in self._terms_by_term_norm that matches 'curation_term_norm',
        this method will check to see if the 'curation_associated_id_set' matches the existing terms
        :class:`.AssociatedIdSets`\\.

        If so, no action will be taken. If not, a warning will be logged as adding it
        will cause irregularities in the database.

        If the term_norm does not exist, this method will create a new :class:`~kazu.data.data.SynonymTerm`
        with the provided :class:`.AssociatedIdSets`\\.

        :param curation_term_norm:
        :param curation_associated_id_set:
        :param curated_synonym:
        :return:
        """
        log_prefix = "%(parser_name)s attempting to create synonym term for <%(synonym)s> term_norm: <%(term_norm)s> IDs: %(ids)s}"
        log_formatting_dict: Dict[str, Any] = {
            "parser_name": self.parser_name,
            "synonym": curated_synonym,
            "term_norm": curation_term_norm,
            "ids": curation_associated_id_set,
        }

        # look up the term norm in the db
        maybe_existing_synonym_term = self._terms_by_term_norm.get(curation_term_norm)
        if curation_associated_id_set is None and maybe_existing_synonym_term is not None:
            logger.debug(
                log_prefix
                + " but no associated id set provided, so term will inherit the parser defaults",
                log_formatting_dict,
            )
            return CurationModificationResult.NO_ACTION
        elif curation_associated_id_set is None and maybe_existing_synonym_term is None:
            logger.error(
                log_prefix
                + " but term_norm <%(term_norm)s> does not exist in synonym database."
                + " Since no id set was provided, no entry can be created",
                log_formatting_dict,
            )
            return CurationModificationResult.NO_ACTION

        # curation_associated_id_set is implicitly not None
        assert curation_associated_id_set is not None
        if len(curation_associated_id_set) == 0:
            logger.debug(
                "all ids removed by global action for %s,  Parser name: %s",
                curated_synonym,
                self.parser_name,
            )
            return CurationModificationResult.NO_ACTION

        if maybe_existing_synonym_term is not None:
            log_formatting_dict["existing_id_set"] = maybe_existing_synonym_term.associated_id_sets

            if (
                len(
                    curation_associated_id_set.symmetric_difference(
                        maybe_existing_synonym_term.associated_id_sets
                    )
                )
                == 0
            ):
                logger.debug(
                    log_prefix
                    + " but term_norm <%(term_norm)s> already exists in synonym database."
                    + "since this SynonymTerm matches the id_set, no action is required. %(existing_id_set)s",
                    log_formatting_dict,
                )
                return CurationModificationResult.NO_ACTION
            else:
                logger.debug(
                    log_prefix
                    + " . Will remove existing term_norm <%(term_norm)s> as an ID set override has been specified",
                    log_formatting_dict,
                )

        # no term exists, or we want to override so one will be made
        assert curation_associated_id_set is not None
        for equiv_id_set in set(curation_associated_id_set):
            for idx in equiv_id_set.ids:
                if idx not in self._terms_by_id:
                    curation_associated_id_set = self._drop_id_from_associated_id_sets(
                        id_to_drop=idx, associated_id_sets=curation_associated_id_set
                    )
                    logger.warning(
                        "Attempted to add term containing %s but this id does not exist in the parser and will be ignored",
                        idx,
                    )
        if len(curation_associated_id_set) > 0:
            is_symbolic = StringNormalizer.classify_symbolic(curated_synonym, self.entity_class)
            new_term = SynonymTerm(
                term_norm=curation_term_norm,
                terms=frozenset((curated_synonym,)),
                is_symbolic=is_symbolic,
                mapping_types=frozenset(("kazu_curated",)),
                associated_id_sets=curation_associated_id_set,
                parser_name=self.parser_name,
                aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
            )
            return self._update_term_lookups(new_term, True)
        else:
            return CurationModificationResult.NO_ACTION


class OntologyParser(ABC):
    """
    Parse an ontology (or similar) into a set of outputs suitable for NLP entity linking.
    Implementations should have a class attribute 'name' to something suitably representative.
    The key method is parse_to_dataframe, which should convert an input source to a dataframe suitable
    for further processing.

    The other important method is find_kb. This should parse an ID string (if required) and return the underlying
    source. This is important for composite resources that contain identifiers from different seed sources

    Generally speaking, when parsing a data source, synonyms that are symbolic (as determined by
    the StringNormalizer) that refer to more than one id are more likely to be ambiguous. Therefore,
    we assume they refer to unique concepts (e.g. COX 1 could be 'ENSG00000095303' OR
    'ENSG00000198804', and thus they will yield multiple instances of EquivalentIdSet. Non symbolic
    synonyms (i.e. noun phrases) are far less likely to refer to distinct entities, so we might
    want to merge the associated ID's non-symbolic ambiguous synonyms into a single EquivalentIdSet.
    The result of StringNormalizer.is_symbolic forms the is_symbolic parameter to .score_and_group_ids.

    If the underlying knowledgebase contains more than one entity type, muliple parsers should be
    implemented, subsetting accordingly (e.g. MEDDRA_DISEASE, MEDDRA_DIAGNOSTIC).
    """

    # the synonym table should have these (and only these columns)
    all_synonym_column_names = [IDX, SYN, MAPPING_TYPE]
    # the metadata table should have at least these columns (note, IDX will become the index)
    minimum_metadata_column_names = [DEFAULT_LABEL, DATA_ORIGIN]

    def __init__(
        self,
        in_path: str,
        entity_class: str,
        name: str,
        string_scorer: Optional[StringSimilarityScorer] = None,
        synonym_merge_threshold: float = 0.70,
        data_origin: str = "unknown",
        synonym_generator: Optional[CombinatorialSynonymGenerator] = None,
        curations_path: Optional[str] = None,
        global_actions: Optional[GlobalParserActions] = None,
    ):
        """
        :param in_path: Path to some resource that should be processed (e.g. owl file, db config, tsv etc)
        :param entity_class: The entity class to associate with this parser throughout the pipeline.
            Also used in the parser when calling StringNormalizer to determine the class-appropriate behaviour.
        :param name: A string to represent a parser in the overall pipeline. Should be globally unique
        :param string_scorer: Optional protocol of StringSimilarityScorer.  Used for resolving ambiguous symbolic
            synonyms via similarity calculation of the default label associated with the conflicted labels. If no
            instance is provided, all synonym conflicts will be assumed to refer to different concepts. This is not
            recommended!
        :param synonym_merge_threshold: similarity threshold to trigger a merge of conflicted synonyms into a single
            EquivalentIdSet. See docs for score_and_group_ids for further details
        :param data_origin: The origin of this dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc. Note, this is different
            from the parser.name, as is used to identify the origin of a mapping back to a data source
        :param synonym_generator: optional CombinatorialSynonymGenerator. Used to generate synonyms for dictionary
            based NER matching
        :param curations_path: path to jsonl file of curations to apply to the parser
        """

        self.in_path = in_path
        self.entity_class = entity_class
        self.name = name
        if string_scorer is None:
            logger.warning(
                "no string scorer configured for %s. Synonym resolution disabled.", self.name
            )
        self.string_scorer = string_scorer
        self.synonym_merge_threshold = synonym_merge_threshold
        self.data_origin = data_origin
        self.synonym_generator = synonym_generator
        self.curations_path = curations_path
        self.global_actions = global_actions
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()

    @abstractmethod
    def find_kb(self, string: str) -> str:
        """
        split an IDX somehow to find the ontology SOURCE reference

        :param string: the IDX string to process
        :return:
        """
        pass

    def resolve_synonyms(self, synonym_df: pd.DataFrame) -> Set[SynonymTerm]:

        result = set()
        synonym_df["syn_norm"] = synonym_df[SYN].apply(
            StringNormalizer.normalize, entity_class=self.entity_class
        )

        for i, row in (
            synonym_df[["syn_norm", SYN, IDX, MAPPING_TYPE]]
            .groupby(["syn_norm"])
            .agg(set)
            .reset_index()
            .iterrows()
        ):

            syn_set = row[SYN]
            mapping_type_set: FrozenSet[str] = frozenset(row[MAPPING_TYPE])
            syn_norm = row["syn_norm"]
            if len(syn_set) > 1:
                logger.debug("normaliser has merged %s into a single term: %s", syn_set, syn_norm)

            is_symbolic = all(
                StringNormalizer.classify_symbolic(x, self.entity_class) for x in syn_set
            )

            ids: Set[str] = row[IDX]
            ids_and_source = set(
                (
                    idx,
                    self.find_kb(idx),
                )
                for idx in ids
            )
            associated_id_sets, agg_strategy = self.score_and_group_ids(
                ids_and_source, is_symbolic, syn_set
            )

            synonym_term = SynonymTerm(
                term_norm=syn_norm,
                terms=frozenset(syn_set),
                is_symbolic=is_symbolic,
                mapping_types=mapping_type_set,
                associated_id_sets=associated_id_sets,
                parser_name=self.name,
                aggregated_by=agg_strategy,
            )

            result.add(synonym_term)

        return result

    def score_and_group_ids(
        self,
        ids_and_source: IdsAndSource,
        is_symbolic: bool,
        original_syn_set: Set[str],
    ) -> Tuple[AssociatedIdSets, EquivalentIdAggregationStrategy]:
        """
        for a given data source, one normalised synonym may map to one or more id. In some cases, the ID may be
        duplicate/redundant (e.g. there are many chembl ids for paracetamol). In other cases, the ID may refer to
        distinct concepts (e.g. COX 1 could be 'ENSG00000095303' OR 'ENSG00000198804').


        Since synonyms from data sources are confused in such a manner, we need to decide some way to cluster them into
        a single SynonymTerm concept, which in turn is a container for one or more EquivalentIdSet (depending on
        whether the concept is ambiguous or not)

        The job of score_and_group_ids is to determine how many EquivalentIdSet's for a given set of ids should be
        produced.

        The default algorithm (which can be overridden by concrete parser implementations) works as follows:

        1. If no StringScorer is configured, create an EquivalentIdSet for each id (strategy NO_STRATEGY -
           not recommended)
        2. If only one ID is referenced, or the associated normalised synonym string is not symbolic, group the
           ids into a single EquivalentIdSet (strategy UNAMBIGUOUS)
        3. otherwise, compare the default label associated with each ID to every other default label. If it's above
           self.synonym_merge_threshold, merge into one EquivalentIdSet, if not, create a new one

        recommendation: Use the SapbertStringSimilarityScorer for comparison

        IMPORTANT NOTE: any calls to this method requires the metadata DB to be populated, as this is the store of
        DEFAULT_LABEL

        :param ids_and_source: ids to determine appropriate groupings of, and their associated sources
        :param is_symbolic: is the underlying synonym symbolic?
        :param original_syn_set: original synonyms associated with ids
        :return:
        """
        if self.string_scorer is None:
            # the NO_STRATEGY aggregation strategy assumes all synonyms are ambiguous
            return (
                frozenset(
                    EquivalentIdSet(ids_and_source=frozenset((single_id_and_source,)))
                    for single_id_and_source in ids_and_source
                ),
                EquivalentIdAggregationStrategy.NO_STRATEGY,
            )
        else:

            if len(ids_and_source) == 1:
                return (
                    frozenset((EquivalentIdSet(ids_and_source=frozenset(ids_and_source)),)),
                    EquivalentIdAggregationStrategy.UNAMBIGUOUS,
                )

            if not is_symbolic:
                return (
                    frozenset((EquivalentIdSet(ids_and_source=frozenset(ids_and_source)),)),
                    EquivalentIdAggregationStrategy.MERGED_AS_NON_SYMBOLIC,
                )
            else:
                # use similarity to group ids into EquivalentIdSets

                DefaultLabels = Set[str]
                id_list: List[Tuple[IdsAndSource, DefaultLabels]] = []
                for id_and_source_tuple in ids_and_source:
                    default_label = cast(
                        str,
                        self.metadata_db.get_by_idx(self.name, id_and_source_tuple[0])[
                            DEFAULT_LABEL
                        ],
                    )
                    most_similar_id_set = None
                    best_score = 0.0
                    for id_and_default_label_set in id_list:
                        sim = max(
                            self.string_scorer(default_label, other_label)
                            for other_label in id_and_default_label_set[1]
                        )
                        if sim > self.synonym_merge_threshold and sim > best_score:
                            most_similar_id_set = id_and_default_label_set
                            best_score = sim

                    # for the first label, the above for loop is a no-op as id_sets is empty
                    # and the below if statement will be true.
                    # After that, it will be True if the id under consideration should not
                    # merge with any existing group and should get its own EquivalentIdSet
                    if not most_similar_id_set:
                        id_list.append(
                            (
                                {id_and_source_tuple},
                                {default_label},
                            )
                        )
                    else:
                        most_similar_id_set[0].add(id_and_source_tuple)
                        most_similar_id_set[1].add(default_label)

                return (
                    frozenset(
                        EquivalentIdSet(ids_and_source=frozenset(ids_and_source))
                        for ids_and_source, _ in id_list
                    ),
                    EquivalentIdAggregationStrategy.RESOLVED_BY_SIMILARITY,
                )

    def _parse_df_if_not_already_parsed(self):
        if self.parsed_dataframe is None:
            self.parsed_dataframe = self.parse_to_dataframe()
            self.parsed_dataframe[DATA_ORIGIN] = self.data_origin
            self.parsed_dataframe[IDX] = self.parsed_dataframe[IDX].astype(str)
            self.parsed_dataframe.loc[
                pd.isnull(self.parsed_dataframe[DEFAULT_LABEL]), DEFAULT_LABEL
            ] = self.parsed_dataframe[IDX]

    @kazu_disk_cache.memoize(ignore={0})
    def export_metadata(self, parser_name: str) -> Dict[str, Dict[str, SimpleValue]]:
        """Export the metadata from the ontology.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too much
            memory)
        :return: {idx:{metadata_key:metadata_value}}
        """
        self._parse_df_if_not_already_parsed()
        assert isinstance(self.parsed_dataframe, pd.DataFrame)
        metadata_columns = self.parsed_dataframe.columns
        metadata_columns.drop([MAPPING_TYPE, SYN])
        metadata_df = self.parsed_dataframe[metadata_columns]
        metadata_df = metadata_df.drop_duplicates(subset=[IDX]).dropna(axis=0)
        metadata_df.set_index(inplace=True, drop=True, keys=IDX)
        assert set(OntologyParser.minimum_metadata_column_names).issubset(metadata_df.columns)
        metadata = metadata_df.to_dict(orient="index")
        return cast(Dict[str, Dict[str, SimpleValue]], metadata)

    def process_curations(
        self, terms: Set[SynonymTerm]
    ) -> Tuple[Optional[List[CuratedTerm]], Set[SynonymTerm]]:
        if self.curations_path is None and self.synonym_generator is not None:
            logger.warning(
                "%s is configured to use synonym generators. This may result in noisy NER performance.",
                self.name,
            )
            (
                original_curations,
                generated_curations,
            ) = self.generate_curations_from_synonym_generators(terms)
            curations = original_curations + generated_curations
        elif (
            self.curations_path is None
        ):  # implies self.synonym_generator is None as failed the if above
            logger.warning(
                "%s is configured to use raw ontology synonyms. This may result in noisy NER performance.",
                self.name,
            )
            curations = [
                curation
                for term in terms
                for curation in self.synonym_term_to_putative_curation(term)
            ]
        else:
            assert self.curations_path is not None
            logger.info(
                "%s is configured to use curations. Synonym generation will be ignored",
                self.name,
            )
            curations = load_curated_terms(self.curations_path)

        curation_processor = CurationProcessor(
            global_actions=self.global_actions,
            curations=curations,
            parser_name=self.name,
            entity_class=self.entity_class,
            synonym_terms=terms,
        )
        return curation_processor.export_curations_and_final_terms()

    @kazu_disk_cache.memoize(ignore={0})
    def export_synonym_terms(self, parser_name: str) -> Set[SynonymTerm]:
        """Export :class:`.SynonymTerm` from the parser.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too much
            memory)
        :return:
        """
        self._parse_df_if_not_already_parsed()
        assert isinstance(self.parsed_dataframe, pd.DataFrame)
        # ensure correct order
        syn_df = self.parsed_dataframe[self.all_synonym_column_names].copy()
        syn_df = syn_df.dropna(subset=[SYN])
        syn_df[SYN] = syn_df[SYN].apply(str.strip)
        syn_df.drop_duplicates(subset=self.all_synonym_column_names)
        assert set(OntologyParser.all_synonym_column_names).issubset(syn_df.columns)
        synonym_terms = self.resolve_synonyms(synonym_df=syn_df)
        return synonym_terms

    def generate_synonyms(
        self, original_synonym_data: Set[SynonymTerm]
    ) -> Tuple[Set[SynonymTerm], Set[SynonymTerm]]:
        """Generate synonyms based on configured synonym generator.

        :param original_synonym_data:
        :return: first set is original terms, second are generated terms
        """
        generated_synonym_data = set()
        if self.synonym_generator:
            generated_synonym_data = self.synonym_generator(original_synonym_data)
        logger.info(
            f"{len(original_synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )
        return original_synonym_data, generated_synonym_data

    def generate_curations_from_synonym_generators(
        self, synonym_terms: Set[SynonymTerm]
    ) -> Tuple[List[CuratedTerm], List[CuratedTerm]]:
        original_terms, generated_terms = self.generate_synonyms(synonym_terms)
        original_curations = [
            curation
            for term in original_terms
            for curation in self.synonym_term_to_putative_curation(term)
        ]
        generated_curations = [
            curation
            for term in generated_terms
            for curation in self.synonym_term_to_putative_curation(term)
        ]

        return original_curations, generated_curations

    def synonym_term_to_putative_curation(self, term: SynonymTerm) -> Iterable[CuratedTerm]:
        """Convert a :class:`.SynonymTerm`\\ s to curations to use for dictionary based NER.

        This is used when curations are not provided to the parser.

        Can handle either 'original' or 'generated' :class:`.SynonymTerm`\\ s.

        :param term:
        :return:
        """
        for term_str in term.terms:
            if term.original_term is None:
                behaviour = CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING
            else:
                behaviour = CuratedTermBehaviour.INHERIT_FROM_SOURCE_TERM
            is_symbolic = StringNormalizer.classify_symbolic(term_str, self.entity_class)
            conf = MentionConfidence.POSSIBLE if is_symbolic else MentionConfidence.PROBABLE
            yield CuratedTerm(
                curated_synonym=term_str,
                mention_confidence=conf,
                case_sensitive=is_symbolic,
                behaviour=behaviour,
                source_term=term.original_term,
            )

    @kazu_disk_cache.memoize(ignore={0})
    def _populate_databases(
        self, parser_name: str
    ) -> Tuple[Optional[List[CuratedTerm]], Dict[str, Dict[str, SimpleValue]], Set[SynonymTerm]]:
        """Disk cacheable method that populates all databases.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too much
            memory)
        :return:
        """
        logger.info("populating database for %s from source", self.name)
        metadata = self.export_metadata(self.name)
        # metadata db needs to be populated before call to export_synonym_terms
        self.metadata_db.add_parser(self.name, metadata)
        intermediate_synonym_terms = self.export_synonym_terms(self.name)
        maybe_ner_curations, final_syn_terms = self.process_curations(intermediate_synonym_terms)
        self.parsed_dataframe = None  # clear the reference to save memory

        self.synonym_db.add(self.name, final_syn_terms)
        return maybe_ner_curations, metadata, final_syn_terms

    def populate_databases(
        self, force: bool = False, return_curations: bool = False
    ) -> Optional[List[CuratedTerm]]:
        """Populate the databases with the results of the parser.

        Also calculates the term norms associated with
        any curations (if provided) which can then be used for Dictionary based NER

        :param force: do not use the cache for the ontology parser
        :param return_curations: should processed curations be returned?
        :return: curations if required
        """
        if self.name in self.synonym_db.loaded_parsers and not force and not return_curations:
            logger.debug("parser %s already loaded.", self.name)
            return None

        cache_key = self._populate_databases.__cache_key__(self, self.name)

        if force:
            kazu_disk_cache.delete(self.export_metadata.__cache_key__(self, self.name))
            kazu_disk_cache.delete(self.export_synonym_terms.__cache_key__(self, self.name))
            kazu_disk_cache.delete(cache_key)

        maybe_curations, metadata, final_syn_terms = self._populate_databases(self.name)

        if self.name not in self.synonym_db.loaded_parsers:
            logger.info("populating database for %s from cache", self.name)
            self.metadata_db.add_parser(self.name, metadata)
            self.synonym_db.add(self.name, final_syn_terms)

        return maybe_curations if return_curations else None

    @abstractmethod
    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        implementations should override this method, returning a 'long, thin' pd.DataFrame of at least the following
        columns:


        [IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE]

        IDX: the ontology id
        DEFAULT_LABEL: the preferred label
        SYN: a synonym of the concept
        MAPPING_TYPE: the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by the ontology

        Note: It is the responsibility of the implementation of parse_to_dataframe to add default labels as synonyms.

        Any 'extra' columns will be added to the :class:`~kazu.database.in_memory_db.MetadataDatabase` as metadata fields for the
        given id in the relevant ontology.
        """
        pass
