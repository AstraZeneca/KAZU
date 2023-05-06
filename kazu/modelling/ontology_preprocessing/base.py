import dataclasses
import functools
import json
import logging
from abc import ABC
from collections import defaultdict
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
    Curation,
    ParserBehaviour,
    SynonymTermBehaviour,
    SynonymTermAction,
    AssociatedIdSets,
    GlobalParserActions,
    AutoNameEnum,
    MentionConfidence,
)
from kazu.modelling.database.in_memory_db import (
    MetadataDatabase,
    SynonymDatabase,
    NormalisedSynonymStr,
    Idx,
)
from kazu.modelling.language.string_similarity_scorers import StringSimilarityScorer
from kazu.modelling.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator
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


def select_smallest_associated_id_set_by_equiv_id_set_size_and_id_count(
    set_of_associated_id_sets: Set[AssociatedIdSets],
) -> AssociatedIdSets:
    """
    For a set of :class:`.AssociatedIdSet`\\s, select the set with the fewest :class:`.EquivalentIdSet`\\s. If more than one
    smallest, pick the one with the fewest total IDs


    :param set_of_associated_id_sets:
    :return:
    """

    by_len_and_equiv_len: DefaultDict[DefaultDict[Set[AssociatedIdSets]]] = defaultdict(  # type: ignore[type-arg]
        lambda: defaultdict(set)
    )
    for assoc_id_set in set_of_associated_id_sets:
        total_ids_this_assoc_id_set = set()
        for equiv_id_set in assoc_id_set:
            total_ids_this_assoc_id_set.update(equiv_id_set.ids)

        by_len_and_equiv_len[len(assoc_id_set)][len(total_ids_this_assoc_id_set)].add(assoc_id_set)
    smallest_assoc_dict = by_len_and_equiv_len[min(by_len_and_equiv_len.keys())]
    smallest_set = smallest_assoc_dict[min(smallest_assoc_dict.keys())]
    return next(iter(smallest_set))


def load_curated_terms(
    path: PathLike,
) -> List[Curation]:
    """
    Load :class:`kazu.data.data.Curation`\\ s from a file path.

    :param path: path to json lines file that map to :class:`kazu.data.data.Curation`
    :return:
    """
    curations_path = as_path(path)
    if curations_path.exists():
        with curations_path.open(mode="r") as jsonlf:
            curations = [Curation.from_json(line) for line in jsonlf]
    else:
        raise ValueError(f"curations do not exist at {path}")
    return curations


def load_global_actions(
    path: PathLike,
) -> GlobalParserActions:
    """
    Load an instance of GlobalParserActions  from a file path.

    :param path: path to a json serialised GlobalParserActions`
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
    """
    A CurationProcessor is responsible for modifying the set of :class:`.SynonymTerm`\\s produced by an :class:`.OntologyParser`
    with any relevant :class:`.GlobalParserActions` and/or :class:`.Curation` associated with the parser. That is to say,
    this class modifies the raw data produced by a parser with any a posteriori observations about the data (such as bad
    synonyms, mis-mapped terms etc. Is also identifies curations that should be used for dictionary based NER.
    This class should be used before instances of :class:`.SynonymTerm`\\s are loaded into the internal database
    representation

    """

    # curations are applied in the following order
    curation_apply_order = (
        SynonymTermBehaviour.IGNORE,
        SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING,
        SynonymTermBehaviour.ADD_FOR_LINKING_ONLY,
        SynonymTermBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM,
        SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
        SynonymTermBehaviour.INHERIT_FROM_SOURCE_TERM,
    )

    def __init__(
        self,
        parser_name: str,
        entity_class: str,
        global_actions: Optional[GlobalParserActions],
        curations: List[Curation],
        synonym_terms: Set[SynonymTerm],
    ):
        """

        :param parser_name: name of parser to process
        :param entity_class: name of parser entity_class to process
        :param global_actions:
        :param curations:
        :param synonym_terms:
        """
        self.global_actions = global_actions
        self.entity_class = entity_class
        self.parser_name = parser_name
        self._terms_by_term_norm: Dict[NormalisedSynonymStr, SynonymTerm] = {}
        self._terms_by_id: DefaultDict[str, Set[SynonymTerm]] = defaultdict(set)
        for term in synonym_terms:
            self._update_term_lookups(term, False)
        self.curations: Set[Curation] = set(curations)
        self._curations_by_id: DefaultDict[Optional[str], Set[Curation]] = defaultdict(set)
        for curation in self.curations:
            for action in curation.actions:
                if action.associated_id_sets is None:
                    self._curations_by_id[None].add(curation)
                else:
                    for equiv_id_set in action.associated_id_sets:
                        for idx in equiv_id_set.ids:
                            self._curations_by_id[idx].add(curation)

    @classmethod
    def curation_sort_func(cls, x: Curation, y: Curation):
        """Determines the order curations are processed in"""

        max_x = max(cls.curation_apply_order.index(action.behaviour) for action in x.actions)
        max_y = max(cls.curation_apply_order.index(action.behaviour) for action in y.actions)
        if max_x > max_y:
            return 1
        elif max_y > max_x:
            return -1
        else:
            return 0

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

    def _drop_synonym_term(self, synonym: NormalisedSynonymStr):
        """
        Remove a synonym term from the database, so that it cannot be
        used as a linking target

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

    def _drop_id_from_all_synonym_terms(self, id_to_drop: Idx) -> Tuple[int, int]:
        """
        Remove a given id from all :class:`.SynonymTerm`\\ s.
        Drop any :class:`.SynonymTerm`\\ s with no remaining ID after removal.

        :param id_to_drop:
        :return: terms modified count, terms dropped count
        """
        terms_modified = 0
        terms_dropped = 0
        maybe_terms_to_modify = self._terms_by_id.pop(id_to_drop)
        if maybe_terms_to_modify is not None:
            for term_to_modify in maybe_terms_to_modify:
                result = self._drop_id_from_synonym_term(
                    id_to_drop=id_to_drop, term_to_modify=term_to_modify
                )

                if result is CurationModificationResult.SYNONYM_TERM_DROPPED:
                    terms_dropped += 1
                elif result is CurationModificationResult.ID_SET_MODIFIED:
                    terms_modified += 1
        return terms_modified, terms_dropped

    def _drop_id_from_synonym_term(
        self, id_to_drop: str, term_to_modify: SynonymTerm
    ) -> Literal[
        CurationModificationResult.ID_SET_MODIFIED,
        CurationModificationResult.SYNONYM_TERM_DROPPED,
        CurationModificationResult.NO_ACTION,
    ]:
        """
        Remove an id from a given :class:`.SynonymTerm`

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
        self, id_to_drop: str, associated_id_sets: AssociatedIdSets
    ) -> AssociatedIdSets:
        """
        Remove an id from a :class:`.AssociatedIdSets`

        :param id_to_drop:
        :param associated_id_sets:
        :return:
        """
        new_assoc_id_set = set()
        for equiv_id_set in associated_id_sets:
            updated_ids_and_source = frozenset(
                id_tup for id_tup in equiv_id_set.ids_and_source if id_tup[0] != id_to_drop
            )
            if len(updated_ids_and_source) > 0:
                updated_equiv_id_set = EquivalentIdSet(updated_ids_and_source)
                new_assoc_id_set.add(updated_equiv_id_set)
        new_assoc_id_frozenset = frozenset(new_assoc_id_set)
        return new_assoc_id_frozenset

    def _drop_equivalent_id_set_from_synonym_term(
        self, synonym: NormalisedSynonymStr, id_set_to_drop: EquivalentIdSet
    ) -> Literal[
        CurationModificationResult.ID_SET_MODIFIED, CurationModificationResult.SYNONYM_TERM_DROPPED
    ]:
        """
        Remove an :class:`~kazu.data.data.EquivalentIdSet` from a :class:`~kazu.data.data.SynonymTerm`\\ ,
        dropping the term altogether if no others remain.

        :param synonym:
        :param id_set_to_drop:
        :return:
        """

        synonym_term = self._terms_by_term_norm[synonym]
        modifiable_id_sets = set(synonym_term.associated_id_sets)
        modifiable_id_sets.discard(id_set_to_drop)
        result = self._modify_or_drop_synonym_term_after_id_set_change(
            frozenset(modifiable_id_sets), synonym_term
        )
        return result

    def _modify_or_drop_synonym_term_after_id_set_change(
        self, new_associated_id_sets: AssociatedIdSets, synonym_term: SynonymTerm
    ) -> Literal[
        CurationModificationResult.ID_SET_MODIFIED, CurationModificationResult.SYNONYM_TERM_DROPPED
    ]:
        """
        Modifies or drops a :class:`.SynonymTerm` after a :class:`.AssociatedIdSets` has changed

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
                    "has failed as it will otherwise modify the value of aggregated_by, when"
                    "nothing has changed"
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

    def export_ner_curations_and_final_terms(
        self,
    ) -> Tuple[Optional[List[Curation]], Set[SynonymTerm]]:
        """
        Perform any updates required to the synonym terms as specified in the
        curations/global actions. The returned :class:`.Curation`\\s can be used for
        Dictionary based NER, whereas the returned :class:`.SynonymTerm`\\s can
        be loaded into the internal database for linking.

        :return:
        """
        self._process_global_actions()
        curations_for_ner = self._process_curations()
        return curations_for_ner, set(self._terms_by_term_norm.values())

    def _process_curations(self) -> List[Curation]:
        safe_curations, conflicts = self.analyse_conflicts_in_curations(self.curations)
        for conflict_lst in conflicts:
            message = (
                "\n\nconflicting curations detected\n\n"
                + "\n".join(curation.to_json() for curation in conflict_lst)
                + "\n"
            )

            logger.warning(message)

        curation_for_ner = []
        for curation in sorted(safe_curations, key=functools.cmp_to_key(self.curation_sort_func)):
            maybe_curation_with_term_norm_actions = self._process_curation_actions(curation)
            if maybe_curation_with_term_norm_actions is not None:
                curation_for_ner.append(maybe_curation_with_term_norm_actions)
        return curation_for_ner

    def _drop_id_from_curation(self, idx: str):
        """
        Remove an ID from the curation. If the curation is no longer valid after this action, it will be discarded

        :param idx: the id to remove
        :return:
        """
        affected_curations = set(self._curations_by_id.get(idx, []))
        if affected_curations is not None:
            for affected_curation in affected_curations:
                new_actions = []
                for action in affected_curation.actions:
                    if action.associated_id_sets is None:
                        new_actions.append(action)
                    else:

                        updated_assoc_id_set = self._drop_id_from_associated_id_sets(
                            id_to_drop=idx, associated_id_sets=action.associated_id_sets
                        )
                        if len(updated_assoc_id_set) == 0:
                            logger.warning(
                                "curation id %s has had all linking target ids removed by a global action, and will be"
                                " ignored. Parser name: %s",
                                affected_curation._id,
                                self.parser_name,
                            )
                            continue
                        if len(updated_assoc_id_set) < len(action.associated_id_sets):
                            logger.info(
                                "curation found with ids that have been removed via a global action. These will be filtered"
                                " from the curation action. Parser name: %s, new ids: %s, curation id: %s",
                                self.parser_name,
                                updated_assoc_id_set,
                                affected_curation._id,
                            )
                        new_actions.append(
                            dataclasses.replace(action, associated_id_sets=updated_assoc_id_set)
                        )
                if len(new_actions) > 0:
                    new_curation = dataclasses.replace(
                        affected_curation, actions=tuple(new_actions)
                    )
                    self.curations.add(new_curation)
                    self._curations_by_id[idx].add(new_curation)
                else:
                    logger.info(
                        "curation no longer has any relevant actions, and will be discarded"
                        " Parser name: %s, curation id: %s",
                        self.parser_name,
                        affected_curation._id,
                    )
                self.curations.remove(affected_curation)
                self._curations_by_id[idx].remove(affected_curation)

    def analyse_conflicts_in_curations(
        self, curations: Set[Curation]
    ) -> Tuple[Set[Curation], List[Set[Curation]]]:
        """
        Check to see if a list of curations contain conflicts.

        Conflicts can occur if two or more curations normalise to the same NormalisedSynonymStr,
        but refer to different AssociatedIdSets, and one of their actions is attempting to add
        a SynonymTerm to the database. This would create an ambiguity over which AssociatedIdSets
        is appropriate for the normalised term

        :param curations:
        :return: safe curations set, list of conflicting curations sets
        """
        curations_by_term_norm = defaultdict(set)
        conflicts = []
        safe = set()

        for curation in curations:
            if curation.source_term is not None:
                # inherited curations cannot conflict as they use term norm of source term
                safe.add(curation)
            else:
                curations_by_term_norm[curation.curated_synonym_norm(self.entity_class)].add(
                    curation
                )

        for term_norm, potentially_conflicting_curations in curations_by_term_norm.items():
            conflicting_id_sets = set()
            curations_by_assoc_id_set = {}

            for curation in potentially_conflicting_curations:
                for action in curation.actions:
                    if (
                        action.behaviour is SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING
                        or action.behaviour is SynonymTermBehaviour.ADD_FOR_LINKING_ONLY
                    ):
                        conflicting_id_sets.add(action.associated_id_sets)
                        curations_by_assoc_id_set[action.associated_id_sets] = curation

            if len(conflicting_id_sets) > 1:
                conflicts.append(potentially_conflicting_curations)
            else:
                safe.update(potentially_conflicting_curations)
        return safe, conflicts

    def _process_curation(self, curation: Curation) -> Optional[Curation]:
        term_norm = curation.term_norm_for_linking(self.entity_class)
        for action in curation.actions:
            if action.behaviour is SynonymTermBehaviour.IGNORE:
                logger.debug("ignoring unwanted curation: %s for %s", curation, self.parser_name)
                return None
            elif action.behaviour is SynonymTermBehaviour.INHERIT_FROM_SOURCE_TERM:
                logger.debug(
                    "action inherits linking behaviour from %s for %s",
                    curation.source_term,
                    self.parser_name,
                )
                if term_norm not in self._terms_by_term_norm:
                    logger.warning(
                        "curation %s is has no linking target in the synonym database, and will be ignored",
                        curation,
                    )
                    return None
                else:
                    return curation
            elif action.behaviour is SynonymTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING:
                self._drop_synonym_term(term_norm)
                return None

            assert action.associated_id_sets is not None
            if action.behaviour is SynonymTermBehaviour.DROP_ID_SET_FROM_SYNONYM_TERM:
                self._drop_id_set_from_synonym_term(
                    action.associated_id_sets,
                    term_norm=term_norm,
                )
            elif (
                action.behaviour is SynonymTermBehaviour.ADD_FOR_LINKING_ONLY
                or action.behaviour is SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING
            ):
                self._attempt_to_add_database_entry_for_curation(
                    curation_associated_id_set=action.associated_id_sets,
                    curated_synonym=curation.curated_synonym,
                    curation_term_norm=term_norm,
                )
                return curation
            else:
                raise ValueError(f"unknown behaviour for parser {self.parser_name}, {action}")
        return None

    def _process_global_actions(self) -> Set[str]:
        dropped_ids: Set[str] = set()
        if self.global_actions is None:
            return dropped_ids

        for action in self.global_actions.parser_behaviour(self.parser_name):
            if action.behaviour is ParserBehaviour.DROP_IDS_FROM_PARSER:
                ids = action.parser_to_target_id_mappings[self.parser_name]
                terms_modified, terms_dropped = 0, 0
                for idx in ids:
                    (
                        terms_modified_this_id,
                        terms_dropped_this_id,
                    ) = self._drop_id_from_all_synonym_terms(idx)
                    terms_modified += terms_modified_this_id
                    terms_dropped += terms_dropped_this_id
                    if terms_modified_this_id == 0 and terms_dropped_this_id == 0:
                        logger.warning("failed to drop %s from %s", idx, self.parser_name)
                    else:
                        dropped_ids.add(idx)
                        logger.debug(
                            "dropped ID %s from %s. SynonymTerm modified count: %s, SynonymTerm dropped count: %s",
                            idx,
                            self.parser_name,
                            terms_modified_this_id,
                            terms_dropped_this_id,
                        )
                    self._drop_id_from_curation(idx=idx)

            else:
                raise ValueError(f"unknown behaviour for parser {self.parser_name}, {action}")
        return dropped_ids

    def _drop_id_set_from_synonym_term(
        self, equivalent_id_sets_to_drop: AssociatedIdSets, term_norm: NormalisedSynonymStr
    ):
        """Remove all the id sets in equivalent_id_sets_to_drop from a :class:`.SynonymTerm`.

        :param equivalent_id_sets_to_drop: ids that should be removed from the :class:`.SynonymTerm`
        :param term_norm: key of term to remove
        :return:
        """
        target_term_to_modify = self._terms_by_term_norm[term_norm]
        for equiv_id_set_to_drop in equivalent_id_sets_to_drop:
            if equiv_id_set_to_drop in target_term_to_modify.associated_id_sets:
                drop_equivalent_id_set_from_synonym_term_result = (
                    self._drop_equivalent_id_set_from_synonym_term(term_norm, equiv_id_set_to_drop)
                )
                if (
                    drop_equivalent_id_set_from_synonym_term_result
                    is CurationModificationResult.ID_SET_MODIFIED
                ):
                    logger.debug(
                        "dropped an EquivalentIdSet containing an id from %s for key %s for %s",
                        equivalent_id_sets_to_drop,
                        term_norm,
                        self.parser_name,
                    )
                else:
                    logger.debug(
                        "dropped a SynonymTerm containing an id from %s for key %s for %s",
                        equivalent_id_sets_to_drop,
                        term_norm,
                        self.parser_name,
                    )
            else:
                logger.warning(
                    "%s was asked to remove ids %s from a SynonymTerm (key: <%s>), but the ids were not found on this term",
                    self.parser_name,
                    equiv_id_set_to_drop,
                    term_norm,
                )

    def _attempt_to_add_database_entry_for_curation(
        self,
        curation_term_norm: NormalisedSynonymStr,
        curation_associated_id_set: AssociatedIdSets,
        curated_synonym: str,
    ) -> Literal[
        CurationModificationResult.SYNONYM_TERM_ADDED, CurationModificationResult.NO_ACTION
    ]:
        """
        Create a new :class:`~kazu.data.data.SynonymTerm` for the database, or return an existing
        matching one if already present.

        Notes:

        Creating new AssociatedIdSets should be avoided wherever possible, as this can lead to
        inconsistencies in entity linking.

        If a term_norm already exists in self._terms_by_term_norm that matches 'curation_term_norm',
        this method will check to see if the 'curation_associated_id_set' is a subset of the existing term.
        If so, no action will be taken. If it is not a subset, an exception will be thrown, as adding it
        will cause irregularities in the database.

        If the term_norm does not exist, this method will try to find another :class:`.AssociatedIdSets` set
        that is a superset of 'curation_associated_id_set'. If no appropriate :class:`.AssociatedIdSets` can be
        found, a new one will be created


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

        if maybe_existing_synonym_term is not None:
            if curation_associated_id_set.issubset(maybe_existing_synonym_term.associated_id_sets):
                log_formatting_dict[
                    "existing_id_set"
                ] = maybe_existing_synonym_term.associated_id_sets
                logger.debug(
                    log_prefix
                    + " but term_norm <%(term_norm)s> already exists in synonym database."
                    + "since this SynonymTerm includes all ids in id_set, no action is required. %(existing_id_set)s",
                    log_formatting_dict,
                )
                return CurationModificationResult.NO_ACTION
            else:
                # check equiv ids on term is a superset
                all_matched = [False for _ in range(len(curation_associated_id_set))]
                for i, curation_equiv_id_set in enumerate(curation_associated_id_set):
                    for term_equiv_id_set in maybe_existing_synonym_term.associated_id_sets:
                        if curation_equiv_id_set.ids_and_source.issubset(
                            term_equiv_id_set.ids_and_source
                        ):
                            all_matched[i] = True
                            break
                    else:
                        break
                if all(all_matched):
                    log_formatting_dict[
                        "existing_id_set"
                    ] = maybe_existing_synonym_term.associated_id_sets
                    logger.debug(
                        log_prefix
                        + " but term_norm <%(term_norm)s> already exists in synonym database."
                        + "the AssociatedIdSet is a superset of the curation AssociatedIdSet. Therefore no action is required. %(existing_id_set)s",
                        log_formatting_dict,
                    )
                    return CurationModificationResult.NO_ACTION
                else:
                    formatted_log_prefix = log_prefix % log_formatting_dict
                    raise CurationException(
                        f"{formatted_log_prefix} but term_norm <{curation_term_norm}> already exists in synonym database, and its\n"
                        f"associated_id_sets don't contain all the ids in id_set. Creating a new\n"
                        f"SynonymTerm would override an existing entry, resulting in inconsistencies.\n"
                        f"This can happen if a synonym appears twice in the underlying ontology,\n"
                        f"with multiple identifiers attached\n"
                        f"Possible mitigations:\n"
                        f"1) use a SynonymTermAction to drop the existing SynonymTerm from the database first.\n"
                        f"2) change the target id set of the curation to match the existing entry\n"
                        f"\t(i.e. {maybe_existing_synonym_term.associated_id_sets}\n"
                        f"3) Change the string normalizer function to generate unique term_norms\n"
                    )

        # see if we've already had to group all the ids in this id_set in some way for a different synonym
        # we need the sort as we want to try to match to the smallest instance of AssociatedIdSets first.
        # This is because this is the least ambiguous - if we don't sort, we're potentially matching to
        # a larger, more ambiguous one than we need, and are potentially creating a disambiguation problem
        # where none exists
        set_of_associated_id_sets = set()
        for equiv_id_set in curation_associated_id_set:
            for idx in equiv_id_set.ids:
                maybe_terms = self._terms_by_id.get(idx)
                if maybe_terms is None:
                    formatted_log_prefix = log_prefix % log_formatting_dict
                    raise CurationException(
                        f"{formatted_log_prefix} but could not find {idx} for this parser"
                    )

                for term in maybe_terms:
                    if curation_associated_id_set.issubset(term.associated_id_sets):
                        set_of_associated_id_sets.add(term.associated_id_sets)

        if len(set_of_associated_id_sets) > 0:
            logger.debug(
                "using smallest AssociatedIDSet that matches all IDs for new SynonymTerm: %s",
                curation_associated_id_set,
            )
            associated_id_set_for_new_synonym_term = (
                select_smallest_associated_id_set_by_equiv_id_set_size_and_id_count(
                    set_of_associated_id_sets
                )
            )
        else:
            # something to be careful about here: we assume that if no appropriate AssociatedIdSet can be
            # reused, we need to create a new one. If one cannot be found, we 'assume' that the input
            # id_sets must relate to different concepts (i.e. - we create a new equivalent ID set for each
            # id, which must later be disambiguated). This assumption may be inappropriate in cases. This is
            # best avoided by having the curation contain as few IDs as possible, such that the chances
            # that an existing AssociatedIdSet can be reused are higher.
            logger.debug(
                "no appropriate AssociatedIdSets exist for the set %s, so a new one will be created",
                curation_associated_id_set,
            )
            associated_id_set_for_new_synonym_term = curation_associated_id_set

        is_symbolic = StringNormalizer.classify_symbolic(curated_synonym, self.entity_class)
        new_term = SynonymTerm(
            term_norm=curation_term_norm,
            terms=frozenset((curated_synonym,)),
            is_symbolic=is_symbolic,
            mapping_types=frozenset(("kazu_curated",)),
            associated_id_sets=associated_id_set_for_new_synonym_term,
            parser_name=self.parser_name,
            aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
        )
        return self._update_term_lookups(new_term, False)


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
        curations: Optional[List[Curation]] = None,
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
        :param curations: Curations to apply to the parser
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
        self.curations = curations
        self.global_actions = global_actions
        self.parsed_dataframe: Optional[pd.DataFrame] = None
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()

    def find_kb(self, string: str) -> str:
        """
        split an IDX somehow to find the ontology SOURCE reference

        :param string: the IDX string to process
        :return:
        """
        raise NotImplementedError()

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
            memory
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
    ) -> Tuple[Optional[List[Curation]], Set[SynonymTerm]]:
        if self.curations is None and self.synonym_generator is not None:
            logger.warning(
                "%s is configured to use synonym generators. This may result in noisy NER performance.",
                self.name,
            )
            (
                original_curations,
                generated_curations,
            ) = self.generate_curations_from_synonym_generators(terms)
            curations = original_curations + generated_curations
        elif self.curations is None and self.synonym_generator is None:
            logger.warning(
                "%s is configured to use raw ontology synonyms. This may result in noisy NER performance.",
                self.name,
            )
            curations = []
            for term in terms:
                curations.extend(self.synonym_term_to_putative_curation(term))
        else:
            assert self.curations is not None
            logger.info(
                "%s is configured to use curations. Synonym generation will be ignored",
                self.name,
            )
            curations = self.curations

        curation_processor = CurationProcessor(
            global_actions=self.global_actions,
            curations=curations,
            parser_name=self.name,
            entity_class=self.entity_class,
            synonym_terms=terms,
        )
        return curation_processor.export_ner_curations_and_final_terms()

    @kazu_disk_cache.memoize(ignore={0})
    def export_synonym_terms(self, parser_name: str) -> Set[SynonymTerm]:
        """Export :class:`.SynonymTerm` from the parser.

        :param parser_name: name of this parser. Required for correct operation of cache
            (Note, we cannot pass self to the disk cache as the constructor consumes too much
            memory
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
            generated_synonym_data.difference_update(original_synonym_data)
        logger.info(
            f"{len(original_synonym_data)} original synonyms and {len(generated_synonym_data)} generated synonyms produced"
        )
        return original_synonym_data, generated_synonym_data

    def generate_curations_from_synonym_generators(
        self, synonym_terms: Set[SynonymTerm]
    ) -> Tuple[List[Curation], List[Curation]]:
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

    def synonym_term_to_putative_curation(self, term: SynonymTerm) -> Iterable[Curation]:
        """
        When curations are not provided, this converts SynonymTerm's from the original
        ontology source and/or generated ones into curations, so they can be used
        for dictionary based NER

        :param term:
        :return:
        """
        for term_str in term.terms:
            if term.original_term is None:
                behaviour = SynonymTermBehaviour.ADD_FOR_NER_AND_LINKING
                associated_id_sets = term.associated_id_sets
            else:
                behaviour = SynonymTermBehaviour.INHERIT_FROM_SOURCE_TERM
                associated_id_sets = None
            action = SynonymTermAction(
                associated_id_sets=associated_id_sets,
                behaviour=behaviour,
            )
            curation = Curation(
                curated_synonym=term_str,
                mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                case_sensitive=StringNormalizer.classify_symbolic(term_str, self.entity_class),
                actions=tuple([action]),
                source_term=term.original_term,
            )
            yield curation

    def populate_databases(
        self, force: bool = False, return_ner_curations: bool = False
    ) -> Optional[List[Curation]]:
        """Populate the databases with the results of the parser.

        Also calculates the term norms associated with
        any curations (if provided) which can then be used for Dictionary based NER

        :param force: do not use the cache for the ontology parser
        :param return_ner_curations: should ner curations be returned?
        :return: curations if required
        """
        if self.name in self.synonym_db.loaded_parsers and not force and not return_ner_curations:
            logger.debug("parser %s already loaded.", self.name)
            return None

        cache_key = f"{self.name}.populate_databases"

        @kazu_disk_cache.memoize(name=cache_key)
        def _populate_databases():
            logger.info("populating database for %s from source", self.name)
            metadata = self.export_metadata(self.name)
            # metadata db needs to be populated before call to export_synonym_terms
            self.metadata_db.add_parser(self.name, metadata)
            intermediate_synonym_terms = self.export_synonym_terms(self.name)
            maybe_ner_curations, final_syn_terms = self.process_curations(
                intermediate_synonym_terms
            )
            self.parsed_dataframe = None  # clear the reference to save memory

            self.synonym_db.add(self.name, final_syn_terms)
            return maybe_ner_curations, metadata, final_syn_terms

        if force:
            kazu_disk_cache.delete(self.export_metadata.__cache_key__(self, self.name))
            kazu_disk_cache.delete(self.export_synonym_terms.__cache_key__(self, self.name))
            kazu_disk_cache.delete(_populate_databases.__cache_key__())

        maybe_ner_curations, metadata, final_syn_terms = _populate_databases()
        if self.name not in self.synonym_db.loaded_parsers or force:
            logger.info("populating database for %s from cache", self.name)
            self.metadata_db.add_parser(self.name, metadata)
            self.synonym_db.add(self.name, final_syn_terms)
        if return_ner_curations:
            return maybe_ner_curations
        else:
            return None

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

        Any 'extra' columns will be added to the :class:`~kazu.modelling.database.in_memory_db.MetadataDatabase` as metadata fields for the
        given id in the relevant ontology.
        """
        raise NotImplementedError()
