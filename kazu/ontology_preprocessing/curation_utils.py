import dataclasses
import json
import logging
from collections import defaultdict, Counter
from collections.abc import Iterable
from enum import auto
from typing import Optional, Literal, Any

from kazu.data.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    SynonymTerm,
    CuratedTerm,
    ParserBehaviour,
    CuratedTermBehaviour,
    AssociatedIdSets,
    GlobalParserActions,
    AutoNameEnum,
    MentionForm,
    MentionConfidence,
)
from kazu.database.in_memory_db import (
    NormalisedSynonymStr,
    Idx,
)
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import (
    PathLike,
    as_path,
)


logger = logging.getLogger(__name__)


class CurationError(Exception):
    pass


def load_curated_terms(
    path: PathLike,
) -> set[CuratedTerm]:
    """Load :class:`kazu.data.data.CuratedTerm`\\ s from a file path.

    :param path: path to json lines file that map to :class:`kazu.data.data.CuratedTerm`
    :return:
    """
    curations_path = as_path(path)
    if curations_path.exists():
        with curations_path.open(mode="r") as jsonlf:
            curations = {CuratedTerm.from_json(line) for line in jsonlf}
    else:
        raise ValueError(f"curations do not exist at {path}")
    return curations


def dump_curated_terms(terms: Iterable[CuratedTerm], path: PathLike, force: bool = False) -> None:
    """Dump an iterable of :class:`kazu.data.data.CuratedTerm`\\s to the file system.

    :param terms: terms to dump
    :param path: path to json lines file that map to :class:`kazu.data.data.CuratedTerm`
    :param force: override existing file, if it exists
    :return:
    """
    curations_path = as_path(path)
    if curations_path.exists() and not force:
        raise ValueError(f"file already exists at {path}")
    else:

        with curations_path.open(mode="w") as jsonlf:
            for curation in terms:
                jsonlf.write(curation.to_json() + "\n")


def dump_curated_term_subsets(
    terms: Iterable[frozenset[CuratedTerm]], path: PathLike, force: bool = False
) -> None:
    """Dump an iterable of sets of :class:`kazu.data.data.CuratedTerm`\\s to the file
    system.

    :param terms: terms to dump
    :param path: path to json lines file that map to :class:`kazu.data.data.CuratedTerm`
    :param force: override existing file, if it exists
    :return:
    """
    curations_path = as_path(path)
    if curations_path.exists() and not force:
        raise ValueError(f"file already exists at {path}")

    with curations_path.open(mode="w") as jsonlf:
        for curation_set in terms:
            for curation in curation_set:
                jsonlf.write(curation.to_json() + "\n")
            jsonlf.write("\n\n")


def load_global_actions(
    path: PathLike,
) -> GlobalParserActions:
    """Load an instance of :class:`.GlobalParserActions` from a file path.

    :param path: path to a json serialised GlobalParserActions
    :return:
    """
    global_actions_path = as_path(path)
    if global_actions_path.exists():
        with global_actions_path.open(mode="r") as jsonf:
            global_actions = GlobalParserActions.from_json(json.load(jsonf))
    else:
        raise ValueError(f"global actions do not exist at {path}")
    return global_actions


@dataclasses.dataclass
class CurationSetIntegrityReport:
    #: Terms with no conflicts
    clean_curations: set[CuratedTerm]
    #: Terms that can be safely merged without affecting behaviour
    merged_curations: set[CuratedTerm]
    #: Terms that conflict on normalisation value
    normalisation_conflicts: set[frozenset[CuratedTerm]]
    #: Terms that conflict on case
    case_conflicts: set[frozenset[CuratedTerm]]


class CuratedTermConflictAnalyser:
    """Find and potentially fix conflicting behaviour in a set of
    :class:`kazu.data.data.CuratedTerm`\\s."""

    CLEAN_CURATIONS_FN = "clean_curations.jsonl"
    MERGED_CURATIONS_FN = "merged_curations.jsonl"
    NORM_CONFLICT_CURATIONS_FN = "normalisation_conflicts.jsonl"
    CASE_CONFLICT_CURATIONS_FN = "case_conflicts.jsonl"

    def __init__(self, entity_class: str, autofix: bool = False):
        """

        :param entity_class: entity class that this analyzer will handle
        :param autofix: Should any conflicts be automatically fixed, such that the
            behaviour is consistent within this set? Note that this does not guarantee that
            the optimal behaviour for a conflict is preserved.
        """
        self.entity_class = entity_class
        self.autofix = autofix

    def verify_curation_set_integrity(
        self, curations: set[CuratedTerm], path: Optional[PathLike] = None
    ) -> CurationSetIntegrityReport:
        """Verify that a set of terms has consistent behaviour.

        Conflicts can occur for the following reasons:

        1) If two or more curations normalise to the same string,
           but have different :class:`kazu.data.data.CuratedTermBehaviour`\\.

        2) If two or more curations normalise to the same value,
           but have different associated ID sets specified, such that one would
           override the other.

        3) If two or more curations have conflicting values for case sensitivity and
           :class:`kazu.data.data.MentionConfidence`\\. E.g. A case-insensitive curation
           cannot have a higher mention confidence value than a case-sensitive one for
           the same string.

        :param curations:
        :param path: if provided, write a report to this location of the conflicts that occur
        :return:
        :raises CurationError: if one or more curations produce multiple normalised values
        """

        (
            curations_by_term_norm,
            maybe_good_curations_by_syn_lower,
        ) = self._group_curations_and_check_for_normalisation_consistency_errors(curations)

        # note, this method updates the maybe_good_curations_by_syn_lower dict, so we
        # don't need to update clean_curations with merged_curations lower down
        (
            maybe_good_curations_by_syn_lower,
            merged_curations,
            normalisation_conflicts,
        ) = self._check_for_normalised_behaviour_conflicts_and_merge_if_possible(
            curations_by_term_norm, maybe_good_curations_by_syn_lower
        )

        case_conflicts, clean_curations = self._check_for_case_conflicts_across_curations(
            maybe_good_curations_by_syn_lower
        )

        if self.autofix:
            logger.info("Curations will be automatically fixed")
            clean_curations.update(self.autofix_curations(normalisation_conflicts))
            clean_curations.update(self.autofix_curations(case_conflicts))
            normalisation_conflicts = set()
            case_conflicts = set()

        if path is not None:
            self._write_integrity_report(
                case_conflicts, clean_curations, merged_curations, normalisation_conflicts, path
            )

        return CurationSetIntegrityReport(
            clean_curations=clean_curations,
            merged_curations=merged_curations,
            normalisation_conflicts=normalisation_conflicts,
            case_conflicts=case_conflicts,
        )

    def autofix_curations(
        self, curation_conflicts: set[frozenset[CuratedTerm]]
    ) -> set[CuratedTerm]:
        """Fix conflicts in curations by producing a new set of curations with
        consistent behaviour.

        This ensures that there are no conflicts regarding any of these properties:

        * the combination of case sensitivity and mention confidence
        * associated id set
        * curated term behaviour

        :param curation_conflicts:
        :return:
        """
        cleaned_curations = set()
        for conflicted_set in curation_conflicts:
            original_forms_by_term_norm: defaultdict[str, set[MentionForm]] = defaultdict(set)
            alt_forms_by_term_norm: defaultdict[str, set[MentionForm]] = defaultdict(set)
            case_sensitive = False
            form_string_lower_to_confidence = defaultdict(set)
            assoc_id_sets: set[EquivalentIdSet] = set()
            behaviours: set[CuratedTermBehaviour] = set()
            for curation in conflicted_set:
                term_norm = curation.term_norm_for_linking(self.entity_class)
                original_forms_by_term_norm[term_norm].update(curation.original_forms)
                alt_forms_by_term_norm[term_norm].update(curation.alternative_forms)
                behaviours.add(curation.behaviour)
                for form in curation.active_ner_forms():
                    form_string_lower_to_confidence[form.string.lower()].add(
                        form.mention_confidence
                    )
                    if form.case_sensitive:
                        case_sensitive = True
                if curation.associated_id_sets is not None:
                    assoc_id_sets.update(curation.associated_id_sets)

            if CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING in behaviours:
                chosen_behaviour = CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING
            elif CuratedTermBehaviour.ADD_FOR_LINKING_ONLY in behaviours:
                chosen_behaviour = CuratedTermBehaviour.ADD_FOR_LINKING_ONLY
            elif CuratedTermBehaviour.IGNORE in behaviours:
                chosen_behaviour = CuratedTermBehaviour.IGNORE
            else:
                chosen_behaviour = CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING

            for term_norm, mergeable_original_forms in original_forms_by_term_norm.items():
                cleaned_curations.add(
                    CuratedTerm(
                        behaviour=chosen_behaviour,
                        original_forms=frozenset(
                            dataclasses.replace(
                                form,
                                case_sensitive=case_sensitive,
                                mention_confidence=min(
                                    form_string_lower_to_confidence[form.string.lower()]
                                ),
                            )
                            for form in mergeable_original_forms
                        ),
                        alternative_forms=frozenset(
                            dataclasses.replace(
                                form,
                                case_sensitive=case_sensitive,
                                mention_confidence=min(
                                    form_string_lower_to_confidence[form.string.lower()]
                                ),
                            )
                            for form in alt_forms_by_term_norm.get(term_norm, set())
                        ),
                        associated_id_sets=frozenset(assoc_id_sets)
                        if len(assoc_id_sets) > 0
                        else None,
                    )
                )
        return cleaned_curations

    def _write_integrity_report(
        self,
        case_conflicts: set[frozenset[CuratedTerm]],
        clean_curations: set[CuratedTerm],
        merged_curations: set[CuratedTerm],
        normalisation_conflicts: set[frozenset[CuratedTerm]],
        path: PathLike,
    ) -> None:
        report_path = as_path(path)
        report_path.mkdir(parents=True, exist_ok=True)
        path_to_curations: dict[str, set[CuratedTerm]] = {
            self.CLEAN_CURATIONS_FN: clean_curations,
            self.MERGED_CURATIONS_FN: merged_curations,
        }
        for path_name, curation_data in path_to_curations.items():
            if curation_data:
                with report_path.joinpath(path_name).open(mode="w") as f:
                    for curation in curation_data:
                        f.write(curation.to_json() + "\n")
        path_to_curation_conflicts: dict[str, set[frozenset[CuratedTerm]]] = {
            self.NORM_CONFLICT_CURATIONS_FN: normalisation_conflicts,
            self.CASE_CONFLICT_CURATIONS_FN: case_conflicts,
        }
        for path_name, curation_conflict_data in path_to_curation_conflicts.items():
            if curation_conflict_data:
                with report_path.joinpath(path_name).open(mode="w") as f:
                    for curation_set in curation_conflict_data:
                        for curation in curation_set:
                            f.write(curation.to_json() + "\n")

    def _check_for_case_conflicts_across_curations(
        self, maybe_good_curations_by_syn_lower: defaultdict[str, set[CuratedTerm]]
    ) -> tuple[set[frozenset[CuratedTerm]], set[CuratedTerm]]:
        case_conflicts = set()
        clean_curations = set()
        for potential_conflict_set in list(maybe_good_curations_by_syn_lower.values()):

            if self._curation_set_has_case_conflicts(potential_conflict_set):
                # uh ho - we have multiple identical forms attached to different curations
                case_conflicts.add(frozenset(potential_conflict_set))
            else:
                clean_curations.update(potential_conflict_set)

        return case_conflicts, clean_curations

    def _check_for_normalised_behaviour_conflicts_and_merge_if_possible(
        self,
        curations_by_term_norm: defaultdict[str, set[CuratedTerm]],
        maybe_good_curations_by_syn_lower: defaultdict[str, set[CuratedTerm]],
    ) -> tuple[defaultdict[str, set[CuratedTerm]], set[CuratedTerm], set[frozenset[CuratedTerm]]]:
        """Find behaviour conflicts in the curation set indexed by term_norm, and remove
        them from the curation set indexed by syn_lower.

        If curations can be merged, without causing conflicts, they will be, resulting
        in a new curation and the destruction of the originals.

        This method modifies the maybe_good_curations_by_syn_lower dictionary, with
        conflicts and merged curations removed, and newly created curations (from
        merged) added.

        :param curations_by_term_norm:
        :param maybe_good_curations_by_syn_lower:
        :return: modified maybe_good_curations_by_syn_lower, a set of newly created
            merged curations and a set of sets of conflicts.
        """

        normalisation_conflicts = set()
        merged_curations = set()
        for term_norm, potentially_conflicting_curations in curations_by_term_norm.items():
            if len(potentially_conflicting_curations) == 1:
                # no conflict
                continue

            # uh ho we have a normalisation/behaviour/id set conflict. If we can't merge, report an error
            original_forms_merged: set[MentionForm] = set()
            generated_forms_merged: set[MentionForm] = set()
            associated_id_sets_this_term_norm: set[frozenset[EquivalentIdSet]] = set()
            comments = []
            behaviours = set()
            for conflicted_curation in potentially_conflicting_curations:

                behaviours.add(conflicted_curation.behaviour)
                original_forms_merged.update(conflicted_curation.original_forms)
                generated_forms_merged.update(conflicted_curation.alternative_forms)
                if conflicted_curation.associated_id_sets is not None:
                    associated_id_sets_this_term_norm.add(conflicted_curation.associated_id_sets)
                if conflicted_curation.comment is not None:
                    comments.append(conflicted_curation.comment)
                for form in conflicted_curation.all_forms():
                    # remove conflicted forms from the working 'good' list
                    maybe_good_curations_by_syn_lower[form.string.lower()].discard(
                        conflicted_curation
                    )

            if len(behaviours) > 1 or len(associated_id_sets_this_term_norm) > 1:
                # uh ho - behaviours/id set clash. Add to norm conflicts
                normalisation_conflicts.add(frozenset(potentially_conflicting_curations))
            else:
                # merge the curations
                merged_curation = CuratedTerm(
                    behaviour=next(iter(behaviours)),
                    original_forms=frozenset(original_forms_merged),
                    alternative_forms=frozenset(generated_forms_merged),
                    associated_id_sets=next(iter(associated_id_sets_this_term_norm))
                    if len(associated_id_sets_this_term_norm) == 1
                    else None,
                    comment="\n".join(comments) if len(comments) > 0 else None,
                )
                merged_curations.add(merged_curation)
                # update the maybe good curations with the merged curation data
                for form in merged_curation.active_ner_forms():
                    maybe_good_curations_by_syn_lower[form.string.lower()].add(merged_curation)
                logger.warning(
                    "duplicate curation set merged. term norm: %s, conflicts:\n%s",
                    term_norm,
                    potentially_conflicting_curations,
                )
        return maybe_good_curations_by_syn_lower, merged_curations, normalisation_conflicts

    def _group_curations_and_check_for_normalisation_consistency_errors(
        self, curations: set[CuratedTerm]
    ) -> tuple[defaultdict[str, set[CuratedTerm]], defaultdict[str, set[CuratedTerm]],]:
        curations_by_term_norm: defaultdict[str, set[CuratedTerm]] = defaultdict(set)
        maybe_good_curations_by_syn_lower: defaultdict[str, set[CuratedTerm]] = defaultdict(set)
        normalisation_errors = set()
        for curation in curations:
            term_norms_this_term = set(
                StringNormalizer.normalize(original_form.string, entity_class=self.entity_class)
                for original_form in curation.original_forms
            )
            if len(term_norms_this_term) > 1:
                normalisation_errors.add(curation)
            term_norm = next(iter(term_norms_this_term))
            curations_by_term_norm[term_norm].add(curation)
            for form in curation.all_forms():
                maybe_good_curations_by_syn_lower[form.string.lower()].add(curation)
        if normalisation_errors:
            # This should never be thrown by automatically generated curations
            raise CurationError(
                "One or more curations contain original forms that no longer normalise to the same value. This"
                " can happen if the implementation of the StringNormalizer has changed. Please correct the following"
                " curations:\n" + "\n".join(curation.to_json() for curation in normalisation_errors)
            )
        return curations_by_term_norm, maybe_good_curations_by_syn_lower

    def _curation_set_has_case_conflicts(self, curations: set[CuratedTerm]) -> bool:
        cs_conf_lookup = defaultdict(set)
        ci_conf_lookup = defaultdict(set)
        for curation in curations:
            for form in curation.active_ner_forms():
                if form.case_sensitive:
                    cs_conf_lookup[form.string].add(form.mention_confidence)
                else:
                    ci_conf_lookup[form.string.lower()].add(form.mention_confidence)

        for form_string, cs_confidences in cs_conf_lookup.items():
            ci_confidences: set[MentionConfidence] = ci_conf_lookup.get(form_string.lower(), set())
            if (
                len(ci_confidences) > 0
                and len(cs_confidences) > 0
                and min(cs_confidences) <= min(ci_confidences)
            ):
                return True
        return False

    def merge_human_and_auto_curations(
        self,
        human_curations: set[CuratedTerm],
        autocurations: set[CuratedTerm],
        path: Optional[PathLike],
    ) -> set[CuratedTerm]:
        """Merge a set of human curations with a set of automatically generated
        curations, preferring the human set where possible.

        :param human_curations:
        :param autocurations:
        :param path: if provided, report any curations with discrepancies (such as where
            an automatically generated curation contains more forms than the human
            equivalent). Also report any superfluous and obsolete human curations
        :return:
        """
        # normalisation errors should have already been checked with self.verify_curation_set_integrity before calling this

        obsolete_human_set = set()
        curations_with_discrepancies = set()
        superfluous_human_curations = human_curations.intersection(autocurations)
        effective_human_curations = human_curations.difference(autocurations)
        human_by_term_norm = {
            curation.term_norm_for_linking(self.entity_class): curation
            for curation in human_curations
        }
        default_by_term_norm = {
            curation.term_norm_for_linking(self.entity_class): curation
            for curation in autocurations
        }

        working_dict = {}
        for default_term_norm, default_curation in default_by_term_norm.items():

            maybe_human_curation = human_by_term_norm.pop(default_term_norm, None)
            if maybe_human_curation is None:
                working_dict[default_term_norm] = default_curation
            else:
                working_dict[default_term_norm] = maybe_human_curation
                # check the set of strings are equivalent. Otherwise new ones may have been created by syn generation
                if set(form.string for form in maybe_human_curation.original_forms) != set(
                    form.string for form in default_curation.original_forms
                ) or set(form.string for form in maybe_human_curation.alternative_forms) != set(
                    form.string for form in default_curation.alternative_forms
                ):
                    curations_with_discrepancies.add(
                        frozenset([maybe_human_curation, default_curation])
                    )

        # add any remaining to the obsolete set, unless they're additional
        for human_term_norm, human_curation in human_by_term_norm.items():
            if human_curation.additional_to_source:
                working_dict[human_term_norm] = human_curation
            else:
                obsolete_human_set.add(human_curation)

        if path is not None:
            path_ = as_path(path)
            if obsolete_human_set:
                dump_curated_terms(
                    obsolete_human_set,
                    path_.joinpath("obsolete_human_curations_set.jsonl"),
                    force=True,
                )
            if superfluous_human_curations:
                dump_curated_terms(
                    superfluous_human_curations,
                    path_.joinpath("superfluous_human_curations_set.jsonl"),
                    force=True,
                )
            if effective_human_curations:
                dump_curated_terms(
                    effective_human_curations,
                    path_.joinpath("effective_human_curations.jsonl"),
                    force=True,
                )
            if curations_with_discrepancies:
                dump_curated_term_subsets(
                    curations_with_discrepancies,
                    path_.joinpath("human_curations_with_form_discrepancies_set.jsonl"),
                    force=True,
                )

        return set(working_dict.values())


class CurationModificationResult(AutoNameEnum):
    ID_SET_MODIFIED = auto()
    SYNONYM_TERM_ADDED = auto()
    SYNONYM_TERM_DROPPED = auto()
    NO_ACTION = auto()


class CurationProcessor:
    """A CurationProcessor is responsible for modifying the set of
    :class:`.SynonymTerm`\\s produced by an
    :class:`kazu.ontology_preprocessing.base.OntologyParser` with any relevant
    :class:`.GlobalParserActions` and/or :class:`.CuratedTerm` associated with the
    parser.

    This class should be used before instances of :class:`.SynonymTerm`\\s are loaded into the
    internal database representation.
    """

    # curations are applied in the following order
    CURATION_APPLY_ORDER = (
        CuratedTermBehaviour.IGNORE,
        CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING,
        CuratedTermBehaviour.ADD_FOR_LINKING_ONLY,
        CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING,
    )
    _BEHAVIOUR_TO_ORDER_INDEX = {behav: i for i, behav in enumerate(CURATION_APPLY_ORDER)}

    def __init__(
        self,
        parser_name: str,
        entity_class: str,
        global_actions: Optional[GlobalParserActions],
        curations: list[CuratedTerm],
        synonym_terms: set[SynonymTerm],
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
        self._terms_by_term_norm: dict[NormalisedSynonymStr, SynonymTerm] = {}
        self._terms_by_id: defaultdict[Idx, set[SynonymTerm]] = defaultdict(set)
        for term in synonym_terms:
            self._update_term_lookups(term, False)
        self.curations = set(curations)
        self.dropped_keys: set[NormalisedSynonymStr] = set()

    @classmethod
    def curation_sort_key(cls, curated_term: CuratedTerm) -> tuple[int, bool]:
        """Determines the order curations are processed in.

        We use associated_id_sets as a key, so that any overrides will be processed
        after any original behaviours
        """
        return (
            cls._BEHAVIOUR_TO_ORDER_INDEX[curated_term.behaviour],
            curated_term.associated_id_sets is not None,
        )

    def _update_term_lookups(
        self, term: SynonymTerm, override: bool
    ) -> Literal[
        CurationModificationResult.SYNONYM_TERM_ADDED, CurationModificationResult.NO_ACTION
    ]:

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
        """Remove a synonym term from the database, so that it cannot be used as a
        linking target.

        :param synonym:
        :return:
        """
        try:
            term_to_remove = self._terms_by_term_norm.pop(synonym)
            self.dropped_keys.add(synonym)
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
            if synonym in self.dropped_keys:
                logger.debug(
                    "tried to drop %s from database, but key already dropped by another CuratedTerm for %s",
                    synonym,
                    self.parser_name,
                )
            else:
                logger.warning(
                    "tried to drop %s from database, but key doesn't exist for %s",
                    synonym,
                    self.parser_name,
                )

    def _drop_id_from_all_synonym_terms(
        self, id_to_drop: Idx
    ) -> Counter[
        Literal[
            CurationModificationResult.ID_SET_MODIFIED,
            CurationModificationResult.SYNONYM_TERM_DROPPED,
            CurationModificationResult.NO_ACTION,
        ]
    ]:
        """Remove a given id from all :class:`.SynonymTerm`\\ s.

        Drop any :class:`.SynonymTerm`\\ s with no remaining ID after removal.

        :param id_to_drop:
        :return:
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
        """Modifies or drops a :class:`.SynonymTerm` after a :class:`.AssociatedIdSets`
        has changed.

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
    ) -> tuple[list[CuratedTerm], set[SynonymTerm]]:
        """Perform any updates required to the synonym terms as specified in the
        curations/global actions.

        The returned :class:`.CuratedTerm`\\s can be used for Dictionary based NER, whereas the
        returned :class:`.SynonymTerm`\\s can be loaded into the internal database for linking.

        :return:
        """
        self._process_global_actions()
        return list(self._process_curations()), set(self._terms_by_term_norm.values())

    def _process_curations(self) -> Iterable[CuratedTerm]:
        for curation in sorted(self.curations, key=self.curation_sort_key):
            curation = self._process_curation_action(curation)
            yield curation

    def _process_curation_action(self, curation: CuratedTerm) -> CuratedTerm:

        if curation.behaviour is CuratedTermBehaviour.IGNORE:
            logger.debug("curation ignored: %s for %s", curation, self.parser_name)
        elif curation.behaviour is CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING:
            self._drop_synonym_term(curation.term_norm_for_linking(self.entity_class))
        elif curation.behaviour is CuratedTermBehaviour.ADD_FOR_LINKING_ONLY:
            self._attempt_to_add_database_entry_for_curated_term(
                curation,
            )
        elif curation.behaviour is CuratedTermBehaviour.ADD_FOR_NER_AND_LINKING:
            self._attempt_to_add_database_entry_for_curated_term(curation)
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

    def _attempt_to_add_database_entry_for_curated_term(
        self,
        curated_term: CuratedTerm,
    ) -> Literal[
        CurationModificationResult.SYNONYM_TERM_ADDED, CurationModificationResult.NO_ACTION
    ]:
        """Create a new :class:`~kazu.data.data.SynonymTerm` for the database, or return
        an existing matching one if already present.

        Notes:

        If a term_norm already exists in self._terms_by_term_norm that matches 'curated_term.term_norm_for_linking',
        this method will check to see if the 'curation_associated_id_set' matches the existing terms
        :class:`.AssociatedIdSets`\\.

        If so, no action will be taken. If not, a warning will be logged as adding it
        will cause irregularities in the database.

        If the term_norm does not exist, this method will create a new :class:`~kazu.data.data.SynonymTerm`
        with the provided :class:`.AssociatedIdSets`\\.

        :param curated_term:
        :return:
        """
        log_prefix = "%(parser_name)s attempting to create synonym term for <%(synonym)s> term_norm: <%(term_norm)s> IDs: %(ids)s}"
        term_norm = curated_term.term_norm_for_linking(self.entity_class)
        log_formatting_dict: dict[str, Any] = {
            "parser_name": self.parser_name,
            "term_norm": term_norm,
            "curated_term": curated_term,
        }

        # look up the term norm in the db
        maybe_existing_synonym_term = self._terms_by_term_norm.get(term_norm)
        curation_associated_id_set = curated_term.associated_id_sets
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
                curated_term,
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
            is_symbolic = any(
                StringNormalizer.classify_symbolic(form.string, self.entity_class)
                for form in curated_term.original_forms
            )
            new_term = SynonymTerm(
                term_norm=term_norm,
                terms=frozenset(term.string for term in curated_term.original_forms),
                is_symbolic=is_symbolic,
                mapping_types=frozenset(("kazu_curated",)),
                associated_id_sets=curation_associated_id_set,
                parser_name=self.parser_name,
                aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
            )
            return self._update_term_lookups(new_term, True)
        else:
            return CurationModificationResult.NO_ACTION
