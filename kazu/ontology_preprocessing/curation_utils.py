import dataclasses
import json
import logging
import shutil
from collections import defaultdict, Counter
from collections.abc import Iterable
from enum import auto
from typing import Optional, Literal, Any

from kazu.data import (
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
    LinkingCandidate,
    OntologyStringResource,
    ParserBehaviour,
    OntologyStringBehaviour,
    AssociatedIdSets,
    GlobalParserActions,
    AutoNameEnum,
    Synonym,
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


def load_ontology_string_resources(
    path: PathLike,
) -> set[OntologyStringResource]:
    """Load :class:`kazu.data.OntologyStringResource`\\ s from a file path or directory.

    :param path: path to a jsonl file or directory of jsonl files that map to :class:`kazu.data.OntologyStringResource`
    :return:
    """
    resources_path = as_path(path)
    resources: set[OntologyStringResource] = set()
    if not resources_path.exists():
        raise ValueError(f"resources file does not exist at: {path}")

    files = resources_path.iterdir() if resources_path.is_dir() else (resources_path,)
    for f in files:
        with f.open(mode="r") as jsonlf:
            resources.update(OntologyStringResource.from_json(line) for line in jsonlf)

    return resources


def _resource_sort_reduce(resource: OntologyStringResource) -> tuple[int, str]:
    return min((len(syn.text), syn.text) for syn in resource.original_synonyms)


def batch(
    iterable: Iterable[OntologyStringResource], n: int = 1
) -> Iterable[list[OntologyStringResource]]:
    lst = sorted(iterable, key=_resource_sort_reduce)
    length = len(lst)
    for ndx in range(0, length, n):
        yield lst[ndx : ndx + n]


def dump_ontology_string_resources(
    resources: Iterable[OntologyStringResource],
    path: PathLike,
    force: bool = False,
    split_at: int = 10000,
) -> None:
    """Dump an iterable of :class:`kazu.data.OntologyStringResource`\\s to the file
    system.

    :param resources: resources to dump
    :param path: path to a directory of json lines files that map to :class:`kazu.data.OntologyStringResource`
    :param force: override existing directory, if it exists
    :param split_at: number of lines per partition
    :return:
    """
    resources_path = as_path(path)
    if resources_path.exists():
        if not resources_path.is_dir():
            raise ValueError(f"file already exists at {path} and is not a directory")
        elif not force:
            raise ValueError(f"directory already exists at {path}")
        else:
            shutil.rmtree(resources_path)

    resources_path.mkdir(parents=True)
    for i, resources_partition in enumerate(batch(resources, n=split_at)):
        with resources_path.joinpath(f"{i}.jsonl").open(mode="w") as jsonlf:
            for resource in resources_partition:
                jsonlf.write(resource.to_json() + "\n")


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
            global_actions = GlobalParserActions.from_dict(json.load(jsonf))
    else:
        raise ValueError(f"global actions do not exist at {path}")
    return global_actions


_CURATION_REPORT_FILENAME = "_curation_report"
_ONTOLOGY_MERGE_REPORT_DIR = "_ontology_merge_report"
CLEAN_RESOURCES_FN = "clean_resources.jsonl"
MERGED_RESOURCES_FN = "merged_resources.jsonl"
NORM_CONFLICT_RESOURCES_FN = "normalisation_conflicts.jsonl"
OBSOLETE_RESOURCES_FN = "obsolete_human_curations_set.jsonl"
SUPERFLUOUS_RESOURCES_FN = "superfluous_human_curations_set.jsonl"
EFFECTIVE_RESOURCES_FN = "effective_human_curations.jsonl"


@dataclasses.dataclass
class OntologyResourceSetConflictReport:
    #: Resources with no conflicts
    clean_resources: set[OntologyStringResource]
    #: Resources that can be safely merged without affecting :class:`.OntologyStringBehaviour`. However,
    #: may still conflict on :class:`.MentionConfidence` and/or case sensitivity
    merged_resources: set[OntologyStringResource]
    #: Resources that conflict on normalisation value
    normalisation_conflicts: set[frozenset[OntologyStringResource]]
    #: Resources that conflict on case
    case_conflicts: set[frozenset[OntologyStringResource]]

    def write_normalisation_conflict_report(
        self,
        path: PathLike,
    ) -> None:
        report_path = as_path(path)
        report_path.mkdir(parents=True, exist_ok=True)
        path_to_resources: dict[str, set[OntologyStringResource]] = {
            CLEAN_RESOURCES_FN: self.clean_resources,
            MERGED_RESOURCES_FN: self.merged_resources,
        }
        for path_name, resource_data in path_to_resources.items():
            if resource_data:
                with report_path.joinpath(path_name).open(mode="w") as f:
                    for resource in resource_data:
                        f.write(resource.to_json() + "\n")

        if self.normalisation_conflicts:
            with report_path.joinpath(NORM_CONFLICT_RESOURCES_FN).open(mode="w") as f:
                for resource_set in self.normalisation_conflicts:
                    for resource in resource_set:
                        f.write(resource.to_json() + "\n")


@dataclasses.dataclass
class OntologyResourceSetMergeReport:
    #: human resources that no longer match any strings in the underlying parser data
    obsolete_resources: set[OntologyStringResource]
    #: human and autogenerated resources that are actively in use
    effective_resources: set[OntologyStringResource]
    #: human resources that match any strings in the underlying parser data, but result in the same behaviour
    #: as the autogenerated resource, and therefore can be eliminated to reduce the management burden of human
    #: resources
    superfluous_resources: set[OntologyStringResource]
    #: a tuple of human resource/autogenerated resource that only partially match on their original forms,
    #: suggesting that the underlying parser data has changed in some way since the last upgrade. In this
    #: scenario, it's recommended to replace the human form with the new autogenerated version for consistency
    resources_with_discrepancies: set[tuple[OntologyStringResource, OntologyStringResource]]

    def write_ontology_merge_report(self, path: PathLike) -> None:
        report_path = as_path(path)
        filename_to_resources: dict[str, set[OntologyStringResource]] = {
            OBSOLETE_RESOURCES_FN: self.obsolete_resources,
            SUPERFLUOUS_RESOURCES_FN: self.superfluous_resources,
            EFFECTIVE_RESOURCES_FN: self.effective_resources,
        }
        for filename, resources in filename_to_resources.items():
            if resources:
                dump_ontology_string_resources(
                    resources,
                    report_path.joinpath(filename),
                    force=True,
                )

        if self.resources_with_discrepancies:
            with report_path.joinpath("human_curations_with_synonym_discrepancies_set.jsonl").open(
                mode="w"
            ) as jsonlf:
                for (
                    human_curation,
                    default_resource,
                ) in self.resources_with_discrepancies:
                    jsonlf.write("HUMAN:" + "\n")
                    jsonlf.write(human_curation.to_json() + "\n")
                    jsonlf.write("DEFAULT:" + "\n")
                    jsonlf.write(default_resource.to_json() + "\n")
                    jsonlf.write("\n\n")


@dataclasses.dataclass
class OntologyResourceSetCompleteReport:
    """Describes the state of :class:`kazu.data.OntologyStringResource`\\s configured
    for a given :class:`kazu.ontology_preprocessing.base.OntologyParser`, such as any
    conflicts in how string matching is configured, or discrepancies between resources
    produced by :class:`kazu.ontology_preprocessing.autocuration.AutoCurator` and their
    human curated overrrides."""

    #: parser linking candidates before they are processed with :class:`.OntologyStringResource`\\s
    intermediate_linking_candidates: set[LinkingCandidate]
    #: report of resource conflict resolution after merging of autogenerated resources and human curations
    final_conflict_report: OntologyResourceSetConflictReport
    #: report of resource conflict in human curation set, if available
    human_conflict_report: Optional[OntologyResourceSetConflictReport] = None
    #: report of result of merging human and autogenerated resources, if available
    merge_report: Optional[OntologyResourceSetMergeReport] = None

    def write_reports_for_parser(self, path: PathLike, parser_name: str) -> None:
        curation_report_path = as_path(path).joinpath(f"{parser_name}{_CURATION_REPORT_FILENAME}")
        if curation_report_path.exists():
            shutil.rmtree(curation_report_path)

        curation_report_path.mkdir()

        human_and_auto_generated_resources_conflict_report_path = curation_report_path.joinpath(
            "active_resources_conflict_report"
        )
        human_and_auto_generated_resources_conflict_report_path.mkdir()
        self.final_conflict_report.write_normalisation_conflict_report(
            human_and_auto_generated_resources_conflict_report_path
        )

        if self.human_conflict_report:
            human_curation_set_report_path = curation_report_path.joinpath(
                "human_curation_conflict_report"
            )
            human_curation_set_report_path.mkdir()
            self.human_conflict_report.write_normalisation_conflict_report(
                human_curation_set_report_path
            )

        if self.merge_report:
            logger.info(
                "%s reporting discrepancies in human curation set and auto generated resources",
                parser_name,
            )
            merge_report_path = as_path(path).joinpath(f"{parser_name}{_ONTOLOGY_MERGE_REPORT_DIR}")
            merge_report_path.mkdir()
            self.merge_report.write_ontology_merge_report(merge_report_path)


class AutofixStrategy(AutoNameEnum):
    OPTIMISTIC = auto()
    PESSIMISTIC = auto()
    NONE = auto()


class OntologyStringConflictAnalyser:
    """Find and potentially fix conflicting behaviour in a set of
    :class:`kazu.data.OntologyStringResource`\\s."""

    def __init__(self, entity_class: str, autofix: AutofixStrategy = AutofixStrategy.NONE):
        """

        :param entity_class: entity class that this analyzer will handle
        :param autofix: Should any conflicts be automatically fixed, such that the
            behaviour is consistent within this set? Note that this does not guarantee that
            the optimal behaviour for a conflict is preserved.
        """
        self.entity_class = entity_class
        self.autofix = autofix

    def verify_resource_set_integrity(
        self, resources: set[OntologyStringResource]
    ) -> OntologyResourceSetConflictReport:
        """Verify that a set of resources has consistent behaviour.

        Conflicts can occur for the following reasons:

        1) If two or more resources normalise to the same string,
           but have different :class:`kazu.data.OntologyStringBehaviour`\\.

        2) If two or more resources normalise to the same value,
           but have different associated ID sets specified, such that one would
           override the other.

        3) If two or more resources have conflicting values for case sensitivity and
           :class:`kazu.data.MentionConfidence`\\. E.g. A case-insensitive resource
           cannot have a higher mention confidence value than a case-sensitive one for
           the same string.

        :param resources:
        :return:
        :raises CurationError: if one or more resources produce multiple normalised values
        """

        resources = set(resources)
        (
            merged_resources,
            eliminated_resources,
            normalisation_conflicts,
        ) = self.check_for_normalised_behaviour_conflicts_and_merge_if_possible(resources)

        resources.difference_update(eliminated_resources)
        resources.update(merged_resources)
        # remove the conflicts from the working set
        for conflict_set in normalisation_conflicts:
            resources.difference_update(conflict_set)
        if self.autofix is not AutofixStrategy.NONE:

            logger.info("Resources will be automatically fixed")
            autofixed_resources_from_conflicts = self.autofix_resources(normalisation_conflicts)
            # add the fixed resources to the merged set
            merged_resources.update(autofixed_resources_from_conflicts)
            resources.update(autofixed_resources_from_conflicts)
            case_conflicts, clean_resources = self.check_for_case_conflicts_across_resources(
                resources
            )
            clean_resources.update(self.autofix_resources(case_conflicts))
            normalisation_conflicts = set()
            case_conflicts = set()
        else:
            case_conflicts, clean_resources = self.check_for_case_conflicts_across_resources(
                resources
            )

        return OntologyResourceSetConflictReport(
            clean_resources=clean_resources,
            merged_resources=merged_resources,
            normalisation_conflicts=normalisation_conflicts,
            case_conflicts=case_conflicts,
        )

    def autofix_resources(
        self, resource_conflicts: set[frozenset[OntologyStringResource]]
    ) -> set[OntologyStringResource]:
        """Fix conflicts in resources by producing a new set of resources with
        consistent behaviour.

        This ensures that there are no conflicts regarding any of these properties:

        * the combination of case sensitivity and mention confidence
        * associated id set
        * resource behaviour

        :param resource_conflicts:
        :return:
        """
        cleaned_resources = set()
        agg_func = max if self.autofix is AutofixStrategy.OPTIMISTIC else min
        for conflicted_set in resource_conflicts:
            original_synonyms_by_syn_norm: defaultdict[str, set[Synonym]] = defaultdict(set)
            alt_syns_by_syn_norm: defaultdict[str, set[Synonym]] = defaultdict(set)
            case_sensitivities = set()
            syn_string_lower_to_confidence = defaultdict(set)
            assoc_id_sets: set[EquivalentIdSet] = set()
            behaviours: set[OntologyStringBehaviour] = set()
            for resource in conflicted_set:
                syn_norm = resource.syn_norm_for_linking(self.entity_class)
                original_synonyms_by_syn_norm[syn_norm].update(resource.original_synonyms)
                alt_syns_by_syn_norm[syn_norm].update(resource.alternative_synonyms)
                behaviours.add(resource.behaviour)
                for syn in resource.all_synonyms():
                    syn_string_lower_to_confidence[syn.text.lower()].add(syn.mention_confidence)
                    case_sensitivities.add(syn.case_sensitive)
                if resource.associated_id_sets is not None:
                    assoc_id_sets.update(resource.associated_id_sets)

            if OntologyStringBehaviour.DROP_FOR_LINKING in behaviours:
                chosen_behaviour = OntologyStringBehaviour.DROP_FOR_LINKING
            elif OntologyStringBehaviour.ADD_FOR_LINKING_ONLY in behaviours:
                chosen_behaviour = OntologyStringBehaviour.ADD_FOR_LINKING_ONLY
            else:
                chosen_behaviour = OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING

            chosen_case_sensitivity = (
                min(case_sensitivities)
                if self.autofix is AutofixStrategy.OPTIMISTIC
                else max(case_sensitivities)
            )

            for syn_norm, mergeable_original_synonyms in original_synonyms_by_syn_norm.items():
                cleaned_resources.add(
                    OntologyStringResource(
                        behaviour=chosen_behaviour,
                        original_synonyms=frozenset(
                            dataclasses.replace(
                                syn,
                                case_sensitive=chosen_case_sensitivity,
                                mention_confidence=agg_func(
                                    syn_string_lower_to_confidence[syn.text.lower()]
                                ),
                            )
                            for syn in mergeable_original_synonyms
                        ),
                        alternative_synonyms=frozenset(
                            dataclasses.replace(
                                syn,
                                case_sensitive=chosen_case_sensitivity,
                                mention_confidence=agg_func(
                                    syn_string_lower_to_confidence[syn.text.lower()]
                                ),
                            )
                            for syn in alt_syns_by_syn_norm.get(syn_norm, set())
                        ),
                        associated_id_sets=frozenset(assoc_id_sets)
                        if len(assoc_id_sets) > 0
                        else None,
                    )
                )
        return cleaned_resources

    @staticmethod
    def check_for_case_conflicts_across_resources(
        resources: set[OntologyStringResource], strict: bool = False
    ) -> tuple[set[frozenset[OntologyStringResource]], set[OntologyStringResource]]:
        """Find conflicts in case sensitivity within a set of resources.

        Conflicts can occur when strings differ by case sensitivity, and a
        case-insensitive synonym will produce a :class:`.MentionConfidence`
        of equal or higher rank than a case-sensitive one.

        :param resources:
        :param strict: if True, then the function will return True if there
            are multiple mention confidences for a given string, regardless of case sensitivity
        :return: a set of conflicted subsets, and a set of clean resources.
        """

        maybe_good_resources_by_active_syn_lower = (
            OntologyStringConflictAnalyser.build_synonym_defaultdict(resources)
        )

        return OntologyStringConflictAnalyser.find_case_conflicts(
            maybe_good_resources_by_active_syn_lower, strict=strict
        )

    @staticmethod
    def find_case_conflicts(
        maybe_good_resources_by_active_syn_lower: defaultdict[str, set[OntologyStringResource]],
        strict: bool = False,
    ) -> tuple[set[frozenset[OntologyStringResource]], set[OntologyStringResource]]:
        all_conflicts = set()
        case_conflict_subsets = set()
        clean_resources = set()
        for potential_conflict_set in maybe_good_resources_by_active_syn_lower.values():

            if OntologyStringConflictAnalyser._resource_set_has_case_conflicts(
                potential_conflict_set, strict
            ):
                # uh ho - we have multiple identical synonyms attached to different resources
                case_conflict_subsets.add(frozenset(potential_conflict_set))
                all_conflicts.update(potential_conflict_set)
            else:
                clean_resources.update(potential_conflict_set)
        # there are scenarios wherein a case conflict can occur transitively.
        # Therefore, we need to eliminate from the clean resources any conflicted ones
        clean_resources.difference_update(all_conflicts)
        return case_conflict_subsets, clean_resources

    @staticmethod
    def build_synonym_defaultdict(
        resources: Iterable[OntologyStringResource],
    ) -> defaultdict[str, set[OntologyStringResource]]:
        maybe_good_resources_by_active_syn_lower = defaultdict(set)
        for resource in resources:
            for syn in resource.all_synonyms():
                maybe_good_resources_by_active_syn_lower[syn.text.lower()].add(resource)
        return maybe_good_resources_by_active_syn_lower

    def check_for_normalised_behaviour_conflicts_and_merge_if_possible(
        self, resources: set[OntologyStringResource]
    ) -> tuple[
        set[OntologyStringResource],
        set[OntologyStringResource],
        set[frozenset[OntologyStringResource]],
    ]:
        """Find behaviour conflicts in the resource set indexed by syn_norm.

        If possible, resources will be merged. If not, they will be added to a set of
        conflicting resources.

        :param resources:
        :return: A set of newly created merged resources, a set of resources eliminated
            through the merge, and a set of resources subsets that conflict and cannot
            be merged,
        """

        resources_by_syn_norm = (
            self._group_resources_by_syn_norm_and_check_for_normalisation_consistency_errors(
                resources
            )
        )

        normalisation_conflicts = set()
        merged_resources = set()
        eliminated_resources = set()
        for syn_norm, potentially_conflicting_resources in resources_by_syn_norm.items():
            if len(potentially_conflicting_resources) == 1:
                # no conflict
                continue

            # uh ho we have a normalisation/behaviour/id set conflict. If we can't merge, report an error
            original_synonyms_merged: set[Synonym] = set()
            generated_synonyms_merged: set[Synonym] = set()
            associated_id_sets_this_syn_norm: set[frozenset[EquivalentIdSet]] = set()
            comments = []
            behaviours = set()
            for conflicted_resource in potentially_conflicting_resources:

                behaviours.add(conflicted_resource.behaviour)
                original_synonyms_merged.update(conflicted_resource.original_synonyms)
                generated_synonyms_merged.update(conflicted_resource.alternative_synonyms)
                if conflicted_resource.associated_id_sets is not None:
                    associated_id_sets_this_syn_norm.add(conflicted_resource.associated_id_sets)
                if conflicted_resource.comment is not None:
                    comments.append(conflicted_resource.comment)

            if len(behaviours) > 1 or len(associated_id_sets_this_syn_norm) > 1:
                # uh ho - behaviours/id set clash. Add to norm conflicts
                normalisation_conflicts.add(frozenset(potentially_conflicting_resources))
            else:
                # merge the resources
                merged_resource = OntologyStringResource(
                    behaviour=next(iter(behaviours)),
                    original_synonyms=frozenset(original_synonyms_merged),
                    alternative_synonyms=frozenset(generated_synonyms_merged),
                    associated_id_sets=next(iter(associated_id_sets_this_syn_norm))
                    if len(associated_id_sets_this_syn_norm) == 1
                    else None,
                    comment="\n".join(comments) if len(comments) > 0 else None,
                )
                merged_resources.add(merged_resource)

                logger.warning(
                    "duplicate resource set merged. synonym_norm: %s, conflicts:\n%s\n merged to: %s",
                    syn_norm,
                    potentially_conflicting_resources,
                    merged_resource,
                )
                eliminated_resources.update(potentially_conflicting_resources)
        return merged_resources, eliminated_resources, normalisation_conflicts

    def _group_resources_by_syn_norm_and_check_for_normalisation_consistency_errors(
        self, resources: set[OntologyStringResource]
    ) -> defaultdict[str, set[OntologyStringResource]]:
        resources_by_syn_norm: defaultdict[str, set[OntologyStringResource]] = defaultdict(set)
        normalisation_errors = set()
        for resource in resources:
            syn_norms_this_resource = set(
                StringNormalizer.normalize(original_syn.text, entity_class=self.entity_class)
                for original_syn in resource.original_synonyms
            )
            if len(syn_norms_this_resource) > 1:
                normalisation_errors.add(resource)
            syn_norm = next(iter(syn_norms_this_resource))
            resources_by_syn_norm[syn_norm].add(resource)
        if normalisation_errors:
            # This should never be thrown by automatically generated resources
            raise CurationError(
                "One or more OntologyStringResource contains original synonyms that no longer normalise to the same value. This"
                " can happen if the implementation of the StringNormalizer has changed. Please correct the following"
                " resources:\n" + "\n".join(resource.to_json() for resource in normalisation_errors)
            )
        return resources_by_syn_norm

    @staticmethod
    def _resource_set_has_case_conflicts(
        resources: set[OntologyStringResource], strict: bool = False
    ) -> bool:
        """Checks for case conflicts in a set of :class:`.OntologyStringResource`\\ s.

        Intended behaviour is to allow different cases to produce different
        confidences based on confidence rank. A case-sensitive rank must always
        be higher than a case-insensitive rank, or a conflict will occur.

        For example, the following situation should be supported:

        "eGFR" -> ci and IGNORE
        "EGFR" -> cs and POSSIBLE
        "Egfr" -> cs and PROBABLE

        ...while the following situations are conflicted:

        "eGFR" -> ci and PROBABLE
        "Egfr" -> cs and POSSIBLE

        "Egfr" -> cs and PROBABLE
        "Egfr" -> cs and POSSIBLE

        "EGFR" -> ci and PROBABLE
        "Egfr" -> ci and POSSIBLE

        "Egfr" -> ci and PROBABLE
        "Egfr" -> ci and POSSIBLE

        :param resources:
        :param strict: if True, then the function will return True if there
            are multiple mention confidences for a given string, regardless of case sensitivity
        :return:
        """
        cs_conf_lookup = defaultdict(set)
        ci_conf_lookup = defaultdict(set)
        for resource in resources:
            for syn in resource.active_ner_synonyms():
                if syn.case_sensitive:
                    cs_conf_lookup[syn.text].add(syn.mention_confidence)
                else:
                    ci_conf_lookup[syn.text.lower()].add(syn.mention_confidence)

        for cased_syn_string, cs_confidences in cs_conf_lookup.items():
            ci_confidences: set[MentionConfidence] = ci_conf_lookup.get(
                cased_syn_string.lower(), set()
            )
            if len(ci_confidences) > 1 or (
                len(ci_confidences) == 1
                and len(cs_confidences) > 0
                and (min(cs_confidences) <= min(ci_confidences) or strict)
            ):
                return True
        for ci_confidences in ci_conf_lookup.values():
            if len(ci_confidences) > 1:
                return True
        return False

    def merge_human_and_auto_resources(
        self,
        human_curated_resources: set[OntologyStringResource],
        autocurated_resources: set[OntologyStringResource],
    ) -> OntologyResourceSetMergeReport:
        """Merge a set of human curated resources with a set of automatically curated
        resources, preferring the human set where possible.

        Note that the output is not guaranteed to be conflict free - consider calling
        :meth:`~.verify_resource_set_integrity`.

        :param human_curated_resources:
        :param autocurated_resources:
        :return:
        """
        # normalisation errors should have already been checked with self.verify_resource_set_integrity before calling this

        obsolete_human_set = set()
        resources_with_discrepancies = set()
        superfluous_human_curations = human_curated_resources.intersection(autocurated_resources)
        human_by_syn_norm = {
            resource.syn_norm_for_linking(self.entity_class): resource
            for resource in human_curated_resources
        }
        default_by_syn_norm = {
            resource.syn_norm_for_linking(self.entity_class): resource
            for resource in autocurated_resources
        }

        working_dict = {}
        for default_syn_norm, default_resource in default_by_syn_norm.items():

            maybe_human_curation = human_by_syn_norm.pop(default_syn_norm, None)
            if maybe_human_curation is None:
                working_dict[default_syn_norm] = default_resource
            else:
                working_dict[default_syn_norm] = maybe_human_curation
                # check the set of strings are equivalent. Otherwise new ones may have been created by syn generation
                if set(syn.text for syn in maybe_human_curation.original_synonyms) != set(
                    syn.text for syn in default_resource.original_synonyms
                ) or set(syn.text for syn in maybe_human_curation.alternative_synonyms) != set(
                    syn.text for syn in default_resource.alternative_synonyms
                ):
                    resources_with_discrepancies.add(
                        (
                            maybe_human_curation,
                            default_resource,
                        )
                    )

        # add any remaining to the obsolete set, unless they're additional
        for human_syn_norm, human_curation in human_by_syn_norm.items():
            if human_curation.additional_to_source:
                working_dict[human_syn_norm] = human_curation
            else:
                obsolete_human_set.add(human_curation)

        return OntologyResourceSetMergeReport(
            effective_resources=set(working_dict.values()),
            superfluous_resources=superfluous_human_curations,
            resources_with_discrepancies=resources_with_discrepancies,
            obsolete_resources=obsolete_human_set,
        )


class LinkingCandidateModificationResult(AutoNameEnum):
    ID_SET_MODIFIED = auto()
    LINKING_CANDIDATE_ADDED = auto()
    LINKING_CANDIDATE_DROPPED = auto()
    NO_ACTION = auto()


class OntologyResourceProcessor:
    """A OntologyResourceProcessor is responsible for modifying the set of
    :class:`.LinkingCandidate`\\s produced by an
    :class:`kazu.ontology_preprocessing.base.OntologyParser` with any relevant
    :class:`.GlobalParserActions` and/or :class:`.OntologyStringResource` associated
    with the parser.

    This class should be used before instances of :class:`.LinkingCandidate`\\s are loaded into the
    internal database representation.
    """

    # OntologyStringBehaviours are applied in the following order
    BEHAVIOUR_APPLICATION_ORDER = (
        OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
        OntologyStringBehaviour.ADD_FOR_LINKING_ONLY,
        OntologyStringBehaviour.DROP_FOR_LINKING,
    )
    _BEHAVIOUR_TO_ORDER_INDEX = {behav: i for i, behav in enumerate(BEHAVIOUR_APPLICATION_ORDER)}

    def __init__(
        self,
        parser_name: str,
        entity_class: str,
        global_actions: Optional[GlobalParserActions],
        resources: list[OntologyStringResource],
        linking_candidates: set[LinkingCandidate],
    ):
        """

        :param parser_name: name of parser to process
        :param entity_class: name of parser entity_class to process (typically as passed to :class:`kazu.ontology_preprocessing.base.OntologyParser`\\ )
        :param global_actions:
        :param resources:
        :param linking_candidates:
        """
        self.global_actions = global_actions
        self.entity_class = entity_class
        self.parser_name = parser_name
        self._candidates_by_syn_norm: dict[NormalisedSynonymStr, LinkingCandidate] = {}
        self._candidates_by_id: defaultdict[Idx, set[LinkingCandidate]] = defaultdict(set)
        for candidate in linking_candidates:
            self._update_candidate_lookups(candidate, False)
        self.resources = set(resources)
        self.dropped_keys: set[NormalisedSynonymStr] = set()

    @classmethod
    def resource_sort_key(cls, resource: OntologyStringResource) -> tuple[int, bool]:
        """Determines the order resources are processed in.

        We use associated_id_sets as a key, so that any overrides will be processed
        after any original behaviours.
        """
        return (
            cls._BEHAVIOUR_TO_ORDER_INDEX[resource.behaviour],
            resource.associated_id_sets is not None,
        )

    def _update_candidate_lookups(
        self, candidate: LinkingCandidate, override: bool
    ) -> Literal[
        LinkingCandidateModificationResult.LINKING_CANDIDATE_ADDED,
        LinkingCandidateModificationResult.NO_ACTION,
    ]:

        safe_to_add = False
        maybe_existing_candidate = self._candidates_by_syn_norm.get(candidate.synonym_norm)
        if maybe_existing_candidate is None:
            logger.debug("adding new candidate %s", candidate)
            safe_to_add = True
        elif override:
            safe_to_add = True
            logger.debug("overriding existing candidate %s", maybe_existing_candidate)
        elif (
            len(
                candidate.associated_id_sets.symmetric_difference(
                    maybe_existing_candidate.associated_id_sets
                )
            )
            > 0
        ):
            logger.warning(
                "conflict on synonym norms \n%s\n%s\nthe latter will be ignored",
                maybe_existing_candidate,
                candidate,
            )
        if safe_to_add:
            self._candidates_by_syn_norm[candidate.synonym_norm] = candidate
            for equiv_ids in candidate.associated_id_sets:
                for idx in equiv_ids.ids:
                    self._candidates_by_id[idx].add(candidate)
            return LinkingCandidateModificationResult.LINKING_CANDIDATE_ADDED
        else:
            return LinkingCandidateModificationResult.NO_ACTION

    def _drop_linking_candidate(self, synonym: NormalisedSynonymStr) -> None:
        """Remove a linking candidate from the database, so that it cannot be used as a
        linking target.

        :param synonym:
        :return:
        """
        try:
            candidate_to_remove = self._candidates_by_syn_norm.pop(synonym)
            self.dropped_keys.add(synonym)
            for equiv_id_set in candidate_to_remove.associated_id_sets:
                for idx in equiv_id_set.ids:
                    candidates_by_id = self._candidates_by_id.get(idx)
                    if candidates_by_id is not None:
                        candidates_by_id.remove(candidate_to_remove)
            logger.debug(
                "successfully dropped %s from database for %s",
                synonym,
                self.entity_class,
            )
        except KeyError:
            if synonym in self.dropped_keys:
                logger.debug(
                    "tried to drop %s from database, but key already dropped by another OntologyStringResource for %s",
                    synonym,
                    self.parser_name,
                )
            else:
                logger.warning(
                    "tried to drop %s from database, but key doesn't exist for %s",
                    synonym,
                    self.parser_name,
                )

    def _drop_id_from_all_linking_candidates(
        self, id_to_drop: Idx
    ) -> Counter[
        Literal[
            LinkingCandidateModificationResult.ID_SET_MODIFIED,
            LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED,
            LinkingCandidateModificationResult.NO_ACTION,
        ]
    ]:
        """Remove a given id from all :class:`.LinkingCandidate`\\ s.

        Drop any :class:`.LinkingCandidate`\\ s with no remaining ID after removal.

        :param id_to_drop:
        :return:
        """

        candidates_to_modify = self._candidates_by_id.get(id_to_drop, set())
        counter = Counter(
            self._drop_id_from_linking_candidate(
                id_to_drop=id_to_drop, candidate_to_modify=candidate_to_modify
            )
            for candidate_to_modify in set(candidates_to_modify)
        )

        return counter

    def _drop_id_from_linking_candidate(
        self, id_to_drop: Idx, candidate_to_modify: LinkingCandidate
    ) -> Literal[
        LinkingCandidateModificationResult.ID_SET_MODIFIED,
        LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED,
        LinkingCandidateModificationResult.NO_ACTION,
    ]:
        """Remove an id from a given :class:`.LinkingCandidate`\\ .

        :param id_to_drop:
        :param candidate_to_modify:
        :return:
        """
        new_assoc_id_frozenset = self._drop_id_from_associated_id_sets(
            id_to_drop, candidate_to_modify.associated_id_sets
        )
        if (
            len(new_assoc_id_frozenset.symmetric_difference(candidate_to_modify.associated_id_sets))
            == 0
        ):
            return LinkingCandidateModificationResult.NO_ACTION
        else:
            return self._modify_or_drop_linking_candidate_after_id_set_change(
                new_associated_id_sets=new_assoc_id_frozenset, candidate=candidate_to_modify
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

    def _modify_or_drop_linking_candidate_after_id_set_change(
        self, new_associated_id_sets: AssociatedIdSets, candidate: LinkingCandidate
    ) -> Literal[
        LinkingCandidateModificationResult.ID_SET_MODIFIED,
        LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED,
    ]:
        """Modifies or drops a :class:`.LinkingCandidate` after a
        :class:`.AssociatedIdSets` has changed.

        :param new_associated_id_sets:
        :param candidate:
        :return:
        """
        result: Literal[
            LinkingCandidateModificationResult.ID_SET_MODIFIED,
            LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED,
        ]
        if len(new_associated_id_sets) > 0:
            if new_associated_id_sets == candidate.associated_id_sets:
                raise ValueError(
                    "function called inappropriately where the id sets haven't changed. This"
                    " has failed as it will otherwise modify the value of aggregated_by, when"
                    " nothing has changed"
                )
            new_candidate = dataclasses.replace(
                candidate,
                associated_id_sets=new_associated_id_sets,
                aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
            )
            add_result = self._update_candidate_lookups(new_candidate, True)
            assert add_result is LinkingCandidateModificationResult.LINKING_CANDIDATE_ADDED
            result = LinkingCandidateModificationResult.ID_SET_MODIFIED
        else:
            # if there are no longer any id sets associated with the record, remove it completely
            self._drop_linking_candidate(candidate.synonym_norm)
            result = LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED
        return result

    def export_resources_and_final_candidates(
        self,
    ) -> tuple[list[OntologyStringResource], set[LinkingCandidate]]:
        """Perform any updates required to the linking candidates as specified in the
        curations/global actions.

        The returned :class:`.OntologyStringResource`\\s can be used for Dictionary based NER, whereas the
        returned :class:`.LinkingCandidate`\\s can be loaded into the internal database for linking.

        :return:
        """
        self._process_global_actions()
        return list(self._process_resources()), set(self._candidates_by_syn_norm.values())

    def _process_resources(self) -> Iterable[OntologyStringResource]:
        for resource in sorted(self.resources, key=self.resource_sort_key):
            maybe_resource = self._process_resource_action(resource)
            if maybe_resource:
                yield maybe_resource

    def _process_resource_action(
        self, resource: OntologyStringResource
    ) -> Optional[OntologyStringResource]:

        if resource.behaviour is OntologyStringBehaviour.DROP_FOR_LINKING:
            self._drop_linking_candidate(resource.syn_norm_for_linking(self.entity_class))
            result = None
        elif resource.behaviour is OntologyStringBehaviour.ADD_FOR_LINKING_ONLY:
            result = self._attempt_to_add_database_entry_for_resource(
                resource,
            )
        elif resource.behaviour is OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING:
            result = self._attempt_to_add_database_entry_for_resource(resource)
        else:
            raise ValueError(f"unknown behaviour for parser {self.parser_name}, {resource}")
        if result is LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED:
            return None
        else:
            return resource

    def _process_global_actions(self) -> None:
        if self.global_actions is None:
            return None

        override_resources_by_id = defaultdict(set)
        for resource in self.resources:
            if resource.associated_id_sets is not None:
                for equiv_id_set in resource.associated_id_sets:
                    for idx in equiv_id_set.ids:
                        override_resources_by_id[idx].add(resource)

        for action in self.global_actions.parser_behaviour(self.parser_name):
            if action.behaviour is ParserBehaviour.DROP_IDS_FROM_PARSER:
                ids = action.parser_to_target_id_mappings[self.parser_name]
                for idx in ids:
                    counter_this_idx = self._drop_id_from_all_linking_candidates(idx)
                    if (
                        counter_this_idx[LinkingCandidateModificationResult.ID_SET_MODIFIED]
                        + counter_this_idx[
                            LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED
                        ]
                        == 0
                    ):
                        logger.warning("failed to drop %s from %s", idx, self.parser_name)
                    else:
                        logger.debug(
                            "dropped ID %s from %s. LinkingCandidate modified count: %s, LinkingCandidate dropped count: %s",
                            idx,
                            self.parser_name,
                            counter_this_idx[LinkingCandidateModificationResult.ID_SET_MODIFIED],
                            counter_this_idx[
                                LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED
                            ],
                        )

                        for override_resource_to_modify in set(
                            override_resources_by_id.get(idx, set())
                        ):
                            assert override_resource_to_modify.associated_id_sets is not None
                            new_associated_id_sets = self._drop_id_from_associated_id_sets(
                                idx, override_resource_to_modify.associated_id_sets
                            )
                            if len(new_associated_id_sets) == 0:

                                self.resources.remove(override_resource_to_modify)
                                override_resources_by_id[idx].remove(override_resource_to_modify)
                                logger.debug(
                                    "removed resource %s because of global action",
                                    override_resource_to_modify,
                                )
                            elif (
                                new_associated_id_sets
                                != override_resource_to_modify.associated_id_sets
                            ):
                                self.resources.remove(override_resource_to_modify)
                                override_resources_by_id[idx].remove(override_resource_to_modify)
                                mod_resource = dataclasses.replace(
                                    override_resource_to_modify,
                                    associated_id_sets=new_associated_id_sets,
                                )
                                self.resources.add(mod_resource)
                                override_resources_by_id[idx].add(mod_resource)
                                logger.debug(
                                    "modified resource %s to %s because of global action",
                                    override_resource_to_modify,
                                    mod_resource,
                                )

            else:
                raise ValueError(f"unknown behaviour for parser {self.parser_name}, {action}")
        return None

    def _attempt_to_add_database_entry_for_resource(
        self,
        resource: OntologyStringResource,
    ) -> Literal[
        LinkingCandidateModificationResult.LINKING_CANDIDATE_ADDED,
        LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED,
        LinkingCandidateModificationResult.NO_ACTION,
    ]:
        """Create a new :class:`~kazu.data.LinkingCandidate` for the database, or return
        an existing matching one if already present.

        Notes:

        If a syn_norm already exists in self._candidates_by_syn_norm that matches 'resource.syn_norm_for_linking',
        this method will check to see if the 'resource_associated_id_set' matches the existing terms
        :class:`.AssociatedIdSets`\\.

        If so, no action will be taken. If not, a warning will be logged as adding it
        will cause irregularities in the database.

        If the syn_norm does not exist, this method will create a new :class:`~kazu.data.LinkingCandidate`
        with the provided :class:`.AssociatedIdSets`\\.

        :param resource:
        :return:
        """
        log_prefix = "%(parser_name)s attempting to create linking candidate for syn_norm: <%(syn_norm)s> resource: %(resource)s"
        syn_norm = resource.syn_norm_for_linking(self.entity_class)
        log_formatting_dict: dict[str, Any] = {
            "parser_name": self.parser_name,
            "syn_norm": syn_norm,
            "resource": resource,
        }

        # look up the syn norm in the db
        maybe_existing_linking_candidate = self._candidates_by_syn_norm.get(syn_norm)
        resource_associated_id_set = resource.associated_id_sets
        if resource_associated_id_set is None and maybe_existing_linking_candidate is not None:
            logger.debug(
                log_prefix
                + " but no associated id set provided, so candidate will inherit the parser defaults",
                log_formatting_dict,
            )
            return LinkingCandidateModificationResult.NO_ACTION
        elif resource_associated_id_set is None and maybe_existing_linking_candidate is None:
            logger.info(
                log_prefix
                + " but syn_norm <%(syn_norm)s> does not exist in synonym database."
                + " It may have been removed by a global action."
                + " Since no id set was provided, no entry can be created",
                log_formatting_dict,
            )
            return LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED

        # resource_associated_id_set is implicitly not None
        assert resource_associated_id_set is not None
        if len(resource_associated_id_set) == 0:
            logger.debug(
                "all ids removed by global action for %s,  Parser name: %s",
                resource,
                self.parser_name,
            )
            return LinkingCandidateModificationResult.LINKING_CANDIDATE_DROPPED

        if maybe_existing_linking_candidate is not None:
            log_formatting_dict[
                "existing_id_set"
            ] = maybe_existing_linking_candidate.associated_id_sets

            if (
                len(
                    resource_associated_id_set.symmetric_difference(
                        maybe_existing_linking_candidate.associated_id_sets
                    )
                )
                == 0
            ):
                logger.debug(
                    log_prefix
                    + " but syn_nom <%(syn_norm)s> already exists in synonym database."
                    + "since this LinkingCandidate matches the id_set, no action is required. %(existing_id_set)s",
                    log_formatting_dict,
                )
                return LinkingCandidateModificationResult.NO_ACTION
            else:
                logger.debug(
                    log_prefix
                    + " . Will remove existing syn_norm <%(syn_norm)s> as an ID set override has been specified",
                    log_formatting_dict,
                )

        # no candidate exists, or we want to override so one will be made
        assert resource_associated_id_set is not None
        for equiv_id_set in set(resource_associated_id_set):
            for idx in equiv_id_set.ids:
                if idx not in self._candidates_by_id:
                    resource_associated_id_set = self._drop_id_from_associated_id_sets(
                        id_to_drop=idx, associated_id_sets=resource_associated_id_set
                    )
                    logger.warning(
                        "Attempted to add candidate containing %s but this id does not exist in the parser and will be ignored",
                        idx,
                    )
        if len(resource_associated_id_set) > 0:
            is_symbolic = any(
                StringNormalizer.classify_symbolic(syn.text, self.entity_class)
                for syn in resource.original_synonyms
            )
            new_candidate = LinkingCandidate(
                synonym_norm=syn_norm,
                raw_synonyms=frozenset(syn.text for syn in resource.original_synonyms),
                is_symbolic=is_symbolic,
                mapping_types=frozenset(("kazu_curated",)),
                associated_id_sets=resource_associated_id_set,
                parser_name=self.parser_name,
                aggregated_by=EquivalentIdAggregationStrategy.MODIFIED_BY_CURATION,
            )
            return self._update_candidate_lookups(new_candidate, True)
        else:
            return LinkingCandidateModificationResult.NO_ACTION
