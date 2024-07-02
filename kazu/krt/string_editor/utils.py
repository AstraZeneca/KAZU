import logging
from collections import defaultdict
from enum import Enum, auto
from typing import Iterable, Optional

import pandas as pd
from kazu.data import OntologyStringResource, MentionConfidence
from kazu.krt.resource_manager import ResourceManager
from kazu.krt.utils import (
    create_new_resource_with_updated_synonyms,
    resource_to_df,
)
from kazu.ontology_preprocessing.curation_utils import OntologyStringConflictAnalyser


class CaseConflictResolutionRequest(Enum):
    PESSIMISTIC = auto()
    OPTIMISTIC = auto()
    CUSTOM = auto()


class ResourceConflict:
    """This class represents a conflict in resources.

    It provides methods to resolve these conflicts optimistically or pessimistically.
    """

    def __init__(self, conflict_dict: dict[str, Iterable[OntologyStringResource]]):
        """
        :param conflict_dict: A dictionary mapping parser names to a list of resources that are in conflict.
        """
        self.parser_to_resource_to_resolution: defaultdict[
            str, dict[OntologyStringResource, Optional[OntologyStringResource]]
        ] = defaultdict(lambda: dict())
        self.parser_names = set()
        self.forms_to_parser = defaultdict(set)
        self.string_set = set()
        self.confidences = set()
        self.cs = set()
        for parser_name, resources in conflict_dict.items():
            self.parser_names.add(parser_name)
            for resource in resources:
                self.parser_to_resource_to_resolution[parser_name][resource] = None
                for syn in resource.active_ner_synonyms():
                    self.forms_to_parser[syn].add(parser_name)
                    self.string_set.add(syn.text)
                    self.confidences.add(syn.mention_confidence)
                    self.cs.add(syn.case_sensitive)

    def _shortest_string_len(self) -> int:
        """Get the length of the shortest string in the conflict.

        :return: The length of the shortest string.
        """
        return min(len(x) for x in self.string_set)

    def __lt__(self, other: "ResourceConflict") -> bool:
        """Compare this ResourceConflict with another based on the length of the
        shortest string in the conflict.

        :param other: The other ResourceConflict to compare with.
        :return: True if this ResourceConflict's shortest string is shorter than the
            other's, False otherwise.
        """
        return self._shortest_string_len() < other._shortest_string_len()

    def batch_resolve(self, optimistic: bool) -> None:
        """Resolve the conflict in batch, changing the parameters of all synonyms at
        once to share the same values.

        The optimistic param indicates the minimum case sensitivity and maximum mention
        confidence should be chosen (otherwise vice-versa)

        :param optimistic: Whether to resolve the conflict optimistically or
            pessimistically.
        """

        new_cs = min(self.cs) if optimistic else max(self.cs)
        new_conf = max(self.confidences) if optimistic else min(self.confidences)
        self._resolve(new_cs=new_cs, new_conf=new_conf)

    def _resolve(self, new_cs: bool, new_conf: MentionConfidence) -> None:
        """Resolve the conflict by creating a new resource with updated synonyms.

        :param new_cs: The new case sensitivity to use.
        :param new_conf: The new mention confidence to use.
        """
        for parser_name, resolution_dict in self.parser_to_resource_to_resolution.items():
            for resource in resolution_dict.keys():
                new_resource = create_new_resource_with_updated_synonyms(new_conf, new_cs, resource)
                self.parser_to_resource_to_resolution[parser_name][resource] = new_resource


class ResourceConflictManager:
    """This class is responsible for managing :class:`.ResourceConflict`\\s in
    resources.

    It provides methods to find conflicts in resources, sync resources for resolved
    string conflicts and find new conflicts.
    """

    def __init__(
        self,
        manager: ResourceManager,
    ):
        """Initialize with a :class:`.ResourceManager`\\.

        :param manager: The :class:`.ResourceManager` to use.
        """
        self.manager = manager
        self.unresolved_conflicts: dict[int, ResourceConflict] = {}
        self.unresolved_conflicts_by_parser: dict[
            tuple[str, OntologyStringResource], ResourceConflict
        ] = {}
        self._init_conflict_maps()

    def sync_resources_for_resolved_resource_conflict_and_find_new_conflicts(
        self, conflict: ResourceConflict
    ) -> None:
        """Sync resources for a resolved conflict and find new conflicts.

        This will refresh the internal map of conflicts.

        :param conflict: a resolved :class:`.ResourceConflict`\\.
        """
        self._sync_resources(conflict)
        self._check_resolved_conflict_for_new_conflicts(conflict)

    def _init_conflict_maps(self) -> None:
        for i, conflict in enumerate(
            self._find_conflicts_in_resources(self.manager.resource_to_parsers.keys())
        ):
            self.unresolved_conflicts[i] = conflict
            self._update_conflict_parser_map((conflict,))

    def _update_conflict_parser_map(self, unresolved_conflicts: Iterable[ResourceConflict]) -> None:
        """Update the conflict parser map with unresolved conflicts.

        :param unresolved_conflicts: The unresolved conflicts to add to the map.
        """
        for conflict in unresolved_conflicts:
            for parser_name, resolution_dict in conflict.parser_to_resource_to_resolution.items():
                for resource in resolution_dict.keys():
                    self.unresolved_conflicts_by_parser[(parser_name, resource)] = conflict

    def _find_conflicts_in_resources(
        self, resources: Iterable[OntologyStringResource]
    ) -> set[ResourceConflict]:
        """Find conflicts in resources.

        :param resources: The resources to check for conflicts against other resources
            currently loaded in the internal resource manager.
        :return: A set of ResourceConflicts.
        """
        logging.info("Looking for conflicts across all parsers...")
        (
            case_conflicts,
            _,
        ) = OntologyStringConflictAnalyser.check_for_case_conflicts_across_resources(
            resources, strict=True  # type: ignore[arg-type]  # dict_keys isn't a subtype of builtin set
        )
        unresolved_conflicts = set()
        for conflict_set in case_conflicts:
            conflict_dict = defaultdict(set)
            for conflict_resource in conflict_set:
                parser_name: str
                for parser_name in self.manager.resource_to_parsers[conflict_resource]:
                    conflict_dict[parser_name].add(conflict_resource)

            conflict = ResourceConflict(dict(conflict_dict))
            unresolved_conflicts.add(conflict)
        return unresolved_conflicts

    def _sync_resources(
        self,
        conflict: ResourceConflict,
    ) -> None:
        """When synching resources, we are implicitly creating a new human curation, as
        the autogenerated set should never be manually changed.

        Therefore, we only care about
        :param conflict:
        :return:
        """
        for parser_name, resolution_map in conflict.parser_to_resource_to_resolution.items():
            for original_resource, new_resource in resolution_map.items():
                assert new_resource is not None
                # pop the key as is no longer valid
                try:
                    self.unresolved_conflicts_by_parser.pop(
                        (
                            parser_name,
                            original_resource,
                        )
                    )
                except KeyError:
                    pass

                self.manager.sync_resources(
                    original_resource=original_resource,
                    new_resource=new_resource,
                    parser_name=parser_name,
                )

    def _check_resolved_conflict_for_new_conflicts(self, conflict: ResourceConflict) -> None:
        parser_name: str
        for parser_name, resolution_map in conflict.parser_to_resource_to_resolution.items():
            for new_resource in resolution_map.values():
                assert new_resource is not None
                new_conflicts = self._find_conflicts_in_resources([new_resource])
                if new_conflicts:
                    self._update_conflict_parser_map(new_conflicts)
        self.unresolved_conflicts = {
            i: conflict
            for i, conflict in enumerate(set(self.unresolved_conflicts_by_parser.values()))
        }

    def _resource_conflict_to_df(self, conflict: ResourceConflict) -> pd.DataFrame:
        data = []
        for parser_name, resource_dict in conflict.parser_to_resource_to_resolution.items():
            df = pd.concat(resource_to_df(resource) for resource in resource_dict.keys())
            df["parser_name"] = parser_name
            data.append(df)
        if len(data) == 1:
            return data[0]
        else:
            return pd.concat(data)

    def summary_df(self) -> pd.DataFrame:
        """Create a summary :class:`~pandas.DataFrame` of unresolved conflicts.

        :return: A DataFrame summarizing the unresolved conflicts.
        """
        data = []
        for i, conflict in self.unresolved_conflicts.items():
            data.append(
                {
                    "parsers": "|".join(sorted(list(conflict.parser_names))),
                    "strings": "|".join(sorted(list(conflict.string_set))),
                    "string len": max(len(x) for x in conflict.string_set),
                    "id": i,
                }
            )
        return pd.DataFrame(data)
