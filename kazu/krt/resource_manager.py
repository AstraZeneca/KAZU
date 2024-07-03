import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterable

from hydra.utils import instantiate
from kazu.data import (
    OntologyStringResource,
)
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.ontology_preprocessing.curation_utils import (
    OntologyStringConflictAnalyser,
    dump_ontology_string_resources,
    OntologyResourceSetCompleteReport,
)
from kazu.pipeline import Pipeline
from kazu.krt import load_config


class ResourceManager:
    """The ResourceManager class is responsible for managing resources in the streamlit
    application.

    It manages the global state of all :class:`.OntologyStringResource`\\s.
    It is also responsible for saving and updates to these resources in the configured model pack.
    """

    def __init__(self, parsers: Iterable[OntologyParser]) -> None:
        """Initializes the ResourceManager instance.

        Sets up dictionaries for managing resources and parsers, and loads resources
        from each parser.
        """
        self.parser_to_curations: defaultdict[str, set[OntologyStringResource]] = defaultdict(set)
        # since duplicate resources may exist in multiple parsers, this mapping controls that
        self.resource_to_parsers: defaultdict[OntologyStringResource, set[str]] = defaultdict(set)
        # where to save the serialised resources to
        self.parser_to_path: dict[str, Path] = {}
        # parser to report
        self.parser_to_report: dict[str, OntologyResourceSetCompleteReport] = {}
        self.parsers: dict[str, OntologyParser] = {}
        self.parser_cache_invalid: set[str] = set()

        for parser in parsers:
            self.parsers[parser.name] = parser
            logging.info(f"loading data from parser: {parser.name}")
            if parser.curations_path is None:
                logging.warning(
                    "Parser %s has no curations path and will not be loaded", parser.name
                )
                continue
            self.parser_to_path[parser.name] = parser.curations_path
            _, resource_report = parser.populate_metadata_db_and_resolve_string_resources()
            self.parser_to_report[parser.name] = resource_report
            # we need the clean resources and the conflicted resources from the parser
            for resource in resource_report.final_conflict_report.clean_resources:
                self.resource_to_parsers[resource].add(parser.name)
            for resource_set in resource_report.final_conflict_report.case_conflicts:
                for resource in resource_set:
                    self.resource_to_parsers[resource].add(parser.name)

            if resource_report.human_conflict_report:
                for resource in resource_report.human_conflict_report.clean_resources:
                    self.resource_to_parsers[resource].add(parser.name)
                    self.parser_to_curations[parser.name].add(resource)
                for resource_set in resource_report.human_conflict_report.case_conflicts:
                    for resource in resource_set:
                        self.resource_to_parsers[resource].add(parser.name)
                        self.parser_to_curations[parser.name].add(resource)
                for (
                    frozen_resource_set
                ) in resource_report.human_conflict_report.normalisation_conflicts:
                    for resource in frozen_resource_set:
                        self.resource_to_parsers[resource].add(parser.name)
                        self.parser_to_curations[parser.name].add(resource)

            if resource_report.merge_report:
                for r1, r2 in resource_report.merge_report.resources_with_discrepancies:
                    self.resource_to_parsers[r1].add(parser.name)
                    self.resource_to_parsers[r2].add(parser.name)

        logging.info("building synonym lookup...")

        self.synonym_lookup = OntologyStringConflictAnalyser.build_synonym_defaultdict(
            self.resource_to_parsers.keys()
        )

    def parser_count(self) -> int:
        """Returns the number of parsers loaded by the ResourceManager instance.

        :return:
        """

        return len(self.parsers)

    def delete_resource(self, resource: OntologyStringResource, parser_name: str) -> None:
        """Deletes a resource from the internal state.

        :param resource: The resource to be deleted.
        :param parser_name: The name of the parser that is handling the resource.
        :return:
        """

        for synonym in resource.all_strings():
            self.synonym_lookup[synonym.lower()].discard(resource)

        self.parser_to_curations[parser_name].discard(resource)
        # it may have already been popped, or not exist
        try:
            self.resource_to_parsers.pop(resource)
        except KeyError:
            pass

    def sync_resources(
        self,
        original_resource: Optional[OntologyStringResource],
        new_resource: OntologyStringResource,
        parser_name: str,
    ) -> None:
        """Synchronizes resources within the internal state.

        If an original resource is provided, it is removed from the resource
        dictionaries and the new resource is added. If no original resource is provided,
        only the new resource is added. Note that no action is taken if
        original_resource == new_resource

        :param original_resource: The original resource to be replaced. If None, no
            resource is replaced.
        :param new_resource: The new resource to be added.
        :param parser_name: The name of the parser that is handling the resource.
        :return:
        """
        resources_are_equal = False
        if original_resource:
            # Only do something if the resource has actually changed
            resources_are_equal = original_resource == new_resource
            if not resources_are_equal:
                self.delete_resource(original_resource, parser_name)
                self.parser_cache_invalid.add(parser_name)

        if not original_resource or not resources_are_equal:
            for synonym in new_resource.all_strings():
                self.synonym_lookup[synonym.lower()].add(new_resource)

            self.parser_to_curations[parser_name].add(new_resource)
            self.resource_to_parsers[new_resource].add(parser_name)
            self.parser_cache_invalid.add(parser_name)

    def save(self) -> Iterable[str]:
        """Saves updated resources to the model pack.

        :return:
        """

        for parser_name, curation_set in self.parser_to_curations.items():
            path = self.parser_to_path[parser_name]
            yield f"Saving updated resources to {path}"
            dump_ontology_string_resources(curation_set, path, force=True)

        self.clear_invalid_caches()

    @staticmethod
    def _load_pipeline() -> Pipeline:
        load_config.clear()  # type: ignore[attr-defined]
        cfg = load_config()
        pipeline: Pipeline = instantiate(cfg.Pipeline)
        return pipeline

    def clear_invalid_caches(self) -> None:
        for parser_name in set(self.parser_cache_invalid):
            parser = self.parsers[parser_name]
            logging.info("clearing invalid cache for %s", parser_name)
            parser.clear_cache()
            self.parser_cache_invalid.discard(parser_name)

    def load_cache_state_aware_pipeline(self) -> Pipeline:
        self.clear_invalid_caches()
        return self._load_pipeline()
