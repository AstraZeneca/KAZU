import logging
from pathlib import Path

from kazu.data import (
    OntologyStringResource,
)
from kazu.ontology_preprocessing.curation_utils import (
    dump_ontology_string_resources,
)

logger = logging.getLogger(__name__)

_ONTOLOGY_UPGRADE_REPORT_DIR = "_ontology_upgrade_report"


class OntologyUpgradeReport:
    """A report on the delta of :class:`~.OntologyStringResource`\\s generated between
    two versions of an ontology, or the same version of the ontology with a different
    configuration of :class:`~.AutoCurator`\\.

    This is useful to highlight which resources are novel or obsolete between versions.
    """

    def __init__(
        self,
        new_version_auto_generated_resources_clean: set[OntologyStringResource],
        previous_version_auto_generated_resources_clean: set[OntologyStringResource],
    ):
        """

        :param new_version_auto_generated_resources_clean: resources from new version.
        :param previous_version_auto_generated_resources_clean: resources from previous version.
        """
        self.new_version_auto_generated_resources_clean = new_version_auto_generated_resources_clean
        self.previous_version_auto_generated_resources_clean = (
            previous_version_auto_generated_resources_clean
        )

    @property
    def new_resources_after_upgrade(self) -> set[OntologyStringResource]:
        """Novel resources generated after upgrade/config change."""
        return self.new_version_auto_generated_resources_clean.difference(
            self.previous_version_auto_generated_resources_clean
        )

    @property
    def obsolete_resources_after_upgrade(self) -> set[OntologyStringResource]:
        """Resources that are now obsolete (i.e. no longer do anything) after upgrade
        :return:"""
        return self.previous_version_auto_generated_resources_clean.difference(
            self.new_version_auto_generated_resources_clean
        )

    def write_report(self, ontology_data_path: Path, parser_name: str) -> None:
        """Write a report for human review of the delta.

        :param ontology_data_path:
        :param parser_name:
        :return:
        """

        upgrade_report_path = ontology_data_path.parent.joinpath(_ONTOLOGY_UPGRADE_REPORT_DIR)
        logger.info(
            "%s writing novel/obsolete resource reports after upgrade (to %s)",
            parser_name,
            upgrade_report_path,
        )

        maybe_new_resources = self.new_resources_after_upgrade
        if maybe_new_resources:
            dump_ontology_string_resources(
                maybe_new_resources,
                upgrade_report_path.joinpath("novel_auto_generated_resources.jsonl"),
                force=True,
            )
        maybe_obsolete_resources = self.obsolete_resources_after_upgrade
        if maybe_obsolete_resources:
            dump_ontology_string_resources(
                maybe_obsolete_resources,
                upgrade_report_path.joinpath("obsolete_auto_generated_resources.jsonl"),
                force=True,
            )
