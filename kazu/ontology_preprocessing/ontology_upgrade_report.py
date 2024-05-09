import dataclasses
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


@dataclasses.dataclass()
class OntologyUpgradeReport:
    parser_name: str
    new_version_auto_generated_resources_clean: set[OntologyStringResource] = dataclasses.field(
        default_factory=set
    )
    previous_version_auto_generated_resources_clean: set[
        OntologyStringResource
    ] = dataclasses.field(default_factory=set)

    @property
    def new_resources_after_upgrade(self) -> set[OntologyStringResource]:
        return self.new_version_auto_generated_resources_clean.difference(
            self.previous_version_auto_generated_resources_clean
        )

    @property
    def obsolete_resources_after_upgrade(self) -> set[OntologyStringResource]:
        return self.previous_version_auto_generated_resources_clean.difference(
            self.new_resources_after_upgrade
        )

    def write_report(self, ontology_data_path: Path) -> None:

        upgrade_report_path = ontology_data_path.parent.joinpath(_ONTOLOGY_UPGRADE_REPORT_DIR)
        logger.info(
            "%s writing novel/obsolete resource reports after upgrade (to %s)",
            self.parser_name,
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
