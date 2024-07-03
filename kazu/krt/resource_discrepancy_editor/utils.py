from typing import Optional, Iterable

import pandas as pd
from kazu.data import OntologyStringResource
from kazu.krt.resource_manager import ResourceManager
from kazu.krt.utils import (
    create_new_resource_with_updated_synonyms,
    resource_to_df,
)


class SynonymDiscrepancy:
    """This class represents a discrepancy between a human-generated
    :class:`.OntologyStringResource` and an auto-generated
    :class:`.OntologyStringResource`.

    It provides methods to automatically resolve the discrepancy, convert the resources
    to a DataFrame, and get an example string for display.
    """

    def __init__(
        self, human_resource: OntologyStringResource, auto_resource: OntologyStringResource
    ):
        self.auto_resource = auto_resource
        self.human_resource = human_resource

    def auto_resolve(self) -> Optional[OntologyStringResource]:
        """This method attempts to automatically resolve discrepancies between human and
        auto resources.

        It first creates a set of tuples containing the mention confidence and case
        sensitivity for all :class:`.Synonym`\\s in the human resource. If there is more than one
        unique tuple in the set, it means there are discrepancies in the human resource
        itself, and the method returns None. If there is exactly one unique tuple, it
        means all synonyms in the human resource have the same mention confidence and
        case sensitivity. In this case, it updates all forms of the auto resource with
        this mention confidence and case sensitivity, and returns the updated auto
        resource.

        :return: The updated auto resource if discrepancies can be auto-resolved, which
            can be used as a human override. None otherwise.
        """
        human_aspects = set(
            (
                x.mention_confidence,
                x.case_sensitive,
            )
            for x in self.human_resource.all_synonyms()
        )
        if len(human_aspects) != 1:
            return None
        else:
            new_conf, new_cs = next(iter(human_aspects))
            return create_new_resource_with_updated_synonyms(
                new_conf=new_conf, new_cs=new_cs, resource=self.auto_resource
            )

    def dataframe(self) -> pd.DataFrame:
        """Converts the human and auto resources to DataFrames, merges them, and returns
        the rows with any null values (i.e. discrepancies)

        :return: A :class:`~pandas.DataFrame` representing the discrepancies between the human and auto
            resources.
        """
        human_df = resource_to_df(self.human_resource)
        auto_df = resource_to_df(self.auto_resource)
        merged = pd.merge(
            human_df, auto_df, how="outer", on=["type", "text"], suffixes=("_human", "_auto")
        )
        return merged[merged.isnull().any(axis=1)]

    def example_string(self) -> str:
        """Returns an example string from the human resource's original synonyms."""
        return next(iter(self.human_resource.original_synonyms)).text


class ResourceDiscrepancyManger:
    """This class manages :class:`.SynonymDiscrepancy`\\s between human-generated
    resources and auto- generated resources.

    It provides methods to automatically resolve all discrepancies, commit changes to
    the resources, and get a summary DataFrame.
    """

    def __init__(
        self,
        parser_name: str,
        manager: ResourceManager,
    ):
        """Initializes the ResourceDiscrepancyManager.

        :param parser_name: The name of the parser used to generate the resources.
        :param manager: The :class:`.ResourceManager` object used to manage the resources.
        """
        self.manager = manager
        self.parser_name = parser_name
        report = manager.parser_to_report[parser_name]
        if not report.merge_report:
            todo = set()
        else:

            todo = set(
                SynonymDiscrepancy(human_resource=human_curation, auto_resource=autocuration)
                for human_curation, autocuration in report.merge_report.resources_with_discrepancies
            )
        self._build_discrepancy_lookup(todo)

    def _build_discrepancy_lookup(self, todo: Iterable[SynonymDiscrepancy]) -> None:
        self.unresolved_discrepancies: dict[int, SynonymDiscrepancy] = {
            i: discrepancy for i, discrepancy in enumerate(todo)
        }

    def apply_autofix_to_all(self) -> None:
        """Attempts to automatically resolve all discrepancies.

        If a discrepancy can be auto-resolved, it syncs the resources and removes the
        discrepancy from the internal unresolved discrepancies list.
        """
        for i in list(self.unresolved_discrepancies):
            discrepancy = self.unresolved_discrepancies[i]
            maybe_new_resource = discrepancy.auto_resolve()
            if maybe_new_resource is not None:
                self.manager.sync_resources(
                    original_resource=discrepancy.human_resource,
                    new_resource=maybe_new_resource,
                    parser_name=self.parser_name,
                )
                self.unresolved_discrepancies.pop(i)
        # need to recalculate lookup as UI indices no longer valid
        self._build_discrepancy_lookup(self.unresolved_discrepancies.values())

    def commit(
        self,
        original_human_resource: OntologyStringResource,
        new_resource: OntologyStringResource,
        index: int,
    ) -> None:
        """Commits changes to the resources and removes the discrepancy from the
        internal todo list.

        :param original_human_resource: The original human-generated resource.
        :param new_resource: The new resource to replace the original one.
        :param index: The index of the discrepancy.
        """
        self.manager.sync_resources(
            original_resource=original_human_resource,
            new_resource=new_resource,
            parser_name=self.parser_name,
        )
        self.unresolved_discrepancies.pop(index)
        self._build_discrepancy_lookup(self.unresolved_discrepancies.values())

    def summary_df(self) -> pd.DataFrame:
        """Returns a :class:`pandas.DataFrame` summarizing the unresolved discrepancies.

        :return: A DataFrame with columns for id, example text, and the number of unique
            synonyms in the human and auto resources.
        """
        data = []
        for i, discrepancy in self.unresolved_discrepancies.items():
            data.append(
                {
                    "id": i,
                    "example_text": discrepancy.example_string(),
                    "human_resource_unique_synonyms": len(
                        set(discrepancy.human_resource.all_synonyms())
                    ),
                    "auto_resource_unique_synonyms": len(
                        set(discrepancy.auto_resource.all_synonyms())
                    ),
                }
            )
        return pd.DataFrame(data)
