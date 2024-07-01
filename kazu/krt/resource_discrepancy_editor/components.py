import streamlit as st
from kazu.data import MentionConfidence
from kazu.krt.components import (
    ResourceEditor,
)
from kazu.krt.resource_discrepancy_editor.utils import (
    SynonymDiscrepancy,
    ResourceDiscrepancyManger,
)
from kazu.krt.utils import create_new_resource_with_updated_synonyms, get_resource_manager


@st.cache_resource
def get_resource_merge_manager(parser_name: str) -> ResourceDiscrepancyManger:
    manager = get_resource_manager()
    return ResourceDiscrepancyManger(
        parser_name=parser_name,
        manager=manager,
    )


def reset() -> None:
    # type ignore as mypy doesn't pick up annotations
    get_resource_manager.clear()  # type: ignore[attr-defined]
    get_resource_merge_manager.clear()  # type: ignore[attr-defined]


def get_resource_merge_manager_for_parser(parser_name: str) -> ResourceDiscrepancyManger:
    return get_resource_merge_manager(parser_name=parser_name)


class ResourceDiscrepancyResolutionForm:
    """This class is used to handle the resolution of resource discrepancies.

    It provides a form interface for the user to interact with and resolve
    discrepancies.
    """

    ATTEMPT_AUTOFIX = "ATTEMPT_AUTOFIX"

    def __init__(self, manager: ResourceDiscrepancyManger):
        self.manager = manager
        self.autofix_already_attempted = False

    @staticmethod
    def state_key(parser_name: str) -> str:
        return f"{ResourceDiscrepancyResolutionForm.__name__}{parser_name}"

    def display_main_form(self) -> None:
        """Display the main form for resolving resource discrepancies.

        :return:
        """
        st.write("select a row to resolve a discrepancy")
        event = st.dataframe(
            self.manager.summary_df(),
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            hide_index=True,
            column_config={"id": None},
        )
        if not self.autofix_already_attempted:
            self._display_attempt_autofix_button()
            self._run_autofix_if_requested()

        for row_id in event.get("selection", {}).get("rows", []):
            self._display_discrepancy_form_for_selected_index(index=row_id)

    def _reset_form(self) -> None:
        """Reset the form by setting the AUTOFIX_ALREADY_ATTEMPTED session state to
        False."""
        self.autofix_already_attempted = False

    def _display_attempt_autofix_button(self) -> None:
        """Display a button for the user to attempt to autofix discrepancies.

        The button is disabled if autofix has already been attempted.
        """
        st.button(
            "Attempt to Autofix discrepancies",
            key=ResourceDiscrepancyResolutionForm.get_attempt_autofix_key(self.manager.parser_name),
            disabled=self.autofix_already_attempted,
        )

    @staticmethod
    def get_attempt_autofix_key(parser_name: str) -> str:
        return f"{ResourceDiscrepancyResolutionForm.ATTEMPT_AUTOFIX}_{parser_name}"

    def _run_autofix_if_requested(self) -> None:
        """If the user has requested to run autofix, apply autofix to all resources and
        rerun the script.

        :return:
        """
        if st.session_state.get(
            ResourceDiscrepancyResolutionForm.get_attempt_autofix_key(self.manager.parser_name)
        ):
            self.manager.apply_autofix_to_all()
            self.autofix_already_attempted = True
            st.rerun()

    def _submit_form_batch(
        self, conf: MentionConfidence, cs: bool, conflict: SynonymDiscrepancy, index: int
    ) -> None:
        """Submit the form to resolve a discrepancy.

        All synonyms will be updated with the provided case sensitivity and confidence.

        :param conf:
        :param cs:
        :param conflict:
        :param index:
        :return:
        """
        new_resource = create_new_resource_with_updated_synonyms(
            new_conf=conf, new_cs=cs, resource=conflict.auto_resource
        )
        self.manager.commit(
            original_human_resource=conflict.human_resource,
            new_resource=new_resource,
            index=index,
        )
        self._reset_form()

    def _submit_form_individual(self, discrepancy: SynonymDiscrepancy, index: int) -> None:
        """Submit the form to resolve a discrepancy with individual edits.

        :param discrepancy:
        :param index:
        :return:
        """

        for _, new_resource in ResourceEditor.extract_form_data_from_state(
            parser_name=self.manager.parser_name, resources={discrepancy.auto_resource}
        ):

            self.manager.commit(
                original_human_resource=discrepancy.human_resource,
                new_resource=new_resource,
                index=index,
            )
            self._reset_form()

    def _display_discrepancy_form_for_selected_index(self, index: int) -> None:
        """Display the discrepancy form for the selected index.

        :param index:
        :return:
        """
        discrepancy = self.manager.unresolved_discrepancies[index]
        st.write(discrepancy.dataframe())
        form = st.radio("select a form", options=["apply to all", "edit individual"])
        if form == "apply to all":
            self._display_batch_edit_form(discrepancy, index)
        else:
            ResourceEditor.display_resource_editor(
                resources={discrepancy.auto_resource},
                on_click_override=ResourceDiscrepancyResolutionForm._submit_form_individual,
                args=(
                    self,
                    discrepancy,
                    index,
                ),
            )

    def _display_batch_edit_form(self, discrepancy: SynonymDiscrepancy, index: int) -> None:
        """Display the batch edit form for the given discrepancy.

        :param discrepancy:
        :param index:
        :return:
        """
        with st.form("conflict_editor"):
            header = st.columns([2, 2])
            header[0].subheader("case sensitivity")
            header[1].subheader("confidence")
            row1 = st.columns([2, 2])
            defaults = next(iter(discrepancy.auto_resource.original_synonyms))
            cs, conf = ResourceEditor.display_case_sensitivity_and_confidence_selector(
                row=row1, default_syn=defaults
            )
            st.form_submit_button(
                "Submit",
                on_click=ResourceDiscrepancyResolutionForm._submit_form_batch,
                args=(
                    self,
                    conf,
                    cs,
                    discrepancy,
                    index,
                ),
            )
