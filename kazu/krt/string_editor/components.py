import logging
from typing import Iterable

import streamlit as st
from kazu.krt.components import (
    ResourceEditor,
    ParserSelector,
    PlaceholderResource,
)
from kazu.krt.string_editor.utils import (
    CaseConflictResolutionRequest,
    ResourceConflict,
    ResourceConflictManager,
)
from kazu.krt.utils import get_resource_manager


@st.cache_resource
def get_manager() -> ResourceConflictManager:
    return ResourceConflictManager(manager=get_resource_manager())


def reset() -> None:
    # type ignore as mypy doesn't pick up annotations
    get_resource_manager.clear()  # type: ignore[attr-defined]
    get_manager.clear()  # type: ignore[attr-defined]


class StringConflictForm:
    STRING_CONFLICT_BATCH_APPLY = "STRING_CONFLICT_BATCH_APPLY"
    DATAFRAME_SELECTION = "DATAFRAME_SELECTION"
    DATAFRAME = "DATAFRAME"
    FORM_PICKER = "FORM_PICKER"

    @staticmethod
    def _submit_form_for_batch_conflict_resolution(conflicts: Iterable[ResourceConflict]) -> None:
        for conflict in conflicts:
            logging.info(f"submit form {id(conflict)}")
            if (
                st.session_state.get(StringConflictForm.STRING_CONFLICT_BATCH_APPLY)
                is CaseConflictResolutionRequest.OPTIMISTIC
            ):
                conflict.batch_resolve(True)
            elif (
                st.session_state.get(StringConflictForm.STRING_CONFLICT_BATCH_APPLY)
                is CaseConflictResolutionRequest.PESSIMISTIC
            ):
                conflict.batch_resolve(False)
            flow = get_manager()
            flow.sync_resources_for_resolved_resource_conflict_and_find_new_conflicts(conflict)

        if "submit_batch" in st.session_state:
            del st.session_state["submit_batch"]
        del st.session_state[StringConflictForm.DATAFRAME]

    @staticmethod
    def _submit_form_for_individual_conflicts(conflicts: Iterable[ResourceConflict]) -> None:
        for conflict in conflicts:
            logging.info(f"submit form {id(conflict)}")
            if (
                st.session_state.get(StringConflictForm.STRING_CONFLICT_BATCH_APPLY)
                is CaseConflictResolutionRequest.OPTIMISTIC
            ):
                conflict.batch_resolve(True)
            elif (
                st.session_state.get(StringConflictForm.STRING_CONFLICT_BATCH_APPLY)
                is CaseConflictResolutionRequest.PESSIMISTIC
            ):
                conflict.batch_resolve(False)
            else:
                for parser_name, resource_dict in conflict.parser_to_resource_to_resolution.items():

                    for (
                        original_resource,
                        new_resource,
                    ) in ResourceEditor.extract_form_data_from_state(
                        parser_name=parser_name, resources=resource_dict.keys()
                    ):
                        conflict.parser_to_resource_to_resolution[parser_name][
                            original_resource
                        ] = new_resource
            flow = get_manager()
            flow.sync_resources_for_resolved_resource_conflict_and_find_new_conflicts(conflict)

        if "submit_batch" in st.session_state:
            del st.session_state["submit_batch"]
        del st.session_state[StringConflictForm.DATAFRAME]

    @staticmethod
    def _individual_conflict_resolution_form() -> None:
        row_ids = (
            st.session_state[StringConflictForm.DATAFRAME_SELECTION]
            .get("selection", {})
            .get("rows", None)
        )
        if len(row_ids) == 1:
            row_id = row_ids[0]
            df = st.session_state[StringConflictForm.DATAFRAME]
            conflict_id = df.iloc[[row_id]]["id"].values[0]
            conflict = get_manager().unresolved_conflicts[conflict_id]
            resources = set(
                resource
                for resource_dict in conflict.parser_to_resource_to_resolution.values()
                for resource in resource_dict
            )
            ResourceEditor.display_resource_editor(
                resources=resources,
                on_click_override=StringConflictForm._submit_form_for_individual_conflicts,
                args=([conflict],),
            )

    @staticmethod
    def _batch_conflict_resolution_form() -> None:
        df = st.session_state[StringConflictForm.DATAFRAME]
        resolution_choices = list(CaseConflictResolutionRequest)
        resolution_choices.remove(CaseConflictResolutionRequest.CUSTOM)
        st.radio(
            "select a resolution",
            options=resolution_choices,
            key=StringConflictForm.STRING_CONFLICT_BATCH_APPLY,
        )
        st.write("select rows to apply the resolution to")
        event = st.dataframe(
            df,
            use_container_width=True,
            selection_mode="multi-row",
            on_select="rerun",
            hide_index=True,
            column_config={"id": None},
            key=StringConflictForm.DATAFRAME_SELECTION,
        )
        conflicts = []
        for row_id in event.get("selection", {}).get("rows", []):
            conflict_id = df.iloc[[row_id]]["id"].values[0]
            conflict = get_manager().unresolved_conflicts[conflict_id]
            conflicts.append(conflict)

        if conflicts:
            disabled = False
        else:
            disabled = True
        submitted = st.button("submit batch", key="submit_batch", disabled=disabled)
        if submitted:
            StringConflictForm._submit_form_for_batch_conflict_resolution(conflicts)
            st.rerun()

    @staticmethod
    def resolve_conflicts_form() -> None:
        with st.container(border=True):
            st.write(f"""{len(get_manager().unresolved_conflicts)} remaining discrepancies""")
            st.radio(
                "mode",
                options=["apply to all", "edit individual"],
                index=0,
                key=StringConflictForm.FORM_PICKER,
            )
            if StringConflictForm.DATAFRAME not in st.session_state:
                maybe_df = get_manager().summary_df()
                if not maybe_df.empty:
                    maybe_df = maybe_df.sort_values(by="string len", ascending=False)
                    st.session_state[StringConflictForm.DATAFRAME] = maybe_df
                    st.rerun()

            elif st.session_state[StringConflictForm.DATAFRAME].empty:
                st.write("no conflicts found")
            else:

                if st.session_state[StringConflictForm.FORM_PICKER] == "apply to all":
                    StringConflictForm._batch_conflict_resolution_form()
                else:
                    df = st.session_state[StringConflictForm.DATAFRAME]
                    st.dataframe(
                        df,
                        use_container_width=True,
                        selection_mode="single-row",
                        on_select="rerun",
                        hide_index=True,
                        column_config={"conflict": None},
                        key=StringConflictForm.DATAFRAME_SELECTION,
                    )
                    StringConflictForm._individual_conflict_resolution_form()

    @staticmethod
    def search_and_edit_synonyms_form() -> None:
        with st.container(border=True):
            text = st.text_input("search for a string")
            if text:
                manager = get_resource_manager()
                maybe_resources = manager.synonym_lookup.get(text.lower())
                if not maybe_resources:
                    st.write("No existing resources found that contain this string")
                    StringConflictForm._add_new_resource_form(text)
                else:
                    st.write(
                        "One or more resources already exists for this string. You can edit them here:"
                    )
                    mode = st.radio(
                        label="edit existing or create new",
                        options=["edit existing", "create new"],
                        index=0,
                    )
                    if mode == "edit existing":
                        ResourceEditor.display_resource_editor(maybe_resources)
                    else:
                        StringConflictForm._add_new_resource_form(text)

    @staticmethod
    def _add_new_resource_form(text: str) -> None:
        ParserSelector.display_parser_selector(clear_state=False)
        maybe_parser_name = ParserSelector.get_selected_parser_name()
        if maybe_parser_name:
            parser = get_resource_manager().parsers[maybe_parser_name]
            st.write(f"selected parser is {parser.name}, entity class: {parser.entity_class}")
            PlaceholderResource.create_placeholder_resource(text=text, parser_name=parser.name)
            parser_name, placeholder_resource = PlaceholderResource.get_placeholder_resource()
            ResourceEditor.display_resource_editor(
                resources={placeholder_resource},
                maybe_parser_name=parser_name,
                on_click_override=ResourceEditor.submit_form_for_addition,
            )
