import streamlit as st

from kazu.krt.components import ParserSelector
from kazu.krt.ontology_update_editor.utils import OntologyUpdateManager
from kazu.krt.utils import get_resource_manager


@st.cache_resource
def get_upgrade_manager(parser_name: str) -> OntologyUpdateManager:
    """Get an OntologyUpdateManager instance for the given parser_name.

    :param parser_name:
    :return:
    """
    return OntologyUpdateManager(parser_name)


class OntologyUpdateForm:
    DOWNLOAD_COMPLETED_OR_SKIPPED = "download_completed"
    ATTEMPT_DOWNLOAD = "download_skipped"
    UPDATES_SAVED = "updates_saved"

    @staticmethod
    def get_download_completed_or_skipped(parser_name: str) -> bool:
        return bool(
            st.session_state.get(
                f"{parser_name}_{OntologyUpdateForm.DOWNLOAD_COMPLETED_OR_SKIPPED}"
            )
        )

    @staticmethod
    def set_download_completed_or_skipped(parser_name: str) -> None:
        st.session_state[f"{parser_name}_{OntologyUpdateForm.DOWNLOAD_COMPLETED_OR_SKIPPED}"] = True

    @staticmethod
    def get_updates_saved_key(parser_name: str) -> str:
        return f"{parser_name}_{OntologyUpdateForm.UPDATES_SAVED}"

    @staticmethod
    def get_attempt_download_key(parser_name: str) -> str:
        return f"{parser_name}_{OntologyUpdateForm.ATTEMPT_DOWNLOAD}"

    @staticmethod
    def get_updates_saved(parser_name: str) -> bool:
        return bool(st.session_state.get(OntologyUpdateForm.get_updates_saved_key(parser_name)))

    @staticmethod
    def display_download_config_modifier_form(manager: OntologyUpdateManager) -> None:
        """

        :param manager:
        :return:
        """
        with st.form("config_mod_form"):
            for arg, val in manager.get_downloader_args_and_vals().items():
                st.text_input(arg, value=val, key=arg)
            st.form_submit_button(
                "Submit",
                on_click=OntologyUpdateForm._submit_download_config_modifier_form,
                args=(manager,),
            )

    @staticmethod
    def _submit_download_config_modifier_form(manager: OntologyUpdateManager) -> None:
        updated_args = {}
        for arg, arg_type in manager.get_downloader_args_and_types().items():
            updated_args[arg] = arg_type(st.session_state[arg])
        manager.modify_downloader_cfg_with_new_args(updated_args)
        manager.modify_parser_cfg_with_new_args()

    @staticmethod
    def display_download_form(manager: OntologyUpdateManager) -> None:
        if not OntologyUpdateForm.get_download_completed_or_skipped(manager.parser.name):
            with st.form("downloader_config_form"):
                st.radio(
                    "Download resource",
                    options=["yes", "no"],
                    index=0,
                    key=OntologyUpdateForm.get_attempt_download_key(manager.parser.name),
                )
                st.form_submit_button(
                    "Continue - warning: downloading may delete the original file/folder in your model pack",
                    on_click=OntologyUpdateForm._submit_download_form,
                    args=(manager,),
                )

    @staticmethod
    def _submit_download_form(manager: OntologyUpdateManager) -> None:
        if (
            st.session_state[OntologyUpdateForm.get_attempt_download_key(manager.parser.name)]
            == "yes"
        ):
            with st.spinner("downloading ontology"):
                manager.download_ontology_with_new_config()
        OntologyUpdateForm.set_download_completed_or_skipped(manager.parser.name)

    @staticmethod
    def _write_new_defaults_to_model_pack(manager: OntologyUpdateManager) -> None:
        with st.spinner("writing new defaults to model pack"):
            manager.write_new_defaults_to_model_pack()
        # writing defaults to model pack will invalidate the resource manager cache
        get_resource_manager.clear()  # type: ignore[attr-defined]

    @staticmethod
    def display_main_form() -> None:
        ParserSelector.display_parser_selector(clear_state=True)
        parser_name = ParserSelector.get_selected_parser_name()
        if parser_name:
            with st.spinner(f"preparing to update {parser_name}"):
                manager = get_upgrade_manager(parser_name)
            if not manager.parser.ontology_downloader:
                st.write(
                    "no downloader available for this parser. To update, manually put the resource in the model pack and delete the default resources folder."
                )
            else:
                if not OntologyUpdateForm.get_download_completed_or_skipped(parser_name):
                    st.header("set the parameters for the download")
                    OntologyUpdateForm.display_download_config_modifier_form(manager)
                    st.markdown("### new downloader yaml config is as follows:")
                    st.markdown(
                        f"""```yaml
                    {manager.display_new_downloader_config_as_yaml()}```"""
                    )
                    OntologyUpdateForm.display_download_form(manager)
                if OntologyUpdateForm.get_download_completed_or_skipped(parser_name):
                    manager.modify_parser_cfg_with_new_args()
                    st.markdown(
                        "### new parser config (note, you may need to correct interpolated paths):"
                    )
                    st.markdown(
                        f"""```yaml
                    {manager.display_new_parser_config_as_yaml()}```"""
                    )

                    if not OntologyUpdateForm.get_updates_saved(parser_name):
                        with st.spinner("building upgrade_report"):

                            report = manager.get_or_build_upgrade_report()

                        st.write(
                            f"found {len(report.obsolete_resources_after_upgrade)} obsolete resources"
                        )
                        st.write(manager.obsolete_df())
                        st.write(f"found {len(report.new_resources_after_upgrade)} new resources")
                        st.write(manager.novel_df())
                        st.button(
                            f"click here to update the default resources for {parser_name}",
                            on_click=OntologyUpdateForm._write_new_defaults_to_model_pack,
                            args=(manager,),
                            key=OntologyUpdateForm.get_updates_saved_key(parser_name),
                        )
                    else:
                        st.write("updates saved")
