import streamlit as st
from kazu.krt.components import ParserSelector, show_save_button, show_reset_button
from kazu.krt.resource_discrepancy_editor.components import (
    get_resource_merge_manager_for_parser,
    ResourceDiscrepancyResolutionForm,
    reset,
)


show_save_button()
show_reset_button(reset)

with st.expander("Description", expanded=True):
    st.markdown(
        """### Correct Resource Discrepancy Issues.

    Discrepancies can occur when there are inconsistencies between the
    autogenerated ontology resources and their human overrides. This can occur:

    1) After an ontology version update, where the generated
    resources have changed but their human overrides haven't.

    2) After the configuration of the [Autocurator](https://astrazeneca.github.io/KAZU/_autosummary/kazu.ontology_preprocessing.autocuration.html#kazu.ontology_preprocessing.autocuration.AutoCurator) has changed.

    3) If the implementation of the [StringNormalizer](https://astrazeneca.github.io/KAZU/_autosummary/kazu.utils.string_normalizer.html#kazu.utils.string_normalizer.StringNormalizer) has changed.


    This tool finds these discrepancies and prompts the user for a fix.
    When saved, the necessary human resources are updated in the model pack.
    """
    )


ParserSelector.display_parser_selector()
parser_name = ParserSelector.get_selected_parser_name()

if parser_name:
    manager = get_resource_merge_manager_for_parser(parser_name)
    if ResourceDiscrepancyResolutionForm.state_key(parser_name) not in st.session_state:
        st.session_state[ResourceDiscrepancyResolutionForm.state_key(parser_name)] = (
            ResourceDiscrepancyResolutionForm(manager)
        )

    form = st.session_state[ResourceDiscrepancyResolutionForm.state_key(parser_name)]
    with st.container(border=True):
        if manager.summary_df().empty:
            st.write(f"""{len(manager.unresolved_discrepancies)} discrepancies remaining""")
        else:
            form.display_main_form()
