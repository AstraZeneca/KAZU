import streamlit as st
from kazu.krt.components import (
    show_save_button,
    show_reset_button,
)
from kazu.krt.string_editor.components import (
    StringConflictForm,
    reset,
)

st.markdown("# String Matching Management")
show_save_button()
show_reset_button(reset)
st.write(
    """This page modifies the configuration of OntologyStringResources, and finds and fixes conflicts."""
)
choice = st.radio(
    "Choose one",
    options=["resolve_conflicts", "modify or add curation"],
    index=0,
)


if choice == "resolve_conflicts":
    StringConflictForm.resolve_conflicts_form()
else:
    StringConflictForm.search_and_edit_synonyms_form()
