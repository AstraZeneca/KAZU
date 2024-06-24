import streamlit as st
from kazu.krt.components import show_save_button, show_reset_button
from kazu.krt.string_editor.components import (
    reset,
)
from kazu.krt.ontology_update_editor.components import OntologyUpdateForm

st.markdown("# Update Ontology Version")
show_save_button()
show_reset_button(reset)
st.write("""This page allows you to update various public ontologies to their latest version. """)
st.write("choose a parser to update")

OntologyUpdateForm.display_main_form()
