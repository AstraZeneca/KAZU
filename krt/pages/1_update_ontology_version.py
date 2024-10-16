import streamlit as st
from kazu.krt.ontology_update_editor.components import OntologyUpdateForm

st.markdown("# Update Ontology Version")
st.write("""This page allows you to update various public ontologies to their latest version. """)
st.write("choose a parser to update")

OntologyUpdateForm.display_main_form()
