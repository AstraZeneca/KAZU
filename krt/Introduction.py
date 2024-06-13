import logging

import streamlit as st

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Kazu Resource Tool",
    layout="wide",
    page_icon="📚",
)

st.write("# Welcome to The Kazu Resource Tool (KRT)")

st.markdown(
    """
    Welcome to Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with
    Korea University, designed to handle production workloads.

    This tool is designed to facilitate the resolution of the [issues described here](https://astrazeneca.github.io/KAZU/ontology_parser.html#using-ontologystringresource-for-dictionary-based-matching-and-or-to-modify-an-ontology-s-behaviour)

    Select a workflow from the sidebar to begin

    ### Want to learn more?
    - Check out [Kazu on github](https://github.com/AstraZeneca/KAZU)
    - [Documentation here](https://astrazeneca.github.io/KAZU/index.html)
"""
)
