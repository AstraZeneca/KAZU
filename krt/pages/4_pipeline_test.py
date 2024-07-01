import logging
from typing import Any, cast

import streamlit as st
from kazu.krt.utils import get_resource_manager
from kazu.data import Document
from kazu.pipeline import Pipeline

st.markdown("# Pipeline test")
st.write("""This page allows you to test the pipeline after changing the resource configuration.""")
logger = logging.getLogger(__name__)


def load_pipeline_after_change() -> Pipeline:
    manager = get_resource_manager()
    if manager.parser_cache_invalid or "pipeline" not in st.session_state:
        pipeline = manager.load_cache_state_aware_pipeline()
        st.session_state["pipeline"] = pipeline

    return cast(Pipeline, st.session_state["pipeline"])


def _process_text() -> dict[str, Any]:
    with st.spinner("loading pipeline"):
        pipeline = load_pipeline_after_change()
    doc = Document.create_simple_document(st.session_state["text_area"])
    pipeline([doc])
    return doc.to_dict()


st.text_area("Enter text to process", key="text_area")
clicked = st.button("submit")
if clicked:
    st.write(_process_text())
