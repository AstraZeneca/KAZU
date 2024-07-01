import dataclasses

import pandas as pd
import streamlit as st
from hydra.utils import instantiate
from kazu.data import MentionConfidence, OntologyStringResource
from kazu.krt.resource_manager import ResourceManager
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.krt import load_config


@st.cache_resource(show_spinner="Loading parsers")
def load_parsers() -> list[OntologyParser]:
    cfg = load_config()
    parsers = []
    for parser in cfg.ontologies.parsers.values():
        parsers.append(instantiate(parser))
    return parsers


@st.cache_resource(show_spinner="Building the ResourceManager")
def get_resource_manager() -> ResourceManager:
    return ResourceManager(load_parsers())


def create_new_resource_with_updated_synonyms(
    new_conf: MentionConfidence, new_cs: bool, resource: OntologyStringResource
) -> OntologyStringResource:
    """Create a new :class:`.OntologyStringResource` with updated :class:`.Synonym`\\s.

    This function takes a :class:`.MentionConfidence`\\, a boolean value, and an
    :class:`.OntologyStringResource` object as inputs. It creates a new :class:`.OntologyStringResource`
    object with updated synonyms based on the provided :class:`.MentionConfidence` and boolean
    values. The new_conf parameter represents the new :class:`.MentionConfidence` to be set for
    the :class:`.Synonym`\\s. The new_cs parameter represents whether the :class:`.Synonym`\\s should be case
    sensitive or not. The resource parameter is the original :class:`.OntologyStringResource`
    object whose :class:`.Synonym`\\s are to be updated.

    :param new_conf:
    :param new_cs:
    :param resource:
    :return:
    """
    new_original_syns = set()
    for syn in resource.original_synonyms:
        new_original_syns.add(
            dataclasses.replace(syn, case_sensitive=new_cs, mention_confidence=new_conf)
        )
    new_alternative_syns = set()
    for syn in resource.alternative_synonyms:
        new_alternative_syns.add(
            dataclasses.replace(syn, case_sensitive=new_cs, mention_confidence=new_conf)
        )

    return dataclasses.replace(
        resource,
        original_synonyms=frozenset(new_original_syns),
        alternative_synonyms=frozenset(new_alternative_syns),
    )


def resource_to_df(
    resource: OntologyStringResource, include_behaviour: bool = False
) -> pd.DataFrame:
    """Convert an :class:`.OntologyStringResource` to a :class:`~pandas.DataFrame` for
    display in Streamlit.

    This function takes an OntologyStringResource object as input and transforms it into a pandas DataFrame.
    The DataFrame has four columns: `type`, `text`, `confidence`, and `case_sensitive`. The `type` column
    indicates whether the synonym is "original" or "alternative". The `text` column contains the synonym text.
    The `confidence` column contains the mention confidence, and the `case_sensitive` column indicates
    whether the synonym is case sensitive or not.

    :param resource:
    :param include_behaviour: if True, include the behaviour column in the DataFrame
    :return:
    """
    data = [
        (
            "original",
            syn.text,
            syn.mention_confidence.name,
            syn.case_sensitive,
        )
        for syn in resource.original_synonyms
    ] + [
        (
            "alternative",
            syn.text,
            syn.mention_confidence.name,
            syn.case_sensitive,
        )
        for syn in resource.alternative_synonyms
    ]
    df = pd.DataFrame.from_records(data, columns=["type", "text", "confidence", "case_sensitive"])
    if include_behaviour:
        df["behaviour"] = resource.behaviour
    return df
