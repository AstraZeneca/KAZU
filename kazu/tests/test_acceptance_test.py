import pandas as pd
import pytest
import pydash
from hydra.utils import instantiate

from kazu.data.data import Entity
from kazu.pipeline import Pipeline, load_steps
from kazu.tests.utils import (
    entity_linking_hard_cases,
    requires_model_pack,
    full_pipeline_test_cases,
)

# TODO: we need a much better/consistent evaluation dataset for acceptance tests. Currently these all fail until we have
# one


# this applies the require_model_pack mark to all tests in this module
pytestmark = requires_model_pack


def test_sapbert_acceptance(kazu_test_config):
    minimum_pass_score = 0.80
    hits = []
    misses = []
    step = instantiate(kazu_test_config.SapBertForEntityLinkingStep)
    easy_test_docs, iris, sources = entity_linking_hard_cases()
    successes, failures = step(easy_test_docs)
    entities = pydash.flatten([x.get_entities() for x in successes])
    for entity, iri, source in zip(entities, iris, sources):
        if len(entity.metadata.mappings) > 0 and entity.metadata.mappings[0].idx == iri:
            hits.append(entity)
        else:
            misses.append(
                (
                    entity,
                    iri,
                )
            )

    for entity, iri in misses:
        print(f"missed {entity.match}: got {entity.metadata.mappings}, wanted {iri} ")
    total = len(hits) + len(misses)
    score = len(hits) / total
    if score < minimum_pass_score:
        pytest.fail(
            f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
        )


def test_full_pipeline_acceptance_test(kazu_test_config):
    pipeline = Pipeline(steps=load_steps(kazu_test_config))
    docs, annotation_dfs = full_pipeline_test_cases()
    successes = pipeline(docs)
    for doc, annotations in zip(successes, annotation_dfs):
        section = doc.sections[0]
        for entity in section.entities:
            matches = query_annotations_df(annotations, entity)
            if matches.shape[0] != 1:
                pytest.fail(f"failed to match {entity} in section: {section.text}")


def query_annotations_df(annotations: pd.DataFrame, entity: Entity):
    if len(entity.metadata.mappings) > 0:
        mapping_id = entity.metadata.mappings[0].idx
    else:
        mapping_id = None

    matches = annotations[
        (annotations["start"] == entity.start)
        & (annotations["end"] == entity.end)
        & (annotations["match"] == entity.match)
        & (annotations["entity_class"] == entity.entity_class)
        & (
            (annotations["mapping_id"] == mapping_id)
            if mapping_id is not None
            else (pd.isna(annotations["mapping_id"]))
        )
    ]
    return matches


def test_dictionary_entity_linking(override_kazu_test_config):
    minimum_pass_score = 0.80
    cfg = override_kazu_test_config(overrides=["pipeline=entity_linking_only"])
    hits = []
    misses = []
    pipeline = Pipeline(steps=load_steps(cfg))
    easy_test_docs, iris, sources = entity_linking_hard_cases()
    successes = pipeline(easy_test_docs)
    entities = pydash.flatten([x.get_entities() for x in successes])
    for entity, iri, source in zip(entities, iris, sources):
        if iri in [x.idx for x in entity.metadata.mappings]:
            hits.append(entity)
        else:
            misses.append(
                (
                    entity,
                    iri,
                )
            )

    for entity, iri in misses:
        if len(entity.metadata.mappings) > 0:
            print(f"missed {entity.match}: got {entity.metadata.mappings[0].idx}, wanted {iri} ")
        else:
            print(f"missed {entity.match}: got Nothing, wanted {iri} ")
    total = len(hits) + len(misses)
    score = len(hits) / total
    if score < minimum_pass_score:
        pytest.fail(
            f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
        )
