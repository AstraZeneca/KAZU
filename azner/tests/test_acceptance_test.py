import os

import pandas as pd
import pydash
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from azner.data.data import Entity
from azner.pipeline.pipeline import Pipeline, load_steps
from azner.tests.utils import (
    entity_linking_hard_cases,
    SKIP_MESSAGE,
)
from azner.tests.utils import full_pipeline_test_cases, AcceptanceTestError

# TODO: we need a much better/consistent evaluation dataset for acceptance tests. Currently these all fail until we have
# one


@pytest.mark.skipif(os.environ.get("KAZU_TEST_CONFIG_DIR") is None, reason=SKIP_MESSAGE)
def test_sapbert_acceptance():
    minimum_pass_score = 0.80
    with initialize_config_dir(config_dir=os.environ.get("KAZU_TEST_CONFIG_DIR")):
        cfg = compose(
            config_name="config",
            overrides=[
                "SapBertForEntityLinkingStep.rebuild_ontology_cache=False",
            ],
        )

        hits = []
        misses = []
        step = instantiate(cfg.SapBertForEntityLinkingStep)
        easy_test_docs, iris, sources = entity_linking_hard_cases()
        successes, failures = step(easy_test_docs)
        entities = pydash.flatten([x.get_entities() for x in successes])
        for entity, iri, source in zip(entities, iris, sources):
            if (
                entity.metadata.mappings is not None
                and len(entity.metadata.mappings) > 0
                and entity.metadata.mappings[0].idx == iri
            ):
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
            raise AcceptanceTestError(
                f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
            )


@pytest.mark.skipif(os.environ.get("KAZU_TEST_CONFIG_DIR") is None, reason=SKIP_MESSAGE)
def test_full_pipeline_acceptance_test():
    with initialize_config_dir(config_dir=os.environ.get("KAZU_TEST_CONFIG_DIR")):
        cfg = compose(config_name="config")
        pipeline = Pipeline(steps=load_steps(cfg))

        docs, annotation_dfs = full_pipeline_test_cases()
        successes = pipeline(docs)
        for doc, annotations in zip(successes, annotation_dfs):
            section = doc.sections[0]
            for entity in section.entities:
                matches = query_annotations_df(annotations, entity)
                if matches.shape[0] != 1:
                    raise AcceptanceTestError(
                        f"failed to match {entity} in section: {section.text}"
                    )


def query_annotations_df(annotations: pd.DataFrame, entity: Entity):
    if (
        entity.metadata is not None
        and entity.metadata.mappings is not None
        and len(entity.metadata.mappings) > 0
    ):
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


@pytest.mark.skipif(os.environ.get("KAZU_TEST_CONFIG_DIR") is None, reason=SKIP_MESSAGE)
def test_dictionary_entity_linking():
    minimum_pass_score = 0.80
    with initialize_config_dir(config_dir=os.environ.get("KAZU_TEST_CONFIG_DIR")):
        cfg = compose(config_name="config", overrides=["pipeline=entity_linking_only"])
        hits = []
        misses = []
        pipeline = Pipeline(steps=load_steps(cfg))
        easy_test_docs, iris, sources = entity_linking_hard_cases()
        successes = pipeline(easy_test_docs)
        entities = pydash.flatten([x.get_entities() for x in successes])
        for entity, iri, source in zip(entities, iris, sources):
            if entity.metadata.mappings is not None and iri in [
                x.idx for x in entity.metadata.mappings
            ]:
                hits.append(entity)
            else:
                misses.append(
                    (
                        entity,
                        iri,
                    )
                )

        for entity, iri in misses:
            if entity.metadata.mappings is not None:
                print(
                    f"missed {entity.match}: got {entity.metadata.mappings[0].idx}, wanted {iri} "
                )
            else:
                print(f"missed {entity.match}: got Nothing, wanted {iri} ")
        total = len(hits) + len(misses)
        score = len(hits) / total
        if score < minimum_pass_score:
            raise AcceptanceTestError(
                f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
            )
