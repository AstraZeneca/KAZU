import os
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from azner.tests.utils import SKIP_MESSAGE, full_pipeline_test_cases, AcceptanceTestError
from azner.data.data import Entity
from azner.pipeline.pipeline import Pipeline, load_steps


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
