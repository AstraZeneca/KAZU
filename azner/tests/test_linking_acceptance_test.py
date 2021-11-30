import os

import pydash
import pytest
from hydra import compose, initialize_config_dir

from azner.pipeline.pipeline import Pipeline, load_steps
from azner.tests.utils import entity_linking_hard_cases, AcceptanceTestError, SKIP_MESSAGE


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
