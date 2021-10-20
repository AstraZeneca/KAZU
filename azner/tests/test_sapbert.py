import os

import pytest
from hydra import initialize_config_module, compose
from hydra.utils import instantiate
from steps.utils.utils import documents_to_entity_list
from tests.utils import entity_linking_easy_cases, TINY_CHEMBL_KB_PATH


@pytest.mark.timeout(10)
def test_sapbert_step():
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.knowledgebase_path={TINY_CHEMBL_KB_PATH}",
                "SapBertForEntityLinkingStep.rebuild_kb_cache=True",
            ],
        )

        step = instantiate(cfg.SapBertForEntityLinkingStep)
        easy_test_docs, iris = entity_linking_easy_cases()
        successes, failures = step(easy_test_docs)
        entities = documents_to_entity_list(successes)
        for entity, iri in zip(entities, iris):
            assert entity.metadata.mappings[0].idx == iri

        # test cache
        for _ in range(1000):
            easy_test_docs, iris = entity_linking_easy_cases()
            successes, failures = step(easy_test_docs)
            entities = documents_to_entity_list(successes)
            for entity, iri in zip(entities, iris):
                assert entity.metadata.mappings[0].idx == iri
