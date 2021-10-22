import os

import pytest
from hydra import initialize_config_module, compose
from hydra.utils import instantiate
from steps.linking.sapbert import SapBertForEntityLinkingStep
from steps.utils.utils import documents_to_entity_list, get_cache_file
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
        easy_test_docs, iris, sources = entity_linking_easy_cases()
        successes, failures = step(easy_test_docs)
        entities = documents_to_entity_list(successes)
        for entity, iri, source in zip(entities, iris, sources):
            assert entity.metadata.mappings[0].idx == iri
            assert entity.metadata.mappings[0].source == source

        # test cache
        for _ in range(1000):
            easy_test_docs, iris, sources = entity_linking_easy_cases()
            successes, failures = step(easy_test_docs)
            entities = documents_to_entity_list(successes)
            for entity, iri, source in zip(entities, iris, sources):
                assert entity.metadata.mappings[0].idx == iri
                assert entity.metadata.mappings[0].source == source


def test_sapbert_kb_caching():
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.knowledgebase_path={TINY_CHEMBL_KB_PATH}",
                "SapBertForEntityLinkingStep.rebuild_kb_cache=False",
            ],
        )
        cache_file_location = get_cache_file(TINY_CHEMBL_KB_PATH)
        if os.path.exists(cache_file_location):
            os.remove(cache_file_location)

        step: SapBertForEntityLinkingStep = instantiate(cfg.SapBertForEntityLinkingStep)
        # creating the step should trigger the cache to be build
        assert os.path.exists(cache_file_location)
        assert len(step.kb_ids) > 0
        assert len(step.kb_embeddings) > 0

        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.knowledgebase_path={TINY_CHEMBL_KB_PATH}",
                "SapBertForEntityLinkingStep.rebuild_kb_cache=True",
            ],
        )
        step: SapBertForEntityLinkingStep = instantiate(cfg.SapBertForEntityLinkingStep)
        # creating the step should trigger the cache to be build
        assert os.path.exists(cache_file_location)

        # remove references to the loaded cached objects
        step.kb_ids = None
        step.kb_embeddings = None
        # load the cache from disk
        step.load_kb_cache()
        assert len(step.kb_ids) > 0
        assert len(step.kb_embeddings) > 0
