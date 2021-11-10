import os
import shutil

import pydash
import pytest
from hydra import initialize_config_module, compose
from hydra.utils import instantiate
from steps.linking.sapbert import SapBertForEntityLinkingStep
from steps.utils.utils import get_cache_dir
from tests.utils import entity_linking_easy_cases, TINY_CHEMBL_KB_PATH, entity_linking_hard_cases


class AcceptanceTestError(Exception):
    def __init__(self, message):
        self.message = message


def test_sapbert_acceptance():
    minimum_pass_score = 0.80
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model.model_name_or_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.ontology_path={os.getenv('SAPBERT_KB_CACHE')}",
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
            if entity.metadata.mappings[0].idx == iri:
                hits.append(entity)
            else:
                misses.append(
                    (
                        entity,
                        iri,
                    )
                )

        for entity, iri in misses:
            print(f"missed {entity.match}: got {entity.metadata.mappings[0].idx}, wanted {iri} ")
        total = len(hits) + len(misses)
        score = len(hits) / total
        if score < minimum_pass_score:
            raise AcceptanceTestError(
                f"sapbert scored {score}, which did not reach minimum pass score of {minimum_pass_score}"
            )


@pytest.mark.timeout(10)
def test_sapbert_step():
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model.model_name_or_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.ontology_path={TINY_CHEMBL_KB_PATH}",
                "SapBertForEntityLinkingStep.rebuild_ontology_cache=True",
            ],
        )

        step = instantiate(cfg.SapBertForEntityLinkingStep)
        easy_test_docs, iris, sources = entity_linking_easy_cases()
        successes, failures = step(easy_test_docs)
        entities = pydash.flatten([x.get_entities() for x in successes])
        for entity, iri, source in zip(entities, iris, sources):
            assert entity.metadata.mappings[0].idx == iri
            assert entity.metadata.mappings[0].source == source

        # test cache
        for _ in range(1000):
            easy_test_docs, iris, sources = entity_linking_easy_cases()
            successes, failures = step(easy_test_docs)
            entities = pydash.flatten([x.get_entities() for x in successes])
            for entity, iri, source in zip(entities, iris, sources):
                assert entity.metadata.mappings[0].idx == iri
                assert entity.metadata.mappings[0].source == source


def test_sapbert_ontology_caching():
    with initialize_config_module(config_module="azner.conf"):

        # no cache found, so should rebuidl automatically
        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model.model_name_or_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.ontology_path={TINY_CHEMBL_KB_PATH}",
                "SapBertForEntityLinkingStep.rebuild_ontology_cache=False",
            ],
        )
        cache_file_location = get_cache_dir(TINY_CHEMBL_KB_PATH, create_if_not_exist=False)
        if os.path.exists(cache_file_location):
            shutil.rmtree(cache_file_location)

        step: SapBertForEntityLinkingStep = instantiate(cfg.SapBertForEntityLinkingStep)
        assert os.path.exists(cache_file_location)
        assert len(step.ontology_index_dict) > 0

        # force cache rebuild
        cfg = compose(
            config_name="config",
            overrides=[
                f"SapBertForEntityLinkingStep.model.model_name_or_path={os.getenv('SapBertForEntityLinkingModelPath')}",
                f"SapBertForEntityLinkingStep.ontology_path={TINY_CHEMBL_KB_PATH}",
                "SapBertForEntityLinkingStep.rebuild_ontology_cache=True",
            ],
        )
        step: SapBertForEntityLinkingStep = instantiate(cfg.SapBertForEntityLinkingStep)
        # creating the step should trigger the cache to be build
        assert os.path.exists(cache_file_location)
        # remove references to the loaded cached objects
        step.ontology_index_dict.clear()
        # load the cache from disk
        step.ontology_index_dict = step.load_ontology_index_dict_from_cache()
        assert len(step.ontology_index_dict) > 0
