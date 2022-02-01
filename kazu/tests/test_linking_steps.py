from unittest import mock

import pydash
from pytorch_lightning import Trainer

import kazu.utils.caching
from kazu.modelling.linking.sapbert.train import PLSapbertModel
from kazu.steps import DictionaryEntityLinkingStep
from kazu.steps import SapBertForEntityLinkingStep
from kazu.tests.utils import (
    entity_linking_easy_cases,
    BERT_TEST_MODEL_PATH,
    MockedCachedIndexGroup,
)
from kazu.utils.caching import (
    CachedIndexGroup,
    EmbeddingIndexCacheManager,
    DictionaryIndexCacheManager,
)


def test_dictionary_step():
    easy_test_docs, iris, sources = entity_linking_easy_cases()
    ds = MockedCachedIndexGroup(iris=iris, sources=sources)
    with mock.patch.object(
        kazu.utils.caching.CachedIndexGroup, "search", side_effect=ds.mock_search
    ):
        manager = DictionaryIndexCacheManager(
            index_type="DictionaryIndex", rebuild_cache=False, parsers=[]
        )
        cached_index_group = CachedIndexGroup(
            cache_managers=[manager],
            entity_class_to_ontology_mappings={},
        )
        step = DictionaryEntityLinkingStep(depends_on=[], index_group=cached_index_group)
        assert_step_runs(easy_test_docs, iris, sources, step)


def test_sapbert_step():
    easy_test_docs, iris, sources = entity_linking_easy_cases()
    ds = MockedCachedIndexGroup(iris=iris, sources=sources)
    with mock.patch.object(
        kazu.utils.caching.CachedIndexGroup, "search", side_effect=ds.mock_search
    ):
        model = PLSapbertModel(model_name_or_path=BERT_TEST_MODEL_PATH)
        trainer = Trainer(logger=False)
        manager = EmbeddingIndexCacheManager(
            model=model,
            batch_size=16,
            trainer=trainer,
            dl_workers=0,
            ontology_partition_size=20,
            index_type="MatMulTensorEmbeddingIndex",
            rebuild_cache=False,
            parsers=[],
        )
        cached_index_group = CachedIndexGroup(
            cache_managers=[manager],
            entity_class_to_ontology_mappings={},
        )
        step = SapBertForEntityLinkingStep(depends_on=[], index_group=cached_index_group)
        assert_step_runs(easy_test_docs, iris, sources, step)


def assert_step_runs(easy_test_docs, iris, sources, step):
    successes, failures = step(easy_test_docs)
    entities = pydash.flatten([x.get_entities() for x in successes])
    for entity, iri, source in zip(entities, iris, sources):
        assert entity.mappings[0].idx == iri
        assert entity.mappings[0].source == source
