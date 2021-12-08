import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pytorch_lightning import Trainer

from azner.modelling.linking.sapbert.train import PLSapbertModel
from azner.modelling.ontology_preprocessing.base import (
    OntologyParser,
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
)
from azner.utils.caching import (
    EmbeddingOntologyCacheManager,
    DictionaryOntologyCacheManager,
    CachedIndexGroup,
)
from azner.utils.link_index import Index
from tests.utils import BERT_TEST_MODEL_PATH

DUMMY_SOURCE = "test_parser"
DUMMY_DATA = {
    IDX: ["first", "first", "second", "second", "third"],
    DEFAULT_LABEL: ["1", "1", "2", "2", "3"],
    SYN: ["1", "one", "2", "two", "3"],
    MAPPING_TYPE: ["int", "text", "int", "text", "int"],
}


class DummyParser(OntologyParser):
    name = DUMMY_SOURCE

    def parse_to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame.from_dict(DUMMY_DATA)


@pytest.mark.parametrize("index_type", ["MatMulTensorEmbeddingIndex", "CDistTensorEmbeddingIndex"])
def test_embedding_cache_manager(index_type):
    with tempfile.TemporaryDirectory() as f:
        in_path = Path(f).joinpath(DUMMY_SOURCE)
        os.mkdir(in_path)
        model = PLSapbertModel(model_name_or_path=BERT_TEST_MODEL_PATH)
        trainer = Trainer(logger=False)
        parser = DummyParser(in_path=str(in_path))
        manager = EmbeddingOntologyCacheManager(
            model=model,
            batch_size=16,
            trainer=trainer,
            dl_workers=0,
            ontology_partition_size=20,
            index_type=index_type,
            rebuild_cache=False,
            parsers=[parser],
        )

        index = manager.get_or_create_ontology_indices()[0]
        assert isinstance(index, Index)
        assert len(index) == len(set(DUMMY_DATA[IDX]))
        # there should now be a cached file at the parent of the target in_path
        cache_dir = [x for x in os.listdir(in_path.parent) if x.startswith("cached")][0]
        cache_dir = in_path.parent.joinpath(cache_dir)
        assert os.path.exists(cache_dir)
        # now check the load cache method is called
        with patch(
            "azner.utils.caching.EmbeddingOntologyCacheManager.load_ontology_from_cache"
        ) as load_ontology_from_cache:
            manager = EmbeddingOntologyCacheManager(
                model=model,
                batch_size=16,
                trainer=trainer,
                dl_workers=0,
                ontology_partition_size=20,
                index_type=index_type,
                rebuild_cache=False,
                parsers=[parser],
            )
            manager.get_or_create_ontology_indices()
            load_ontology_from_cache.assert_called_with(cache_dir, parser)


def test_dictionary_cache_manager():
    with tempfile.TemporaryDirectory() as f:
        in_path = Path(f).joinpath(DUMMY_SOURCE)
        os.mkdir(in_path)
        parser = DummyParser(in_path=str(in_path))
        manager = DictionaryOntologyCacheManager(
            index_type="DictionaryIndex", parsers=[parser], rebuild_cache=False
        )
        index = manager.get_or_create_ontology_indices()[0]
        assert isinstance(index, Index)
        assert len(index) == len(set(DUMMY_DATA[IDX]))
        # there should now be a cached file at the parent of the target in_path
        cache_dir = [x for x in os.listdir(in_path.parent) if x.startswith("cached")][0]
        cache_dir = in_path.parent.joinpath(cache_dir)
        assert os.path.exists(cache_dir)
        # now check the load cache method is called
        with patch(
            "azner.utils.caching.DictionaryOntologyCacheManager.load_ontology_from_cache"
        ) as load_ontology_from_cache:
            manager = DictionaryOntologyCacheManager(
                index_type="DictionaryIndex", parsers=[parser], rebuild_cache=False
            )
            manager.get_or_create_ontology_indices()
            load_ontology_from_cache.assert_called_with(cache_dir, parser)


def test_cached_index_group():
    with tempfile.TemporaryDirectory() as f:
        in_path1 = Path(f).joinpath(DUMMY_SOURCE + "1")
        os.mkdir(in_path1)

        parser1 = DummyParser(in_path=str(in_path1))
        parser1.name = "ontology_1"
        in_path2 = Path(f).joinpath(DUMMY_SOURCE + "2")
        os.mkdir(in_path2)
        parser2 = DummyParser(in_path=str(in_path2))
        parser2.name = "ontology_2"
        manager = DictionaryOntologyCacheManager(
            index_type="DictionaryIndex", rebuild_cache=False, parsers=[parser1, parser2]
        )

        entity_ontology_mappings = {
            "entity_class_1": [parser1.name],
            "entity_class_2": [parser2.name],
        }
        cached_index_group = CachedIndexGroup(
            cache_managers=[manager],
            entity_class_to_ontology_mappings=entity_ontology_mappings,
        )
        cached_index_group.load()
        ontology_1_mappings = cached_index_group.search(
            query="two", entity_class="entity_class_1", namespace="test"
        )
        assert ontology_1_mappings[0].idx == "second"
        assert ontology_1_mappings[0].source == parser1.name

        ontology_2_mappings = cached_index_group.search(
            query="3", entity_class="entity_class_2", namespace="test2"
        )
        assert ontology_2_mappings[0].idx == "third"
        assert ontology_2_mappings[0].source == parser2.name
