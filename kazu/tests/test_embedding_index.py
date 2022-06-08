import tempfile
from pathlib import Path
from typing import Type, Union
from unittest.mock import patch

import pytest
from hydra.utils import instantiate

from kazu.modelling.linking.sapbert.train import PLSapbertModel
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.tests.utils import DummyParser, requires_model_pack
from kazu.utils.link_index import (
    MatMulTensorEmbeddingIndex,
    CDistTensorEmbeddingIndex,
    SAPBERT_SCORE,
)
from kazu.utils.utils import get_cache_dir


@pytest.mark.parametrize(
    "index_type",
    (
        MatMulTensorEmbeddingIndex,
        CDistTensorEmbeddingIndex,
    ),
)
@requires_model_pack
def test_embedding_index(
    index_type: Union[Type[CDistTensorEmbeddingIndex], Type[MatMulTensorEmbeddingIndex]],
    kazu_test_config,
):
    embedding_model = instantiate(kazu_test_config.PLSapbertModel)
    with tempfile.TemporaryDirectory("kazu") as dummy_parser_path:
        parser = DummyParser(dummy_parser_path)
        cache_dir = get_cache_dir(
            dummy_parser_path,
            prefix=f"{parser.name}_{index_type(DummyParser(dummy_parser_path)).__class__.__name__}",
            create_if_not_exist=False,
        )

        assert_cache_built(cache_dir, embedding_model, index_type, parser, dummy_parser_path)
        asset_cache_loaded(cache_dir, embedding_model, index_type, parser, dummy_parser_path)


def assert_search_is_working(
    parser: OntologyParser,
    index_type: Union[Type[CDistTensorEmbeddingIndex], Type[MatMulTensorEmbeddingIndex]],
    embedding_model: PLSapbertModel,
):
    index = index_type(parser)
    index.set_embedding_model(embedding_model)
    index.load_or_build_cache(False)
    scores = set()
    for _ in range(5):
        query_embedding = embedding_model.get_embeddings_for_strings(["4"])
        hits = list(index.search(query_embedding))
        assert len(hits) == 1
        scores.add(hits[0].metrics[SAPBERT_SCORE])
    # multiple calls to the same string should return same score
    assert len(scores) == 1


def asset_cache_loaded(
    cache_dir: Path,
    embedding_model: PLSapbertModel,
    index_type: Union[Type[CDistTensorEmbeddingIndex], Type[MatMulTensorEmbeddingIndex]],
    parser: OntologyParser,
    dummy_parser_path: str,
):
    # now test that the prebuilt cache is loaded

    with patch("kazu.utils.link_index.EmbeddingIndex.load") as load:
        index = index_type(DummyParser(dummy_parser_path))
        index.set_embedding_model(embedding_model)
        index.load_or_build_cache(False)
        load.assert_called_with(cache_dir)
    # now actually load the cache and check search is working
    assert_search_is_working(parser, index_type, embedding_model)


def assert_cache_built(
    cache_dir: Path,
    embedding_model: PLSapbertModel,
    index_type: Union[Type[CDistTensorEmbeddingIndex], Type[MatMulTensorEmbeddingIndex]],
    parser: OntologyParser,
    dummy_parser_path: str,
):
    # test that the cache is built

    with patch("kazu.utils.link_index.EmbeddingIndex.build_ontology_cache") as build_ontology_cache:
        index = index_type(DummyParser(dummy_parser_path))
        index.set_embedding_model(embedding_model)
        index.load_or_build_cache(False)
        build_ontology_cache.assert_called_with(cache_dir)
    # now actually build the cache and check search is working
    assert_search_is_working(parser, index_type, embedding_model)
