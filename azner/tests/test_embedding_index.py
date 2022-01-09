import tempfile
from typing import Type

import numpy as np
import pandas as pd
import pytest
import torch

from azner.modelling.ontology_preprocessing.base import IDX
from azner.utils.link_index import (
    MatMulTensorEmbeddingIndex,
    CDistTensorEmbeddingIndex,
    FaissEmbeddingIndex,
    Index,
)

index_embedding = [1.1, 2.2, 3.3, 4.4, 5.5]
index_embeddings = torch.tensor(
    [
        [x + 3 for x in index_embedding],
        [x + 1 for x in index_embedding],
        index_embedding,
        [x + 2 for x in index_embedding],
        [x + 4 for x in index_embedding],
    ]
)

metadata = pd.DataFrame.from_dict({IDX: [3, 1, 0, 2, 4]})

query_embedding = torch.tensor([index_embedding])

SKIP_FAISS = False
try:
    import faiss  # noqa
except ImportError:
    SKIP_FAISS = True


@pytest.mark.skipif(SKIP_FAISS, reason="Skipping faiss tests as not available")
def test_faiss_index():
    perform_index_tests(FaissEmbeddingIndex)


# TODO: find a better way to test matmul search
@pytest.mark.skip
def test_matmul_tensor_index():
    perform_index_tests(MatMulTensorEmbeddingIndex)


def test_cdist_tensor_index():
    perform_index_tests(CDistTensorEmbeddingIndex)


def perform_index_tests(index_type: Type[Index]):
    with tempfile.TemporaryDirectory() as f:
        index_name = "test_index"
        index_save_dir = f
        index = index_type(name=index_name)
        index.add(index_embeddings, metadata)
        df = index.search(query_embedding, top_n=3)
        assert np.array_equal(df.index.to_numpy(), np.array([2, 1, 3]))
        assert np.array_equal(df[IDX].array, np.array([0, 1, 2]))
        metadata_copy = metadata.copy()
        metadata_copy[IDX] = metadata_copy[IDX] * 2
        index.add(index_embeddings * 2, metadata_copy)
        df = index.search(query_embedding, top_n=3)
        assert np.array_equal(df.index.to_numpy(), np.array([2, 1, 3]))
        assert np.array_equal(df[IDX].array, np.array([0, 1, 2]))
        index.save(index_save_dir)
        Index.load(index_save_dir, index_name)
        hit_info = index.search(query_embedding, top_n=3)
        assert np.array_equal(hit_info.index.to_numpy(), np.array([2, 1, 3]))
        assert np.array_equal(hit_info[IDX].array, np.array([0, 1, 2]))
