from typing import Type
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kazu.modelling.ontology_preprocessing.base import IDX
from kazu.utils.link_index import (
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


@pytest.mark.parametrize(
    "index_type",
    (
        pytest.param(
            FaissEmbeddingIndex,
            marks=pytest.mark.skipif(SKIP_FAISS, reason="Skipping faiss tests as not available"),
        ),
        # TODO: find a better way to test matmul search
        pytest.param(MatMulTensorEmbeddingIndex, marks=pytest.mark.skip),
        CDistTensorEmbeddingIndex,
    ),
)
def test_embedding_index(tmp_path: Path, index_type: Type[Index]):
    index_name = "test_index"
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
    index.save(tmp_path)
    Index.load(tmp_path, index_name)
    hit_info = index.search(query_embedding, top_n=3)
    assert np.array_equal(hit_info.index.to_numpy(), np.array([2, 1, 3]))
    assert np.array_equal(hit_info[IDX].array, np.array([0, 1, 2]))
