import copy
from pathlib import Path
from typing import Type, Union, Iterable, Dict

import numpy as np
import pytest
import torch

from kazu.data.data import Mapping
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

metadata: Dict[str, Dict] = {"3": {}, "1": {}, "0": {}, "2": {}, "4": {}}

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
def test_embedding_index(
    tmp_path: Path,
    index_type: Union[
        Type[CDistTensorEmbeddingIndex], Type[MatMulTensorEmbeddingIndex], Type[FaissEmbeddingIndex]
    ],
):
    index_name = "test_index"
    index = index_type(name=index_name)
    index.add(data=index_embeddings, metadata=copy.deepcopy(metadata))
    mappings: Iterable[Mapping] = list(index.search(query_embedding, original_string="", top_n=3))
    for mapping, idx in zip(mappings, np.array(["0", "1", "2"])):
        assert mapping.idx == idx
    # assert np.array_equal(df.index.to_numpy(), np.array(["2", "1", "3"]))
    # assert np.array_equal(df[IDX].array.to_numpy(), np.array(["0", "1", "2"]))
    metadata_copy = copy.deepcopy(metadata)
    metadata_copy = {k * 2: v for k, v in metadata_copy.items()}
    index.add(data=(index_embeddings * 2), metadata=metadata_copy)
    mappings = index.search(query_embedding, original_string="", top_n=3)
    for mapping, idx in zip(mappings, np.array(["0", "1", "2"])):
        assert mapping.idx == idx
    index.save(tmp_path)
    Index.load(tmp_path)
    mappings = index.search(query_embedding, original_string="", top_n=3)
    for mapping, idx in zip(mappings, np.array(["0", "1", "2"])):
        assert mapping.idx == idx
