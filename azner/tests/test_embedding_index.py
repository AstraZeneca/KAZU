import os

import pandas as pd

from azner.utils.embedding_index import EmbeddingIndexFactory
import torch
import numpy as np

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

metadata = pd.DataFrame.from_dict({"id": [3, 1, 0, 2, 4]})

query_embedding = torch.tensor([index_embedding])


def test_faiss_index():
    factory = EmbeddingIndexFactory("FaissEmbeddingIndex", 3)
    perform_index_tests(factory)


def test_matmul_tensor_index():
    factory = EmbeddingIndexFactory("MatMulTensorEmbeddingIndex", 3)
    perform_index_tests(factory)


def test_cdist_tensor_index():
    factory = EmbeddingIndexFactory("CDistTensorEmbeddingIndex", 3)
    perform_index_tests(factory)


def perform_index_tests(factory: EmbeddingIndexFactory):
    index = factory.create_index()
    index.add(index_embeddings, metadata)
    distance, neighbours, hit_info = index.search(query_embedding)
    assert np.array_equal(neighbours, np.array([2, 1, 3]))
    assert np.array_equal(hit_info["id"].array, np.array([0, 1, 2]))
    metadata_copy = metadata.copy()
    metadata_copy["id"] = metadata_copy["id"] * 2
    index.add(index_embeddings * 2, metadata_copy)
    distance, neighbours, hit_info = index.search(query_embedding)
    assert np.array_equal(neighbours, np.array([2, 1, 3]))
    assert np.array_equal(hit_info["id"].array, np.array([0, 1, 2]))
    index.save("test_index.test")
    index = factory.create_index()
    assert index.index is None
    index.load("test_index.test")
    distance, neighbours, hit_info = index.search(query_embedding)
    assert np.array_equal(neighbours, np.array([2, 1, 3]))
    assert np.array_equal(hit_info["id"].array, np.array([0, 1, 2]))
    os.remove("test_index.test")
    os.remove(index.get_meta_path("test_index.test"))
