import os
from steps.utils.embedding_index import EmbeddingIndexFactory
import torch
import numpy as np

index_embedding = [1.1, 2.2, 3.3, 4.4, 5.5]
index_embeddings = torch.tensor(
    [
        [pow(x, 5) for x in index_embedding],
        [pow(x, 2) for x in index_embedding],
        index_embedding,
        [pow(x, 3) for x in index_embedding],
        [pow(x, 6) for x in index_embedding],
    ]
)

query_embedding = torch.tensor([index_embedding])


def test_faiss_index():
    factory = EmbeddingIndexFactory("FaissEmbeddingIndex", 3)
    perform_index_tests(factory)


def test_tensor_index():
    factory = EmbeddingIndexFactory("TensorEmbeddingIndex", 3)
    perform_index_tests(factory)


def perform_index_tests(factory: EmbeddingIndexFactory):
    index = factory.create_index()
    index.add(index_embeddings)
    distance, neighbours = index.search(query_embedding)
    assert np.array_equal(neighbours, np.array([2, 1, 3]))
    index.add(index_embeddings * 2)
    distance, neighbours = index.search(query_embedding)
    assert np.array_equal(neighbours, np.array([2, 7, 1]))
    index.save("test_index.test")
    index = factory.create_index()
    assert index.index is None
    index.load("test_index.test")
    distance, neighbours = index.search(query_embedding)
    assert np.array_equal(neighbours, np.array([2, 7, 1]))
    os.remove("test_index.test")
