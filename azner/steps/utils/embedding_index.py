import abc
from typing import Tuple

import faiss
import torch
import numpy as np


class EmbeddingIndex(abc.ABC):
    """
    a wrapper around an embedding index strategy. Concrete implementations below
    """

    def add(self, embeddings: torch.Tensor):
        """
        add embeddings to the index
        :param embeddings: a 2d tensor of embeddings
        :return:
        """
        raise NotImplementedError()

    def search(self, embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        search the index
        :param embedding: a 2d tensor to query the index with
        :return: a tuple of 2d numpy arrays: distances, and nearest neighbours
        """
        raise NotImplementedError()

    def save(self, path: str):
        """
        save to disk
        :param path:
        :return:
        """
        raise NotImplementedError()

    def load(self, path: str):
        """
        load from disk
        :param path:
        :return:
        """
        raise NotImplementedError()

    def __len__(self):
        """
        the number of embeddings in the index
        :return:
        """
        raise NotImplementedError()


class FaissEmbeddingIndex(EmbeddingIndex):
    """
    an embedding index that uses faiss.IndexFlatL2
    """

    def __init__(self, return_n_nearest_neighbours: int):
        """
        :param dims: the dimensions of the embeddings
        :param return_n_nearest_neighbours: when a call to .search is made, this may neightbours will be returned
        """
        self.return_n_nearest_neighbours = return_n_nearest_neighbours
        self.index = None

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def search(self, embedding: torch.Tensor):
        distances, neighbors = self.index.search(
            embedding.numpy(), self.return_n_nearest_neighbours
        )
        return distances, neighbors

    def add(self, embeddings: torch.Tensor):
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.add(embeddings)
        else:
            self.index.add(embeddings.numpy())

    def __len__(self):
        return self.index.ntotal


class TensorEmbeddingIndex(EmbeddingIndex):
    """
    a simple index of torch tensors, that is queried with torch.matmul
    """

    def __init__(self, return_n_nearest_neighbours: int):
        self.return_n_nearest_neighbours = return_n_nearest_neighbours
        self.index = None

    def load(self, path: str):
        self.index = torch.load(path)

    def save(self, path: str):
        torch.save(self.index, path)

    def search(self, embedding: torch.Tensor):
        score_matrix = torch.matmul(embedding, self.index.T)

        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=True)[
            : self.return_n_nearest_neighbours
        ]
        distances = score_matrix[neighbours]
        distances = 1 / distances
        return distances.cpu().numpy(), neighbours.cpu().numpy()

    def add(self, embeddings: torch.Tensor):
        if self.index is None:
            self.index = embeddings
        else:
            self.index = torch.cat([self.index, embeddings])

    def __len__(self):
        return self.index.shape[0]


class EmbeddingIndexFactory:
    """
    since we want to make multiple indices (one per ontology), this factory class helps us build them consistently
    """

    def __init__(self, embedding_index_class_name, return_n_nearest_neighbours):
        self.return_n_nearest_neighbours = return_n_nearest_neighbours
        self.embedding_index_class_name = embedding_index_class_name

    def create_index(self):
        if self.embedding_index_class_name == TensorEmbeddingIndex.__name__:
            return TensorEmbeddingIndex(self.return_n_nearest_neighbours)
        elif self.embedding_index_class_name == FaissEmbeddingIndex.__name__:
            return FaissEmbeddingIndex(self.return_n_nearest_neighbours)
