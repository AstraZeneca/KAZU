import abc
import json
import os
import shutil
from pathlib import Path
from typing import Tuple, Any, Optional

import faiss
import numpy as np
import pandas as pd
import torch


class EmbeddingIndex(abc.ABC):
    """
    a wrapper around an embedding index strategy. Concrete implementations below
    """

    def __init__(self, name: str = "unnamed_index"):
        """
        :param name: a name for this index
        """
        self.name = name

    def add(self, embeddings: torch.Tensor, metadata: pd.DataFrame):
        """
        add embeddings to the index
        :param embeddings: a 2d tensor of embeddings
        :param metadata: a pd.DataFrame of metadata
        :return:
        """

        if embeddings.shape[0] != metadata.shape[0]:
            raise ValueError("embeddings shape not equal to metadata length")
        if self.index is None:
            self.index = self._create_index(embeddings)
            self.metadata = metadata
        else:
            self.index = self._add(embeddings)
            self.metadata = pd.concat([self.metadata, metadata])

    def _create_index(self, embeddings: torch.Tensor) -> Any:
        """
        concrete implementations should implement this to create an index. This should also add the embeddings
        after the index is created
        :param embeddings:
        :return:
        """
        raise NotImplementedError()

    def _add(self, embeddings: torch.Tensor) -> None:
        """
        concrete implementations should implement this to add to an index
        :return:
        """
        raise NotImplementedError()

    def search(self, embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        search the index
        :param embedding: a 2d tensor to query the index with
        :return: a tuple of 2d numpy arrays: distances, nearest neighbours, and a pd.DataFrame of metadata
        """
        distances, neighbors = self._search(embedding)
        return distances, neighbors, self.metadata.iloc[neighbors]

    def _search(self, embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        concrete implementations should implement this to search  an index
        :param embedding: a 2d tensor to query the index with
        :return: a tuple of 2d numpy arrays: distances, nearest neighbours
        """
        raise NotImplementedError()

    def get_path_for_ancillory_file(self, directory: str, filename: str) -> Path:
        path = Path(directory).joinpath(filename)
        return path

    def get_dataframe_path(self, path: str) -> Path:
        return self.get_path_for_ancillory_file(path, "metadata.parquet")

    def get_index_path(self, path: Path) -> Path:
        return self.get_path_for_ancillory_file(path, "index.sapbert")

    def get_index_metadata_path(self, path: Path) -> Path:
        return self.get_path_for_ancillory_file(path, "index_meta.sapbert")

    def save(self, path: str) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets
        :param path:
        :return: a Path to where the index was saved
        """
        directory = Path(path).joinpath(self.name)
        if directory.exists():
            shutil.rmtree(directory)
        os.mkdir(directory)

        self.metadata.to_parquet(self.get_dataframe_path(directory), index=None)
        with open(self.get_index_metadata_path(directory), "w") as f:
            json.dump({"name": self.name}, f)
        self._save(str(self.get_index_path(directory)))
        return directory

    def _save(self, path: str):
        """
        concrete implementations should implement this to save an index
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
        self.metadata = pd.read_parquet(self.get_dataframe_path(path))
        with open(self.get_index_metadata_path(path), "r") as f:
            dict = json.load(f)
            self.name = dict["name"]
        self.index = self._load(str(self.get_index_path(path)))

    def _load(self, path: str) -> Any:
        """
        concrete implementations should implement this to load an index
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

    def __init__(self, return_n_nearest_neighbours: int, *args, **kwargs):
        """
        :param dims: the dimensions of the embeddings
        :param return_n_nearest_neighbours: when a call to .search is made, this may neightbours will be returned
        """
        super().__init__(*args, **kwargs)
        self.return_n_nearest_neighbours = return_n_nearest_neighbours
        self.index = None

    def _load(self, path: str):
        return faiss.read_index(path)

    def _save(self, path: str):
        faiss.write_index(self.index, path)

    def _search(self, embedding: torch.Tensor):
        distances, neighbors = self.index.search(
            embedding.numpy(), self.return_n_nearest_neighbours
        )
        return np.squeeze(distances), np.squeeze(neighbors)

    def _create_index(self, embeddings: torch.Tensor):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.numpy())
        return index

    def _add(self, embeddings: torch.Tensor):
        self.index.add(embeddings.numpy())
        return self.index

    def __len__(self):
        return self.index.ntotal


class TensorEmbeddingIndex(EmbeddingIndex):
    """
    a simple index of torch tensors, that is queried with torch.matmul
    """

    def __init__(self, return_n_nearest_neighbours: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_n_nearest_neighbours = return_n_nearest_neighbours
        self.index = None

    def _load(self, path: str):
        return torch.load(path)

    def _save(self, path: str):
        torch.save(self.index, path)

    def _search(self, embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def _add(self, embeddings: torch.Tensor):
        return torch.cat([self.index, embeddings])

    def _create_index(self, embeddings: torch.Tensor) -> Any:
        return embeddings

    def __len__(self):
        return self.index.shape[0]


class CDistTensorEmbeddingIndex(TensorEmbeddingIndex):
    def _search(self, embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        score_matrix = torch.cdist(embedding, self.index)

        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=False)[
            : self.return_n_nearest_neighbours
        ]
        distances = score_matrix[neighbours]
        return distances.cpu().numpy(), neighbours.cpu().numpy()


class MatMulTensorEmbeddingIndex(TensorEmbeddingIndex):
    def _search(self, embedding: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        score_matrix = torch.matmul(embedding, self.index.T)

        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=True)[
            : self.return_n_nearest_neighbours
        ]
        distances = score_matrix[neighbours]
        distances = 1 / distances
        return distances.cpu().numpy(), neighbours.cpu().numpy()


class EmbeddingIndexFactory:
    """
    since we want to make multiple indices (one per ontology), this factory class helps us build them consistently
    """

    def __init__(self, embedding_index_class_name, return_n_nearest_neighbours):
        self.return_n_nearest_neighbours = return_n_nearest_neighbours
        self.embedding_index_class_name = embedding_index_class_name

    def create_index(self, name: Optional[str] = None):
        if self.embedding_index_class_name == MatMulTensorEmbeddingIndex.__name__:
            return MatMulTensorEmbeddingIndex(
                name=name, return_n_nearest_neighbours=self.return_n_nearest_neighbours
            )
        elif self.embedding_index_class_name == FaissEmbeddingIndex.__name__:
            return FaissEmbeddingIndex(
                name=name, return_n_nearest_neighbours=self.return_n_nearest_neighbours
            )
        elif self.embedding_index_class_name == CDistTensorEmbeddingIndex.__name__:
            return CDistTensorEmbeddingIndex(
                name=name, return_n_nearest_neighbours=self.return_n_nearest_neighbours
            )
        else:
            raise NotImplementedError(
                f"{self.embedding_index_class_name} not implemented in factory"
            )
