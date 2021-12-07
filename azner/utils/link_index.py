import abc
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
import torch
from rapidfuzz import process, fuzz

from azner.data.data import LINK_SCORE
from azner.modelling.ontology_preprocessing.base import SYN, IDX, MAPPING_TYPE

logger = logging.getLogger(__name__)


class Index(abc.ABC):
    """
    base class for all indices.
    """

    def __init__(
        self,
        name: str = "unnamed_index",
    ):
        """

        :param path: path to parquet file of synonyms
        :param name: a name for this index
        :param fuzzy: use fuzzy matching
        :param score_cutoff: minimum score for fuzzy matching
        :param top_n: number of hits to return
        """
        self.name = name
        self.metadata = None

    def _search(
        self, query: Any, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        calls to search should return a :py:class:`<pd.DataFrame>`

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def search(
        self, query: Any, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> pd.DataFrame:
        """
        search the index

        :param query: a string to query against
        :return: a df of hits
        """
        hit_df, scores = self._search(query, score_cutoff=score_cutoff, top_n=top_n, **kwargs)
        hit_df[LINK_SCORE] = scores
        return hit_df

    def save(self, path: str) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets

        :param path:
        :return: a Path to where the index was saved
        """
        directory = Path(path).joinpath(self.name)
        if directory.exists():
            shutil.rmtree(directory)
        os.makedirs(directory)
        with open(self.get_index_path(directory), "wb") as f:
            pickle.dump(self, f)
        self.metadata.to_parquet(self.get_dataframe_path(directory), index=None)

        self._save(str(self.get_index_data_path(directory)))
        return directory

    @staticmethod
    def load(path: str, name: str):
        """
        load from disk

        :param path:
        :return:
        """
        path = Path(path).joinpath(name)
        with open(Index.get_index_path(path), "rb") as f:
            index = pickle.load(f)
        index.metadata = pd.read_parquet(Index.get_dataframe_path(path))
        index._load(Index.get_index_data_path(path))
        return index

    @staticmethod
    def get_path_for_ancillory_file(directory: Path, filename: str) -> Path:
        path = Path(directory).joinpath(filename)
        return path

    @staticmethod
    def get_dataframe_path(path: Path) -> Path:
        return Index.get_path_for_ancillory_file(path, "ontology_metadata.parquet")

    @staticmethod
    def get_index_path(path: Path) -> Path:
        return Index.get_path_for_ancillory_file(path, "index.pkl")

    @staticmethod
    def get_index_data_path(path: Path) -> Path:
        return Index.get_path_for_ancillory_file(path, "index.data")

    def _save(self, path: str):
        """
        concrete implementations should implement this to save an index

        :param path:
        :return:
        """
        raise NotImplementedError()

    def __getstate__(self):
        return self.name

    def __setstate__(self, state):
        self.name = state

    def _load(self, path: str) -> None:
        """
        concrete implementations should implement this to load any fields required by the index

        :param path:
        :return:
        """
        raise NotImplementedError()

    def add(self, *args, **kwargs):
        raise NotImplementedError()


class DictionaryIndex(Index):
    """
    a simple dictionary index for linking
    """

    def __init__(self, name: str = "unnamed_index"):
        """

        :param path: path to parquet file of synonyms
        :param name: a name for this index
        :param fuzzy: use fuzzy matching
        :param score_cutoff: minimum score for fuzzy matching
        :param top_n: number of hits to return
        """
        super().__init__(name)
        self.synonym_df = None

    def _search(
        self, query: Any, score_cutoff: int = 99.0, top_n: int = 20, fuzzy: bool = True, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        search the index

        :param query: a string to query against
        :return: a df of hits
        """
        query = query.lower()
        if not fuzzy:
            idx_list = self.synonym_df[self.synonym_df[SYN] == query]
            # score 100 for exact matches
            scores = [100.0 for _ in range(len(idx_list))]
        else:
            hits = process.extract(
                query,
                self.synonym_df[SYN].tolist(),
                scorer=fuzz.WRatio,
                limit=top_n,
                score_cutoff=score_cutoff,
            )
            locs = [x[2] for x in hits]
            idx_list = self.synonym_df.iloc[locs]
            scores = [x[1] for x in hits]
        hit_df = self.metadata.join(idx_list, on=[IDX], how="left")
        return hit_df, np.array(scores)

    def _load(self, path: str) -> Any:
        self.synonym_df = pd.read_parquet(path)

    def _save(self, path: str):
        self.synonym_df.to_parquet(path, index=None)

    def add(self, synonym_df: pd.DataFrame, metadata_df: pd.DataFrame):
        if self.synonym_df is None and self.metadata is None:
            self.synonym_df = synonym_df
            self.metadata = metadata_df
        else:
            self.synonym_df = pd.concat([self.synonym_df, synonym_df])
            self.metadata = pd.concat([self.metadata, metadata_df])

    def __len__(self):
        return self.metadata.shape[0]


class EmbeddingIndex(Index):
    """
    a wrapper around an embedding index strategy. Concrete implementations below
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)

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

    def __len__(self):
        """
        the number of embeddings in the index

        :return:
        """
        raise NotImplementedError()

    def _search_func(
        self, query: torch.Tensor, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        should be implemented
        :param query:
        :param score_cutoff:
        :param top_n:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def _search(
        self, query: torch.Tensor, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        distances, neighbours = self._search_func(
            query=query, top_n=top_n, score_cutoff=score_cutoff, **kwargs
        )
        hit_df = self.metadata.iloc[neighbours].copy()
        # all mapping types are inferred for embedding based matches
        hit_df[MAPPING_TYPE] = "inferred"
        return hit_df, distances


class FaissEmbeddingIndex(EmbeddingIndex):
    """
    an embedding index that uses faiss.IndexFlatL2
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)
        try:
            import faiss

            self.faiss = faiss
        except ImportError:
            raise RuntimeError(f"faiss is not installed. Cannot use {self.__class__.__name__}")

        self.index = None

    def _load(self, path: str):
        return self.faiss.read_index(path)

    def _save(self, path: str):
        self.faiss.write_index(self.index, path)

    def _search_func(
        self, query: torch.Tensor, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances, neighbours = self.index.search(query.numpy(), top_n)
        return distances.numpy(), neighbours.numpy()

    def _create_index(self, embeddings: torch.Tensor):
        index = self.faiss.IndexFlatL2(embeddings.shape[1])
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

    def __init__(self, name: str):

        super().__init__(name)
        self.index = None

    def _load(self, path: str):
        self.index = torch.load(path, map_location="cpu")

    def _save(self, path: str):
        torch.save(self.index, path)

    def _add(self, embeddings: torch.Tensor):
        return torch.cat([self.index, embeddings])

    def _create_index(self, embeddings: torch.Tensor) -> Any:
        return embeddings

    def __len__(self):
        return self.index.shape[0]


class CDistTensorEmbeddingIndex(TensorEmbeddingIndex):
    """
    Calculate embedding based on cosine distance
    """

    def _search_func(
        self, query: torch.Tensor, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        score_matrix = torch.cdist(query, self.index)

        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=False)[:top_n]
        distances = score_matrix[neighbours]
        return distances.cpu().numpy(), neighbours.cpu().numpy()


class MatMulTensorEmbeddingIndex(TensorEmbeddingIndex):
    """
    calculate embedding based on MatMul
    """

    def _search_func(
        self, query: torch.Tensor, score_cutoff: int = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        score_matrix = torch.matmul(query, self.index.T)

        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=True)[:top_n]
        distances = score_matrix[neighbours]
        distances = 100 - (1 / distances)
        return distances.cpu().numpy(), neighbours.cpu().numpy()
