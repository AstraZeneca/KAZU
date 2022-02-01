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

from kazu.data.data import LINK_SCORE
from kazu.modelling.ontology_preprocessing.base import SYN, IDX, MAPPING_TYPE, DEFAULT_LABEL
from kazu.utils.utils import PathLike, as_path

logger = logging.getLogger(__name__)


class Index(abc.ABC):
    """
    base class for all indices.
    """

    column_type_dict = {SYN: str, IDX: str, MAPPING_TYPE: list, DEFAULT_LABEL: str}

    def __init__(
        self,
        name: str = "unnamed_index",
    ):
        """

        :param name: the name of the index. default is unnamed_index
        """
        self.name = name
        self.metadata: pd.DataFrame
        self.index: Any

    def _search(self, query: Any, **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        subclasses should implement this method, which describes the logic to actually perform the search
        calls to search should return a tuple of  :py:class:`pandas.DataFrame` and numpy.ndarray.
        the dataframe should be a slice of self.metadata, and the ndarray should be a 1 d array equal to the row count,
        with the scores for each hit
        :param query: the query to use
        :param kwargs: any other arguments that are required
        :return:
        """
        raise NotImplementedError()

    def search(self, query: Any, **kwargs) -> pd.DataFrame:
        """
        search the index
        :param query: the query to use
        :param kwargs: any other arguments to pass to self._search
        :return: a :class:`pandas.DataFrame` of hits, with a score column
        """
        hit_df, scores = self._search(query, **kwargs)
        hit_df[LINK_SCORE] = scores
        return hit_df

    def save(self, path: PathLike) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets

        :param path:
        :return: a Path to where the index was saved
        """
        directory = as_path(path).joinpath(self.name)
        if directory.exists():
            shutil.rmtree(directory)
        os.makedirs(directory)
        with open(self.get_index_path(directory), "wb") as f:
            pickle.dump(self, f)
        self.metadata.to_parquet(self.get_dataframe_path(directory), index=None)

        self._save(self.get_index_data_path(directory))
        return directory

    @classmethod
    def load(cls, path: PathLike, name: str):
        """
        load from disk
        :param path: the parent path of the index
        :param name: the name of the index within the parent path
        :return:
        """

        root_path = as_path(path)
        path = root_path.joinpath(name)
        with open(cls.get_index_path(path), "rb") as f:
            index = pickle.load(f)
        index.metadata = cls.check_dataframe_types(pd.read_parquet(cls.get_dataframe_path(path)))
        index._load(cls.get_index_data_path(path))
        return index

    @staticmethod
    def get_dataframe_path(path: Path) -> Path:
        return path.joinpath("ontology_metadata.parquet")

    @staticmethod
    def get_index_path(path: Path) -> Path:
        return path.joinpath("index.pkl")

    @staticmethod
    def get_index_data_path(path: Path) -> Path:
        return path.joinpath("index.data")

    def _save(self, path: PathLike):
        """
        concrete implementations should implement this to save any data specific to the implementation. This method is
        called by self.save
        :param path:
        :return:
        """
        raise NotImplementedError()

    def __getstate__(self):
        return self.name

    def __setstate__(self, state):
        self.name = state

    def _load(self, path: PathLike) -> None:
        """
        concrete implementations should implement this to load any data specific to the implementation. This method is
        called by self.load

        :param path:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def check_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        certain fields in the metadata dataframe must be of a certain type. This method modifies a df to ensure the
        types are consistently used.
        :param df:
        :return:
        """
        for column_name, _type in Index.column_type_dict.items():
            if column_name in df.columns:
                df[column_name] = df[column_name].apply(_type)
        # all df indices must be of type str
        df.index = df.index.astype(str)
        return df

    def __len__(self) -> int:
        """
        should return the size of the index
        :return:
        """
        raise NotImplementedError()


class DictionaryIndex(Index):
    """
    a simple dictionary index for linking
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)
        self.synonym_df: pd.DataFrame

    def _search(
        self, query: str, score_cutoff: float = 99.0, top_n: int = 20, fuzzy: bool = True, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        search the index
        :param query: a string of text
        :param score_cutoff: minimum rapidfuzz match score. ignored if fuzzy-False
        :param top_n: return up to this many hits
        :param fuzzy: use rapidfuzz fuzzy matching
        :param kwargs:
        :return:
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
        hit_df = idx_list.set_index(IDX).join(self.metadata, how="left")
        return hit_df, np.array(scores)

    def _load(self, path: PathLike) -> Any:
        self.synonym_df = self.check_dataframe_types(pd.read_parquet(path))

    def _save(self, path: PathLike):
        self.synonym_df.to_parquet(path, index=None)

    def add(self, synonym_df: pd.DataFrame, metadata_df: pd.DataFrame):
        """
        add data to the index. Two indices are required - synonyms and metadata. Metadata should have a primary key
        (IDX) and synonyms should use IDX as a foreign key
        :param synonym_df: synonyms dataframe
        :param metadata_df: metadata dataframe
        :return:
        """
        metadata_df = self.check_dataframe_types(metadata_df)
        synonym_df = self.check_dataframe_types(synonym_df)
        if not hasattr(self, "synonym_df") and not hasattr(self, "metadata"):
            self.synonym_df = synonym_df
            self.metadata = metadata_df
        else:
            self.synonym_df = pd.concat([self.synonym_df, synonym_df])
            self.metadata = pd.concat([self.metadata, metadata_df])

    def __len__(self):
        return len(self.metadata)


class EmbeddingIndex(Index):
    """
    a wrapper around an embedding index strategy.
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)

    def add(self, embeddings: torch.Tensor, metadata_df: pd.DataFrame):
        """
        add embeddings to the index

        :param embeddings: a 2d tensor of embeddings
        :param metadata_df: a pd.DataFrame of metadata
        :return:
        """
        metadata_df = self.check_dataframe_types(metadata_df)
        if len(embeddings) != len(metadata_df):
            raise ValueError("embeddings length not equal to metadata length")
        if not hasattr(self, "index"):
            self.index = self._create_index(embeddings)
            self.metadata = metadata_df
        else:
            self._add(embeddings)
            self.metadata = pd.concat([self.metadata, metadata_df])

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
        concrete implementations should implement this to add embeddings to the index

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
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        should be implemented
        :param query: a string of text
        :param score_cutoff: minimum rapidfuzz match score
        :param top_n: return up to this many hits
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def _search(
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        distances, neighbours = self._search_func(
            query=query, top_n=top_n, score_cutoff=score_cutoff, **kwargs
        )
        hit_df = self.metadata.iloc[neighbours.astype(str)].copy()
        self.add_inferred_mapping_type_to_dataframe(hit_df)
        return hit_df, distances

    @staticmethod
    def add_inferred_mapping_type_to_dataframe(df: pd.DataFrame):
        # all mapping types are inferred for embedding based matches
        df[MAPPING_TYPE] = [["inferred"] for _ in range(df.shape[0])]


class FaissEmbeddingIndex(EmbeddingIndex):
    """
    an embedding index that uses faiss.IndexFlatL2
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)
        self.import_faiss()

    def import_faiss(self):
        try:
            import faiss

            self.faiss = faiss
        except ImportError:
            raise RuntimeError(f"faiss is not installed. Cannot use {self.__class__.__name__}")

    def _load(self, path: PathLike):
        self.import_faiss()
        self.index = self.faiss.read_index(str(path))

    def _save(self, path: PathLike):
        self.faiss.write_index(self.index, str(path))

    def _search_func(
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances, neighbours = self.index.search(query.numpy(), top_n)
        return np.squeeze(distances), np.squeeze(neighbours)

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
    a simple index of torch tensors.
    """

    def __init__(self, name: str):

        super().__init__(name)
        self.index: torch.Tensor

    def _load(self, path: PathLike):
        self.index = torch.load(path, map_location="cpu")

    def _save(self, path: PathLike):
        torch.save(self.index, path)

    def _add(self, embeddings: torch.Tensor):
        return torch.cat([self.index, embeddings])

    def _create_index(self, embeddings: torch.Tensor) -> Any:
        return embeddings

    def __len__(self):
        return len(self.index)


class CDistTensorEmbeddingIndex(TensorEmbeddingIndex):
    """
    Calculate embedding based on cosine distance
    """

    def _search_func(
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
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
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        score_matrix = torch.matmul(query, self.index.T)

        score_matrix = torch.squeeze(score_matrix)
        neighbours = torch.argsort(score_matrix, descending=True)[:top_n]
        distances = score_matrix[neighbours]
        distances = 100 - (1 / distances)
        return distances.cpu().numpy(), neighbours.cpu().numpy()
