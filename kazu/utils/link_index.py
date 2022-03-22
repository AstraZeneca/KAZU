import abc
import copy
import logging
import os
import pickle
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Dict, List, Iterable

import numpy as np
import torch
from rapidfuzz import process, fuzz

from kazu.data.data import LINK_SCORE, Mapping
from kazu.modelling.ontology_preprocessing.base import SYN, IDX, MAPPING_TYPE, DEFAULT_LABEL
from kazu.utils.utils import PathLike, as_path

logger = logging.getLogger(__name__)


class MetadataDatabase:
    """
    Singleton of a spacy pipeline, so we can reuse it across steps without needing to load the model
    multiple times
    """

    instance: "__MetadataDatabase"

    class __MetadataDatabase:
        database: Dict[str, Dict[str, Any]] = defaultdict(dict)

        def add(self, name: str, metadata: Dict[str, Any]):
            self.database[name].update(metadata)

        def get(self, name: str, idx: str) -> Any:
            return self.database[name].get(idx)

    def __init__(self):
        if not MetadataDatabase.instance:
            MetadataDatabase.instance = MetadataDatabase.__MetadataDatabase()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get(self, name: str, idx: str) -> Any:
        return copy.deepcopy(self.instance.database[name].get(idx))

    def get_all(self, name: str):
        return self.instance.database[name]

    def add(self, name: str, metadata: Dict[str, Any]):
        self.instance.add(name, metadata)


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
        self.metadata: MetadataDatabase = MetadataDatabase()
        self.index: Any
        self.namespace = self.__class__.__name__

    def _search(self, query: Any, **kwargs) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """
        subclasses should implement this method, which describes the logic to actually perform the search
        calls to search should return an iterable Tuple[str, Dict[str, Any]].
        the tuple[0] should be the kb id and tuple[1] should be a dict of any additional metadata associated with the
        hit
        :param query: the query to use
        :param kwargs: any other arguments that are required
        :return:
        """
        raise NotImplementedError()

    def search(self, query: Any, **kwargs) -> Iterable[Mapping]:
        """
        search the index
        :param query: the query to use
        :param kwargs: any other arguments to pass to self._search
        :return: a iterable of :class:`kazu.data.data.Mapping` of hits
        """
        for ontology_id, metadata in self._search(query, **kwargs):
            default_label = metadata.pop(DEFAULT_LABEL, "na")
            mapping_type = [str(x) for x in metadata.pop(MAPPING_TYPE, ["na"])]
            yield Mapping(
                default_label=str(default_label),
                source=self.name,
                idx=str(ontology_id),
                mapping_type=mapping_type,
                metadata=metadata,
            )

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
        with open(self.get_metadata_path(directory), "wb") as f:
            pickle.dump(self.metadata.get_all(self.name), f)

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
        with open(cls.get_metadata_path(path), "rb") as f:
            index.metadata.add(index.name, pickle.load(f))
        index._load(cls.get_index_data_path(path))
        return index

    @staticmethod
    def get_metadata_path(path: Path) -> Path:
        return path.joinpath("ontology_metadata.pkl")

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

    def _add(self, data: Any):
        """
        concrete implementations should implement this to add data to the index - e.g. synonym or embedding info. This
        method is called by self.add
        :param data:
        :return:
        """
        raise NotImplementedError()

    def add(self, data: Any, metadata: Dict[str, Any]):
        """
        add data to the index
        :param data:
        :return:
        """
        self.metadata.add(self.name, metadata)
        self._add(data)

    def __len__(self) -> int:
        """
        should return the size of the index
        :return:
        """
        return len(self.metadata.get_all(self.name))


@dataclass
class SynonymData:
    """
    data class required by DictionaryIndex add method. See docs on :py:class:`kazu.utils.link_index.DictionaryIndex`
    for usage
    """

    idx: str
    mapping_type: List[str]


class DictionaryIndex(Index):
    """
    a simple dictionary index for linking. Uses a Dict[str, List[SynonymData]] for matching synonyms,
    with optional fuzzy matching. Note, since a given synonym can match to more than one metadata entry (even
    within the same knowledgebase), we have a pathological situation in which 'true' synonyms can not be said to
    exist. In such situations, we return multiple kb references for overloaded synonyms - i.e. the disambiguation is
    delegated elsewhere
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)
        self.synonym_dict: Dict[str, List[SynonymData]]

    def _search(
        self, query: str, score_cutoff: float = 99.0, top_n: int = 20, fuzzy: bool = True, **kwargs
    ) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """
        search the index
        :param query: a string of text
        :param score_cutoff: minimum rapidfuzz match score. ignored if fuzzy-False
        :param top_n: return up to this many hits
        :param fuzzy: use rapidfuzz fuzzy matching
        :param kwargs:
        :return: Iterable of Tuple [kb.id, metadata_dict]
        """
        query = query.lower()
        if not fuzzy:
            yield from self.gather_match_result(query, 100.0)
        else:
            hits = process.extract(
                query,
                self.synonym_dict.keys(),
                scorer=fuzz.WRatio,
                limit=top_n,
                score_cutoff=score_cutoff,
            )
            for match_string, score, _ in hits:
                yield from self.gather_match_result(match_string, score)

    def gather_match_result(
        self, match_string: str, score: float
    ) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """
        for a given match string and score, format the return dict of metadata.
        :param match_string:
        :param score:
        :return:
        """
        synonym_data_lst = self.synonym_dict.get(match_string, [])
        for synonym_data in synonym_data_lst:
            metadata_hits = self.metadata.get(self.name, synonym_data.idx)
            metadata_hits[LINK_SCORE] = score
            metadata_hits[SYN] = match_string
            metadata_hits[MAPPING_TYPE] = synonym_data.mapping_type
            if len(synonym_data_lst) > 1:
                metadata_hits["ambiguous_synonyms"] = synonym_data_lst
            yield synonym_data.idx, metadata_hits

    def _load(self, path: PathLike) -> Any:
        with open(path, "rb") as f:
            self.synonym_dict = pickle.load(f)

    def _save(self, path: PathLike):
        with open(path, "wb") as f:
            pickle.dump(self.synonym_dict, f)

    def _add(self, data: Dict[str, List[SynonymData]]):
        """
        add data to the index. Two dicts are required - synonyms and metadata. Metadata should have a primary key
        (IDX) and synonyms should use IDX as a foreign key
        :param synonym_dict: synonyms dict of {synonym:List[SynonymData]}
        :param metadata_dict: metadata dict
        :return:
        """
        if not hasattr(self, "synonym_dict"):
            self.synonym_dict = defaultdict(list)

        for k, v in data.items():
            # syn keys must always be lower case
            self.synonym_dict[k.lower()].extend(v)


class EmbeddingIndex(Index):
    """
    a wrapper around an embedding index strategy.
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)

    def _add(self, embeddings: torch.Tensor):
        """
        add embeddings to the index

        :param embeddings: a 2d tensor of embeddings
        :param metadata_df: an ordered dict of metadata, with the order of each key corresponding to the embedding of
        the first dimension of embeddings
        :return:
        """
        if not hasattr(self, "index"):
            self.index = self._create_index(embeddings)
        else:
            self._add(embeddings)
        self.keys_lst = list(self.metadata.get_all(self.name).keys())

    def _create_index(self, embeddings: torch.Tensor) -> Any:
        """
        concrete implementations should implement this to create an index. This should also add the embeddings
        after the index is created

        :param embeddings:
        :return:
        """
        raise NotImplementedError()

    def _search_func(
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        should be implemented
        :param query: a string of text
        :param score_cutoff:
        :param top_n: return up to this many hits
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def _search(
        self, query: torch.Tensor, score_cutoff: float = 99.0, top_n: int = 20, **kwargs
    ) -> Iterable[Tuple[str, Any]]:
        distances, neighbours = self._search_func(
            query=query, top_n=top_n, score_cutoff=score_cutoff, **kwargs
        )
        for score, n in zip(distances, neighbours):
            key = self.keys_lst[n]
            hit_metadata = self.metadata.get(self.name, key)
            # the mapping type is always inferred for embedding based indices
            hit_metadata[MAPPING_TYPE] = ["inferred"]
            hit_metadata[LINK_SCORE] = score
            yield key, hit_metadata


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
        self.keys_lst = list(self.metadata.keys())

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
        self.keys_lst = list(self.metadata.keys())

    def _save(self, path: PathLike):
        torch.save(self.index, path)

    def _add(self, embeddings: torch.Tensor):
        self.index = torch.cat([self.index, embeddings])
        return self.index

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
