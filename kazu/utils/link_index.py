import abc
import logging
import os
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Tuple, Any, Dict, List, Iterable

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from kazu.data.data import SearchRanks, SynonymData, Hit
from kazu.modelling.ontology_preprocessing.base import (
    SYN,
    IDX,
    MAPPING_TYPE,
    DEFAULT_LABEL,
    MetadataDatabase,
    StringNormalizer,
    SynonymDatabase,
    OntologyParser,
)
from kazu.utils.utils import PathLike, create_char_ngrams, get_cache_dir

logger = logging.getLogger(__name__)

SAPBERT_SCORE = "sapbert_score"
SEARCH_SCORE = "search_score"
DICTIONARY_HITS = "dictionary_hits"


class NumberResolver:
    number_finder = re.compile("[0-9]+")

    def __init__(self, query_string_norm):
        self.ent_match_number_count = Counter(re.findall(self.number_finder, query_string_norm))

    def __call__(self, synonym_string_norm: str):
        synonym_string_norm_match_number_count = Counter(
            re.findall(self.number_finder, synonym_string_norm)
        )
        return synonym_string_norm_match_number_count == self.ent_match_number_count


def to_torch(matrix):
    """
    convert a sparse CSR matrix to a sparse torch matrix
    :param matrix:
    :param shape:
    :return:
    """

    Acoo = matrix.tocoo()
    result = torch.sparse_coo_tensor(
        torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
        torch.FloatTensor(Acoo.data),
        matrix.shape,
    ).to_sparse_csr()
    return result


class Index(abc.ABC):
    """
    base class for all indices.
    """

    column_type_dict = {SYN: str, IDX: str, MAPPING_TYPE: list, DEFAULT_LABEL: str}

    def __init__(
        self,
        parser: OntologyParser,
    ):
        """

        :param name: the name of the index. default is unnamed_index
        """
        self.parser = parser
        self.metadata_db: MetadataDatabase = MetadataDatabase()
        self.synonym_db: SynonymDatabase = SynonymDatabase()
        self.namespace = self.__class__.__name__

    def save(self, directory: Path, overwrite: bool = False) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets

        :param path:
        :return: a Path to where the index was saved
        """
        if directory.exists() and overwrite:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=False)
        with open(self.get_metadata_path(directory), "wb") as f:
            pickle.dump(self.metadata_db.get_all(self.parser.name), f)
        self._save(self.get_index_data_path(directory))
        return directory

    def load(self, cache_path: Path):
        """
        load from disk
        :param path: the parent path of the index
        :return:
        """

        self._load_cache(self.get_index_data_path(cache_path))
        with open(self.get_metadata_path(cache_path), "rb") as f:
            self.metadata_db.add(self.parser.name, pickle.load(f))

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

    def _load_cache(self, path: Path) -> None:
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
        self.metadata_db.add(self.parser.name, metadata)
        self._add(data)

    def __len__(self) -> int:
        """
        should return the size of the index
        :return:
        """
        return len(self.metadata_db.get_all(self.parser.name))

    def load_or_build_cache(self, force_rebuild_cache: bool = False):
        """
        for each parser in self.parsers, create an index. If a cached version is available, load it instead
        :return:
        """
        cache_dir = get_cache_dir(
            self.parser.in_path,
            prefix=f"{self.parser.name}_{self.__class__.__name__}",
            create_if_not_exist=False,
        )
        if force_rebuild_cache:
            logger.info("forcing a rebuild of the ontology cache")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir()
            self.build_ontology_cache(cache_dir)
        elif cache_dir.exists():
            logger.info(f"loading cached ontology file from {cache_dir}")
            self.load(cache_dir)
        else:
            logger.info("No ontology cache file found. Building a new one")
            self.build_ontology_cache(cache_dir)

    def build_ontology_cache(self, cache_dir: Path):
        cache_dir.mkdir()
        self._build_ontology_cache(cache_dir)

    @abc.abstractmethod
    def _build_ontology_cache(self, cache_dir: Path):
        """
        Implementations should implement this method to determine how an index gets built for a given parser
        :param cache_dir:
        :param parser:
        :return:
        """
        pass


class DictionaryIndex(Index):
    """
    a simple dictionary index for linking. Uses a Dict[str, List[SynonymData]] for matching synonyms,
    with optional fuzzy matching. Note, since a given synonym can match to more than one metadata entry (even
    within the same knowledgebase), we have a pathological situation in which 'true' synonyms can not be said to
    exist. In such situations, we return multiple kb references for overloaded synonyms - i.e. the disambiguation is
    delegated elsewhere
    """

    def __init__(
        self,
        parser: OntologyParser,
    ):
        super().__init__(parser=parser)

    def _search_index(self, string_norm: str, top_n: int = 15) -> List[Hit]:
        if string_norm in self.normalised_syn_dict:
            return [
                Hit(
                    string_norm=string_norm,
                    parser_name=self.parser.name,
                    metrics={SEARCH_SCORE: 100.0},
                    syn_data=frozenset(self.normalised_syn_dict[string_norm]),
                    confidence=SearchRanks.EXACT_MATCH,
                )
            ]
        else:

            query = self.vectorizer.transform([string_norm]).todense()
            # minus to negate, so arg sort works in correct order
            score_matrix = np.squeeze(-np.asarray(self.tf_idf_matrix.dot(query.T)))
            neighbours = score_matrix.argsort()[:top_n]
            # don't use torch for this - it's slow
            # query = torch.FloatTensor(query)
            # score_matrix = self.tf_idf_matrix_torch.matmul(query.T)
            # score_matrix = torch.squeeze(score_matrix.T)
            # neighbours = torch.argsort(score_matrix, descending=True)[:top_n]

            distances = score_matrix[neighbours]
            distances = 100 * -distances
            hits = []
            for neighbour, score in zip(neighbours, distances):
                found_norm = self.key_lst[neighbour]
                hits.append(
                    Hit(
                        string_norm=found_norm,
                        parser_name=self.parser.name,
                        syn_data=frozenset(self.normalised_syn_dict[found_norm]),
                        metrics={SEARCH_SCORE: score},
                        confidence=SearchRanks.NEAR_MATCH,
                    )
                )

        return sorted(hits, key=lambda x: x.metrics[SEARCH_SCORE], reverse=True)

    def search(self, query: str, top_n: int = 15) -> Iterable[Hit]:
        """
        search the index
        :param query: a string of text
        :return: Iterable of Tuple [kb.id, metadata_dict]
        """

        string_norm = StringNormalizer.normalize(query)
        hits = self._search_index(string_norm, top_n=top_n)
        yield from hits

    def _load_cache(self, path: Path) -> Any:
        with open(path.joinpath("objects.pkl"), "rb") as f:
            (
                self.vectorizer,
                self.normalised_syn_dict,
                self.tf_idf_matrix,
            ) = pickle.load(f)
        self.key_lst = list(self.normalised_syn_dict.keys())
        self.tf_idf_matrix_torch = to_torch(self.tf_idf_matrix)
        self.synonym_db.add(self.parser.name, self.normalised_syn_dict)

    def _save(self, path: PathLike):
        if isinstance(path, str):
            path = Path(path)
        path.mkdir()
        pickleable = (self.vectorizer, self.normalised_syn_dict, self.tf_idf_matrix)
        with open(path.joinpath("objects.pkl"), "wb") as f:
            pickle.dump(pickleable, f)

    def _add(self, data: Dict[str, List[SynonymData]]):
        """
        deprecated
        add data to the index. Two dicts are required - synonyms and metadata. Metadata should have a primary key
        (IDX) and synonyms should use IDX as a foreign key
        :param synonym_dict: synonyms dict of {synonym:List[SynonymData]}
        :param metadata_dict: metadata dict
        :return:
        """
        raise NotImplementedError()

    def _build_ontology_cache(self, cache_dir: Path):
        logger.info(f"creating index for {self.parser.in_path}")

        self.parser.populate_metadata_database()
        self.normalised_syn_dict = self.parser.collect_aggregate_synonym_data(True)
        self.synonym_db.add(self.parser.name, self.normalised_syn_dict)

        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
        self.key_lst = list(self.normalised_syn_dict.keys())
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.key_lst)
        self.tf_idf_matrix_torch = to_torch(self.tf_idf_matrix)
        index_path = self.save(cache_dir, overwrite=True)
        logger.info(f"saved {self.parser.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {self.parser.name} is {len(self)}")


class EmbeddingIndex(Index):
    """
    a wrapper around an embedding index strategy.
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)
        self.synonym_db = SynonymDatabase()

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
            self._add_embeddings(embeddings)

    def _add_embeddings(self, embeddings: torch.Tensor):
        """
        concrete implementations should implement this method to add embeddings to the index
        :param embeddings:
        :return:
        """
        raise NotImplementedError()

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
        :return:
        """
        raise NotImplementedError()

    def search(
        self,
        query: torch.Tensor,
        top_n: int = 1,
    ) -> Iterable[Hit]:
        distances, neighbours = self._search_func(query=query, top_n=top_n)
        for score, n in zip(distances, neighbours):
            idx, metadata = self.metadata_db.get_by_index(self.name, n)
            # the norm form of the default label should always be in the syn database
            string_norm = StringNormalizer.normalize(metadata[DEFAULT_LABEL])
            try:
                syn_data = self.synonym_db.get(self.parser.name, string_norm)
                # confidence is always medium, dso can be later disambiguated
                hit = Hit(
                    string_norm=string_norm,
                    syn_data=frozenset(syn_data),
                    parser_name=self.parser.name,
                    confidence=SearchRanks.NEAR_MATCH,
                    metrics={SAPBERT_SCORE: score},
                )
                yield hit
            except KeyError:
                logger.warning(
                    f"{string_norm} is not in the synonym database! is the parser for {self.parser.name} correctly configured?"
                )


class TensorEmbeddingIndex(EmbeddingIndex):
    """
    a simple index of torch tensors.
    """

    def __init__(self, name: str):

        super().__init__(name)
        self.index: torch.Tensor

    def _load(self, path: PathLike):
        self.index = torch.load(path, map_location="cpu")
        self.keys_lst = list(self.metadata_db.get_all(self.name).keys())

    def _save(self, path: PathLike):
        torch.save(self.index, path)

    def _add_embeddings(self, embeddings: torch.Tensor):
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
