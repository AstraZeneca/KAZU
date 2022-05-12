import abc
import logging
import os
import pickle
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple, Any, Dict, List, Iterable, Optional, FrozenSet, Set

import numpy as np
import torch
from kazu.data.data import SearchRanks, SynonymData, Hit
from kazu.modelling.ontology_preprocessing.base import (
    SYN,
    IDX,
    MAPPING_TYPE,
    DEFAULT_LABEL,
    MetadataDatabase,
    StringNormalizer,
    SynonymDatabase,
)
from kazu.utils.utils import PathLike, as_path
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from strsimpy import NGram

logger = logging.getLogger(__name__)

SAPBERT_SCORE = "sapbert_score"
MATCHED_NUMBER_SCORE = "matched_number_score"
NGRAM_SCORE = "ngram_score"
FUZZ_SCORE = "fuzz_score"
SEARCH_SCORE = "search_score"
EXACT_MATCH = "exact_match"
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


class HitPostProcessor:
    def __init__(self):
        self.ngram = NGram(2)
        self.numeric_class_phrase_disambiguation = ["TYPE"]
        self.numeric_class_phrase_disambiguation_re = [
            re.compile(x + " [0-9]+") for x in self.numeric_class_phrase_disambiguation
        ]
        self.modifier_phrase_disambiguation = ["LIKE"]

    def phrase_disambiguation_filter(self, hits, text):
        new_hits = []
        for numeric_phrase_re in self.numeric_class_phrase_disambiguation_re:
            match = re.search(numeric_phrase_re, text)
            if match:
                found_string = match.group()
                for hit in hits:
                    if found_string in hit.string_norm:
                        new_hits.append(hit)
        if not new_hits:
            for modifier_phrase in self.modifier_phrase_disambiguation:
                in_text = modifier_phrase in text
                if in_text:
                    for hit in filter(lambda x: modifier_phrase in x.string_norm, hits):
                        new_hits.append(hit)
                else:
                    for hit in filter(lambda x: modifier_phrase not in x.string_norm, hits):
                        new_hits.append(hit)
        if new_hits:
            return new_hits
        else:
            return hits

    def ngram_scorer(self, hits: List[Hit], text):
        # low conf
        for hit in hits:
            hit.metrics[NGRAM_SCORE] = 2 / (self.ngram.distance(text, hit.string_norm) + 1.0)
        return hits

    def run_fuzz_algo(self, hits: List[Hit], text):
        # low conf
        choices = [x.string_norm for x in hits]
        if len(text) > 10 and len(text.split(" ")) > 4:
            scores = process.extract(text, choices, scorer=fuzz.token_sort_ratio)
        else:
            scores = process.extract(text, choices, scorer=fuzz.WRatio)
        for score in scores:
            hit = hits[score[2]]
            hit.metrics[FUZZ_SCORE] = score[1]
        return hits

    def run_number_algo(self, hits, text):
        number_resolver = NumberResolver(text)
        for hit in hits:
            numbers_matched = number_resolver(hit.string_norm)
            hit.metrics[MATCHED_NUMBER_SCORE] = numbers_matched
        return hits

    def __call__(self, hits: List[Hit], string_norm: str) -> List[Hit]:

        hits = self.phrase_disambiguation_filter(hits, string_norm)
        hits = self.run_number_algo(hits, string_norm)
        hits = self.run_fuzz_algo(hits, string_norm)
        hits = self.ngram_scorer(hits, string_norm)
        return hits


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


def create_char_ngrams(string, n=2):
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]


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
        self.metadata_db: MetadataDatabase = MetadataDatabase()
        self.index: Any
        self.namespace = self.__class__.__name__

    def save(self, path: PathLike, overwrite: bool = False) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets

        :param path:
        :return: a Path to where the index was saved
        """
        directory = as_path(path)
        if directory.exists() and overwrite:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=False)
        with open(self.get_index_path(directory), "wb") as f:
            pickle.dump(self, f)
        with open(self.get_metadata_path(directory), "wb") as f:
            pickle.dump(self.metadata_db.get_all(self.name), f)

        self._save(self.get_index_data_path(directory))
        return directory

    @classmethod
    def load(cls, path: PathLike):
        """
        load from disk
        :param path: the parent path of the index
        :return:
        """

        root_path = as_path(path)
        with open(cls.get_index_path(root_path), "rb") as f:
            index = pickle.load(f)
        index.metadata_db = MetadataDatabase()
        with open(cls.get_metadata_path(root_path), "rb") as f:
            index.metadata_db.add(index.name, pickle.load(f))
        index._load(cls.get_index_data_path(root_path))
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
        self.metadata_db = MetadataDatabase()

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
        self.metadata_db.add(self.name, metadata)
        self._add(data)

    def __len__(self) -> int:
        """
        should return the size of the index
        :return:
        """
        return len(self.metadata_db.get_all(self.name))


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
        synonym_dict: Dict[str, Set[SynonymData]],
        hit_post_processor: Optional[HitPostProcessor] = None,
        requires_normalisation: bool = True,
        name: str = "unnamed_index",
    ):
        super().__init__(name)
        self.hit_post_processor = hit_post_processor if hit_post_processor else HitPostProcessor()
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
        if requires_normalisation:
            self.normalised_syn_dict = self.gen_normalised(synonym_dict)
        else:
            self.normalised_syn_dict = synonym_dict
        self.key_lst = list(self.normalised_syn_dict.keys())
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.key_lst)
        self.tf_idf_matrix_torch = to_torch(self.tf_idf_matrix)

    def gen_normalised(
        self, synonym_dict: Dict[str, Set[SynonymData]]
    ) -> Dict[str, Set[SynonymData]]:
        norm_syn_dict: Dict[str, Set[SynonymData]] = defaultdict(set)
        for syn, syn_set in synonym_dict.items():

            new_syn = StringNormalizer.normalize(syn)
            for syn_data in syn_set:
                norm_syn_dict[new_syn].add(syn_data)
        return norm_syn_dict

    def _search_index(self, string_norm: str, top_n: int = 15) -> List[Hit]:
        if string_norm in self.normalised_syn_dict:
            return [
                Hit(
                    string_norm=string_norm,
                    parser_name=self.name,
                    metrics={EXACT_MATCH: True, SEARCH_SCORE: 100.0},
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
                        parser_name=self.name,
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
        if not (len(hits) == 1 and hits[0].confidence == SearchRanks.EXACT_MATCH):
            hits = self.hit_post_processor(hits, string_norm)

        yield from hits

    def _load(self, path: PathLike) -> Any:
        if isinstance(path, str):
            path = Path(path)
        with open(path.joinpath("objects.pkl"), "rb") as f:
            (
                self.vectorizer,
                self.normalised_syn_dict,
                self.tf_idf_matrix,
                self.hit_post_processor,
            ) = pickle.load(f)
        self.key_lst = list(self.normalised_syn_dict.keys())
        self.tf_idf_matrix_torch = to_torch(self.tf_idf_matrix)

    def _save(self, path: PathLike):
        if isinstance(path, str):
            path = Path(path)
        path.mkdir()
        pickleable = (
            self.vectorizer,
            self.normalised_syn_dict,
            self.tf_idf_matrix,
            self.hit_post_processor,
        )
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


class EmbeddingIndex(Index):
    """
    a wrapper around an embedding index strategy.
    """

    def __init__(self, name: str = "unnamed_index"):
        super().__init__(name)
        self.metadata_db = MetadataDatabase()
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
        original_string: str,
        score_cutoffs: Tuple[float, float] = (99.9945, 99.995),
        top_n: int = 1,
    ) -> Iterable[Hit]:
        distances, neighbours = self._search_func(query=query, top_n=top_n)
        for score, n in zip(distances, neighbours):
            idx, metadata = self.metadata_db.get_by_index(self.name, n)
            # the norm form of the default label should always be in the syn database
            string_norm = StringNormalizer.normalize(metadata[DEFAULT_LABEL])
            try:
                syn_data = self.synonym_db.get(self.name, string_norm)
                # confidence is always medium, dso can be later disambiguated
                hit = Hit(
                    string_norm=string_norm,
                    syn_data=frozenset(syn_data),
                    parser_name=self.name,
                    confidence=SearchRanks.NEAR_MATCH,
                    metrics={SAPBERT_SCORE: score},
                )
                yield hit
            except KeyError:
                logger.warning(
                    f"{string_norm} is not in the synonym database! is the parser for {self.name} correctly configured?"
                )

    def __getstate__(self):
        return self.name

    def __setstate__(self, state):
        self.name = state
        self.metadata_db = MetadataDatabase()
        self.synonym_db = SynonymDatabase()


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
        self.keys_lst = list(self.metadata_db.get_all(self.name).keys())

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

    def _add_embeddings(self, embeddings: torch.Tensor):
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
