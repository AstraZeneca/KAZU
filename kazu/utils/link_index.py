import itertools
import logging
import os
import pickle
import re
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Tuple, Any, Dict, List, Iterable, Iterator, Optional, Set, cast

import numpy as np
import torch
from pytorch_lightning import Trainer
from sklearn.feature_extraction.text import TfidfVectorizer

from kazu.data.data import EquivalentIdSet, Hit
from kazu.modelling.linking.sapbert.train import PLSapbertModel
from kazu.modelling.ontology_preprocessing.base import (
    SYN,
    IDX,
    MAPPING_TYPE,
    DEFAULT_LABEL,
    MetadataDatabase,
    StringNormalizer,
    SynonymDatabase,
    OntologyParser,
    SimpleValue,
)
from kazu.utils.utils import create_char_ngrams, get_cache_dir

logger = logging.getLogger(__name__)

SAPBERT_SCORE = "sapbert_score"
SEARCH_SCORE = "search_score"
DICTIONARY_HITS = "dictionary_hits"
EXACT_MATCH = "exact_match"


class NumberResolver:
    number_finder = re.compile("[0-9]+")

    def __init__(self, query_string_norm):
        self.ent_match_number_count = Counter(re.findall(self.number_finder, query_string_norm))

    def __call__(self, synonym_string_norm: str):
        synonym_string_norm_match_number_count = Counter(
            re.findall(self.number_finder, synonym_string_norm)
        )
        return synonym_string_norm_match_number_count == self.ent_match_number_count


class Index(ABC):
    """
    base class for all indices.
    """

    column_type_dict = {SYN: str, IDX: str, MAPPING_TYPE: list, DEFAULT_LABEL: str}

    def __init__(
        self,
        parser: OntologyParser,
    ):
        """

        :param parser: The ontology parser to use with this index
        """
        self.parser = parser
        self.metadata_db: MetadataDatabase = MetadataDatabase()
        self.synonym_db: SynonymDatabase = SynonymDatabase()
        self.namespace = self.__class__.__name__

    def save(self, directory: Path, overwrite: bool = False) -> Path:
        """
        save to disk. Makes a directory at the path location with all the index assets

        :param directory: a dir to save the index.
        :param overwrite: should the directory be deleted before attempting to save? (CAREFUL!)
        :return: a Path to where the index was saved
        """
        if directory.exists() and overwrite:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=False)
        with open(self.get_metadata_path(directory), "wb") as f:
            pickle.dump(self.metadata_db.get_all(self.parser.name), f)
        with open(self.get_synonym_data_path(directory), "wb") as f:
            pickle.dump(self.synonym_db.get_all(self.parser.name), f)
        self._save(self.get_index_data_path(directory))
        return directory

    def load(self, cache_path: Path):
        """
        load from disk
        :param cache_path: the path to the cached files. Normally created via .save
        :return:
        """
        with open(self.get_metadata_path(cache_path), "rb") as f:
            self.metadata_db.add(self.parser.name, pickle.load(f))
        with open(self.get_synonym_data_path(cache_path), "rb") as f:
            self.synonym_db.add(self.parser.name, pickle.load(f))
        self._load(self.get_index_data_path(cache_path))

    @staticmethod
    def get_metadata_path(path: Path) -> Path:
        return path.joinpath("ontology_metadata.pkl")

    @staticmethod
    def get_synonym_data_path(path: Path) -> Path:
        return path.joinpath("ontology_synonym_data.pkl")

    @staticmethod
    def get_index_data_path(path: Path) -> Path:
        return path.joinpath("index.pkl")

    def _save(self, path: Path):
        """
        concrete implementations should implement this to save any data specific to the implementation. This method is
        called by self.save
        :param path:
        :return:
        """
        raise NotImplementedError()

    def _load(self, path: Path) -> None:
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

    def add(self, data: Any, metadata: Dict[str, Dict[str, SimpleValue]]):
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
        self.load(cache_dir)

    @abstractmethod
    def _build_ontology_cache(self, cache_dir: Path):
        """
        Implementations should implement this method to determine how an index gets built for a given parser
        :param cache_dir:
        :return:
        """
        pass

    def _populate_databases(self) -> Dict[str, Set[EquivalentIdSet]]:
        """
        populate the databases with the results of the parser
        :return: normalised dictionary of syn: set(ids)
        """
        # populate the databases
        self.parser.populate_metadata_database()
        normalised_syn_dict = self.parser.collect_aggregate_synonym_data(True)
        self.synonym_db.add(self.parser.name, normalised_syn_dict)
        return normalised_syn_dict


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
        self.synonym_db: SynonymDatabase = SynonymDatabase()

    def _search_index(self, string_norm: str, top_n: int = 15) -> List[Hit]:
        hits = []
        if string_norm in self.synonym_db.get_all(self.parser.name):
            for id_set in self.synonym_db.get(self.parser.name, string_norm):
                hits.append(
                    Hit(
                        parser_name=self.parser.name,
                        per_normalized_syn_metrics={
                            string_norm: {SEARCH_SCORE: 100.0, EXACT_MATCH: True}
                        },
                        id_set=id_set,
                    )
                )
            return hits

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
            synonym_list = list(self.synonym_db.get_all(self.parser.name))
            for neighbour, score in zip(neighbours, distances):
                # get by index
                found_norm = synonym_list[neighbour]
                for id_set in self.synonym_db.get(self.parser.name, found_norm):
                    hits.append(
                        Hit(
                            parser_name=self.parser.name,
                            per_normalized_syn_metrics={
                                found_norm: {SEARCH_SCORE: score, EXACT_MATCH: False}
                            },
                            id_set=id_set,
                        )
                    )
        # we don't sort the hits as we expect them to enter an unsorted set
        return hits

    def search(self, query: str, top_n: int = 15) -> Iterable[Hit]:
        """
        search the index
        :param query: a string of text
        :return: Iterable of Tuple [kb.id, metadata_dict]
        """

        string_norm = StringNormalizer.normalize(query)
        hits = self._search_index(string_norm, top_n=top_n)
        yield from hits

    def _load(self, path: Path) -> Any:
        with open(path, "rb") as f:
            (
                self.vectorizer,
                self.tf_idf_matrix,
            ) = pickle.load(f)

    def _save(self, path: Path):
        pickleable = (self.vectorizer, self.tf_idf_matrix)
        with open(path, "wb") as f:
            pickle.dump(pickleable, f)

    def _add(self, data: Dict[str, List[EquivalentIdSet]]):
        """
        deprecated
        add data to the index. Two dicts are required - synonyms and metadata. Metadata should have a primary key
        (IDX) and synonyms should use IDX as a foreign key
        :param data: synonyms dict of {synonym:List[EquivalentIdSet]}
        :return:
        """
        raise NotImplementedError()

    def _build_ontology_cache(self, cache_dir: Path):
        logger.info(f"creating index for {self.parser.in_path}")

        normalised_syn_dict = self._populate_databases()
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
        self.tf_idf_matrix = self.vectorizer.fit_transform(normalised_syn_dict.keys())
        index_path = self.save(cache_dir, overwrite=True)
        logger.info(f"saved {self.parser.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {self.parser.name} is {len(self)}")


class EmbeddingIndex(Index):
    """
    a wrapper around an embedding index strategy.
    """

    def __init__(self, parser: OntologyParser, ontology_partition_size: int = 1000):
        super().__init__(parser=parser)
        self.ontology_partition_size = ontology_partition_size
        self.embedding_model: PLSapbertModel
        self.trainer: Optional[Trainer]

    def set_embedding_model(
        self, embedding_model: PLSapbertModel, trainer: Optional[Trainer] = None
    ):
        """
        set the embedding model to be used to generate target ontology embeddings when the cache is generated
        (currently only PLSapbertModel supported). Also, optionally set a PL Trainer that should be used (a Trainer
        configured with default params will be used otherwise)
        :param embedding_model:
        :param trainer:
        :return:
        """
        self.embedding_model = embedding_model
        self.trainer = trainer

    def _add(self, embeddings: torch.Tensor):
        """
        add embeddings to the index

        :param embeddings: a 2d tensor of embeddings
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
        for score, neighbour in zip(distances, neighbours):
            idx, metadata = self.metadata_db.get_by_index(self.parser.name, neighbour)
            # the norm form of the default label should always be in the syn database
            default_label = metadata[DEFAULT_LABEL]
            assert isinstance(default_label, str)
            string_norm = StringNormalizer.normalize(default_label)
            try:
                for id_set in self.synonym_db.get(self.parser.name, string_norm):
                    # confidence is always medium, dso can be later disambiguated
                    hit = Hit(
                        id_set=id_set,
                        parser_name=self.parser.name,
                        per_normalized_syn_metrics={string_norm: {SAPBERT_SCORE: score}},
                    )
                    yield hit
            except KeyError:
                logger.warning(
                    f"{string_norm} is not in the synonym database! is the parser for {self.parser.name} correctly configured?"
                )

    def _build_ontology_cache(self, cache_dir: Path):
        if not hasattr(self, "embedding_model"):
            raise RuntimeError("you must call set_embedding_model before trying to build the cache")

        logger.info(f"creating index for {self.parser.in_path}")

        # populate the databases
        self._populate_databases()
        for (
            partition_number,
            metadata,
            ontology_embeddings,
        ) in self.predict_ontology_embeddings(self.parser.name):
            logger.info(f"processing partition {partition_number} ")
            self.add(data=ontology_embeddings, metadata=metadata)
            logger.info(f"index size is now {len(self)}")
        index_path = self.save(cache_dir, overwrite=True)
        logger.info(f"saved {self.parser.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {self.parser.name} is {len(self)}")

    def enumerate_database_chunks(
        self, name: str, chunk_size: int = 100000
    ) -> Iterable[Tuple[int, Dict[str, Dict[str, SimpleValue]]]]:
        """
        generator to split up a dataframe into partitions
        :param name: ontology name to query the metadata database with
        :param chunk_size: size of partittions to create
        :return:
        """

        data = self.metadata_db.get_all(name)
        for i in range(0, len(data), chunk_size):
            yield i, dict(itertools.islice(data.items(), i, i + chunk_size))

    def predict_ontology_embeddings(
        self, name: str
    ) -> Iterator[Tuple[int, Dict[str, Dict[str, SimpleValue]], torch.Tensor]]:
        """
        since embeddings are memory hungry, we use a generator to partition an input dataframe into manageable chucks,
        and add them to the index sequentially
        :param name: name of ontology
        :return: partition number, metadata and embeddings
        """

        for partition_number, metadata in self.enumerate_database_chunks(
            name, self.ontology_partition_size
        ):
            len_df = len(metadata)
            if len_df == 0:
                return
            logger.info(f"creating partitions for partition {partition_number}")
            logger.info(f"read {len_df} rows from ontology")
            # use cast to tell mypy this is already the right type
            default_labels = cast(List[str], [x[DEFAULT_LABEL] for x in metadata.values()])
            logger.info(f"predicting embeddings for default_labels. Examples: {default_labels[:3]}")
            results = self.embedding_model.get_embeddings_for_strings(
                texts=default_labels, trainer=self.trainer
            )
            yield partition_number, metadata, results


class TensorEmbeddingIndex(EmbeddingIndex):
    """
    a simple index of torch tensors.
    """

    def __init__(
        self,
        parser: OntologyParser,
    ):
        super().__init__(parser=parser)
        self.index: torch.Tensor

    def _load(self, path: Path) -> Any:
        with open(path, "rb") as f:
            self.index = pickle.load(f)

    def _save(self, path: Path):
        pickleable = self.index
        with open(path, "wb") as f:
            pickle.dump(pickleable, f)

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
