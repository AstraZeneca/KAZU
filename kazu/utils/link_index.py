import itertools
import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Any, Dict, List, Iterable, Iterator, Optional, cast

import numpy as np
import torch
from kazu.data.data import SimpleValue, SynonymTermWithMetrics
from kazu.modelling.database.in_memory_db import MetadataDatabase, SynonymDatabase
from kazu.modelling.language.string_similarity_scorers import BooleanStringSimilarityScorer
from kazu.modelling.linking.sapbert.train import PLSapbertModel
from kazu.modelling.ontology_preprocessing.base import (
    SYN,
    IDX,
    MAPPING_TYPE,
    DEFAULT_LABEL,
    OntologyParser,
)
from kazu.utils.string_normalizer import StringNormalizer
from kazu.utils.utils import create_char_ngrams, get_cache_dir
from pytorch_lightning import Trainer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


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
            pickle.dump(list(self.synonym_db.get_all(self.parser.name).values()), f)
        self._save(self.get_index_data_path(directory))
        return directory

    def load(self, cache_path: Path):
        """
        load from disk
        :param cache_path: the path to the cached files. Normally created via .save
        :return:
        """
        with open(self.get_metadata_path(cache_path), "rb") as f:
            self.metadata_db.add_parser(self.parser.name, pickle.load(f))
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
        self.metadata_db.add_parser(self.parser.name, metadata)
        self._add(data)

    def __len__(self) -> int:
        """
        should return the size of the index
        :return:
        """
        return len(self.metadata_db.get_all(self.parser.name))

    def load_or_build_cache(self, force_rebuild_cache: bool = False):
        """
        build the index, or if a cached version is available, load it instead
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


class DictionaryIndex(Index):
    """
    The dictionary index looks for SynonymTerms via a char ngram search between the normalised version of the
    query string and the term_norm of all SynonymTerms associated with the provided OntologyParser.
    """

    def __init__(
        self,
        parser: OntologyParser,
        boolean_scorers: Optional[List[BooleanStringSimilarityScorer]] = None,
    ):
        """

        :param parser:
        :param boolean_scorers: precision can be increased by applying boolean checks on the returned result, for
            instance, checking that all integers are represented, that any noun modifiers are present etc
        """
        super().__init__(parser=parser)
        self.boolean_scorers = boolean_scorers
        self.synonym_db: SynonymDatabase = SynonymDatabase()

    def apply_boolean_scorers(self, reference_term: str, query_term: str) -> bool:

        if self.boolean_scorers is not None:
            return all(
                scorer(reference_term=reference_term, query_term=query_term)
                for scorer in self.boolean_scorers
            )
        else:
            return True

    def _search_index(
        self, match: str, match_norm: str, top_n: int = 15
    ) -> List[SynonymTermWithMetrics]:

        exact_match_term = self.synonym_db.get_all(self.parser.name).get(match_norm)
        if exact_match_term is not None:
            term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                exact_match_term, search_score=100.0, bool_score=True, exact_match=True
            )
            return [term_with_metrics]

        else:
            # TODO: this seems to not be using sparse mat mul, and can be a lot faster. See TfIdfDocumentScorer
            # TODO: for example code on how to reimplement with sparse
            query = self.vectorizer.transform([match_norm]).todense()
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
            result = []
            synonym_list = list(self.synonym_db.get_all(self.parser.name).values())
            for neighbour, score in zip(neighbours, distances):
                # get by index
                term = synonym_list[neighbour]
                if self.apply_boolean_scorers(reference_term=match_norm, query_term=term.term_norm):
                    term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                        term, search_score=score, bool_score=True, exact_match=False
                    )
                    result.append(term_with_metrics)
                else:
                    logger.debug("filtered term %s as failed boolean checks", term)
                # we don't sort the terms as we expect them to enter an unsorted set
            return result

    def search(self, query: str, top_n: int = 15) -> Iterable[SynonymTermWithMetrics]:
        """
        search the index
        :param top_n: return a maximum of top_n SynonymTermWithMetrics
        :param query: a string of text
        :return:
        """

        match_norm = StringNormalizer.normalize(query, entity_class=self.parser.entity_class)
        terms = self._search_index(query, match_norm, top_n=top_n)
        yield from terms

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

    def _add(self, data: Any):
        raise NotImplementedError(f"It's not possible to add data to a {self.__class__.__name__}")

    def _build_ontology_cache(self, cache_dir: Path):
        logger.info(f"creating index for {self.parser.in_path}")

        self.parser.populate_databases()
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
        self.tf_idf_matrix = self.vectorizer.fit_transform(
            self.synonym_db.get_all(self.parser.name).keys()
        )
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
        :param top_n: return up to this many results
        :return:
        """
        raise NotImplementedError()

    def search(
        self,
        query: torch.Tensor,
        top_n: int = 1,
    ) -> Iterable[SynonymTermWithMetrics]:
        """

        :param query: a query tensor
        :param top_n: return top_n instances of SynonymTermWithMetrics
        :return:
        """

        distances, neighbours = self._search_func(query=query, top_n=top_n)
        for score, neighbour in zip(distances, neighbours):
            idx, metadata = self.metadata_db.get_by_index(self.parser.name, neighbour)
            # the norm form of the default label should always be in the syn database
            default_label = metadata[DEFAULT_LABEL]
            assert isinstance(default_label, str)
            string_norm = StringNormalizer.normalize(
                default_label, entity_class=self.parser.entity_class
            )
            try:

                term = self.synonym_db.get(self.parser.name, string_norm)
                term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                    term, embed_score=score
                )
                yield term_with_metrics
            except KeyError:
                logger.warning(
                    f"{string_norm} is not in the synonym database! is the parser for {self.parser.name} correctly configured?"
                )

    def _build_ontology_cache(self, cache_dir: Path):
        if not hasattr(self, "embedding_model"):
            raise RuntimeError("you must call set_embedding_model before trying to build the cache")

        logger.info(f"creating index for {self.parser.in_path}")

        # populate the databases
        self.parser.populate_databases()
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
