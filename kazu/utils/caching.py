import copy
import itertools
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, Dict, Any, Iterable, Tuple, List, Set, Iterator
import torch
from cachetools import LFUCache
from pytorch_lightning import Trainer

from kazu.data.data import Entity, NAMESPACE
from kazu.data.data import Mapping
from kazu.modelling.linking.sapbert.train import (
    PLSapbertModel,
)
from kazu.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
    MetadataDatabase,
)
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.utils.link_index import (
    Index,
    MatMulTensorEmbeddingIndex,
    FaissEmbeddingIndex,
    CDistTensorEmbeddingIndex,
    DictionaryIndex,
    EmbeddingIndex,
)
from kazu.utils.utils import (
    get_cache_dir,
)
from kazu.utils.utils import get_match_entity_class_hash

logger = logging.getLogger(__name__)


class EntityLinkingLookupCache:
    """
    A simple wrapper around LFUCache to reduce calls to expensive processes (e.g. bert)
    """

    def __init__(self, lookup_cache_size: int = 5000):
        self.lookup_cache: LFUCache = LFUCache(lookup_cache_size)

    def update_lookup_cache(self, entity: Entity, mappings: List[Mapping]):
        hash_val = get_match_entity_class_hash(entity)
        cache_hit = self.lookup_cache.get(hash_val)
        if cache_hit is None:
            self.lookup_cache[hash_val] = mappings

    def check_lookup_cache(self, entities: List[Entity]) -> List[Entity]:
        """
        checks the cache for mappings. If relevant mappings are found for an entity, update its mappings
        accordingly. If not return as a list of cache misses (e.g. for further processing)

        :param entities:
        :return:
        """
        cache_misses = []
        for ent in entities:
            hash_val = get_match_entity_class_hash(ent)
            cache_hits = self.lookup_cache.get(hash_val, None)
            if cache_hits is None:
                cache_misses.append(ent)
            else:
                for mapping in cache_hits:
                    ent.add_mapping(copy.deepcopy(mapping))
        return cache_misses


class SynonymTableCache:
    def __init__(self, parsers: List[OntologyParser], rebuild_cache: bool = False):
        self.parsers = parsers
        self.rebuild_cache = rebuild_cache

    def get_synonym_table_paths(self) -> List[Path]:
        """
        for each parser in self.parsers, create a synonym table. If a cached version is available, load it instead
        :return:
        """
        synonym_table_paths: List[Path] = []
        for parser in self.parsers:
            cache_dir = get_cache_dir(
                parser.in_path, prefix="synonym_table", create_if_not_exist=False
            )
            cache_path = self._get_synonym_cache_path(cache_dir, parser)

            if self.rebuild_cache:
                logger.info("forcing a rebuild of the synonym table cache")
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                cache_dir.mkdir()
                self.build_synonym_table_cache(cache_path, parser)
            elif cache_dir.exists() and cache_path.exists():
                logger.info(f"using cached synonym table file from {cache_path}")
            else:
                logger.info("No synonym table cache file found. Building a new one")
                cache_dir.mkdir(exist_ok=True)
                self.build_synonym_table_cache(cache_path, parser)

            synonym_table_paths.append(cache_path)

        return synonym_table_paths

    @staticmethod
    def build_synonym_table_cache(cache_path: Path, parser: OntologyParser) -> None:
        logger.info(f"creating synonym table for {parser.in_path}")
        syn_df, _ = parser.parse_and_cache()
        syn_df.to_parquet(cache_path, index=None)
        logger.info(f"saved {parser.name} synonym table to {cache_path.absolute()}")

    @staticmethod
    def _get_synonym_cache_path(cache_dir: Path, parser: OntologyParser) -> Path:
        return cache_dir.joinpath(f"{parser.name}_synonyms.parquet")


class IndexCacheManager(ABC):
    """
    An IndexCacheManager is responsible for creating, saving and loading a set of :class:`kazu.utils.caching.Index`.
    It's useful to use this class, instead of an instance of :class:`kazu.utils.caching.Index` directly, as this class
    will automatically cache any expensive index creation aspects (such as embedding generation for sapbert)
    """

    def __init__(self, index_type: str, parsers: List[OntologyParser], rebuild_cache: bool = False):
        self.parsers = parsers
        self.index_type = self.select_index_type(index_type)
        self.rebuild_cache = rebuild_cache

    def get_or_create_ontology_indices(self) -> List[Index]:
        """
        for each parser in self.parsers, create an index. If a cached version is available, load it instead
        :return:
        """

        indices = []
        for parser in self.parsers:
            cache_dir = get_cache_dir(
                parser.in_path, prefix=self.index_type.__name__, create_if_not_exist=False
            )
            if self.rebuild_cache:
                logger.info("forcing a rebuild of the ontology cache")
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                cache_dir.mkdir()
                indices.append(self.build_ontology_cache(cache_dir, parser))
            elif cache_dir.exists():
                logger.info(f"loading cached ontology file from {cache_dir}")
                indices.append(self.load_ontology_from_cache(cache_dir, parser))
            else:
                logger.info("No ontology cache file found. Building a new one")
                cache_dir.mkdir()
                indices.append(self.build_ontology_cache(cache_dir, parser))
        return indices

    @abstractmethod
    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        """
        Implementations should implement this method to determine how an index gets built for a given parser
        :param cache_dir:
        :param parser:
        :return:
        """
        pass

    def load_ontology_from_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        """
        load an index from the cache
        :param cache_dir:
        :param parser:
        :return:
        """
        return Index.load(str(cache_dir), parser.name)

    def select_index_type(self, index_class_name: str) -> Type:
        """
        select a index type based on its string name. Note, this should return a type compatible with the concrete
        implementation (e.g. EmbeddingIndexCacheManager -> EmbeddingIndex )
        :param index_class_name:
        :return:
        """
        raise NotImplementedError()


class DictionaryIndexCacheManager(IndexCacheManager):
    """
    implementation for use with :class:`kazu.steps.linking.dictionary.DictionaryEntityLinkingStep`
    """

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        logger.info(f"creating index for {parser.in_path}")
        index = self.index_type(name=parser.name)
        assert isinstance(index, DictionaryIndex)
        parser.populate_synonym_database()
        parser.populate_metadata_database()
        # the indices read directly from the database singletons, so no need to add the data directly
        index_path = index.save(str(cache_dir))
        logger.info(f"saved {index.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {index.name} is {len(index)}")
        return index

    def select_index_type(self, index_class_name: str) -> Type[DictionaryIndex]:
        """
        select a index type based on its string name. Note, this should return a type compatible with the concrete
        implementation (e.g. EmbeddingIndexCacheManager -> EmbeddingIndex )
        :param index_class_name:
        :return:
        """
        if index_class_name == DictionaryIndex.__name__:
            return DictionaryIndex
        else:
            raise NotImplementedError(f"{index_class_name} not implemented")


class EmbeddingIndexCacheManager(IndexCacheManager):
    """
    implementation for use with embedding style linking steps, such as
    :class:`kazu.steps.linking.sapbert.SapBertForEntityLinkingStep`
    """

    def __init__(
        self,
        index_type: str,
        parsers: List[OntologyParser],
        model: PLSapbertModel,
        batch_size: int,
        trainer: Trainer,
        dl_workers: int,
        ontology_partition_size: int,
        rebuild_cache: bool = False,
    ):
        """

        :param index_type: type of index to create
        :param parsers: list of parsers to create indices for
        :param model: model for generating embeddings with
        :param batch_size: batch size to use
        :param trainer: instance of lightning trainer to use
        :param dl_workers: number of data loaders to use
        :param ontology_partition_size: size of each partition when generating embeddings. reduce if running into
            memory issues
        :param rebuild_cache: force a rebuild of the cache
        """

        super().__init__(index_type, parsers, rebuild_cache)
        self.ontology_partition_size = ontology_partition_size
        self.dl_workers = dl_workers
        self.batch_size = batch_size
        self.model = model
        self.trainer = trainer

    def select_index_type(self, index_class_name: str) -> Type[EmbeddingIndex]:
        """
        select a index type based on its string name. Note, this should return a type compatible with the concrete
        implementation (e.g. EmbeddingIndexCacheManager -> EmbeddingIndex )
        :param index_class_name:
        :return:
        """
        if index_class_name == MatMulTensorEmbeddingIndex.__name__:
            return MatMulTensorEmbeddingIndex
        elif index_class_name == FaissEmbeddingIndex.__name__:
            return FaissEmbeddingIndex
        elif index_class_name == CDistTensorEmbeddingIndex.__name__:
            return CDistTensorEmbeddingIndex
        else:
            raise NotImplementedError(f"{index_class_name} not implemented")

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:

        logger.info(f"creating index for {parser.in_path}")

        index: EmbeddingIndex = self.index_type(parser.name)
        assert issubclass(type(index), EmbeddingIndex)
        parser.populate_metadata_database()
        for (
            partition_number,
            metadata,
            ontology_embeddings,
        ) in self.predict_ontology_embeddings(parser.name):
            logger.info(f"processing partition {partition_number} ")
            index.add(data=ontology_embeddings, metadata=metadata)
            logger.info(f"index size is now {len(index)}")
        index_path = index.save(cache_dir)
        logger.info(f"saved {parser.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {parser.name} is {len(index)}")
        return index

    @staticmethod
    def enumerate_database_chunks(
        name: str, chunk_size: int = 100000
    ) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """
        generator to split up a dataframe into partitions
        :param name: ontology name to query the metadata database with
        :param chunk_size: size of partittions to create
        :return:
        """

        data: Dict[str, Any] = MetadataDatabase().get_all(name)
        for i in range(0, len(data), chunk_size):
            yield i, dict(itertools.islice(data.items(), i, i + chunk_size))

    def predict_ontology_embeddings(
        self, name: str
    ) -> Iterator[Tuple[int, Dict[str, Any], torch.Tensor]]:
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
            default_labels = [x[DEFAULT_LABEL] for x in metadata.values()]
            logger.info(f"predicting embeddings for default_labels. Examples: {default_labels[:3]}")
            results = self.model.get_embeddings_for_strings(
                texts=default_labels, trainer=self.trainer, batch_size=self.batch_size
            )
            yield partition_number, metadata, results


class CachedIndexGroup:
    """
    Convenience class for building and loading Indices, and querying against them
    """

    def __init__(
        self,
        entity_class_to_ontology_mappings: Dict[str, List[str]],
        cache_managers: List[IndexCacheManager],
    ):
        """

        :param entity_class_to_ontology_mappings: mapping of entity classes to ontologies - i.e. which ontologies
            should be queried for each entity class
        :param cache_managers: list of IndexCacheManagers to use for this instance
        """
        self.cache_managers = cache_managers
        self.entity_class_to_ontology_mappings = entity_class_to_ontology_mappings
        self.entity_class_to_indices: Dict[str, Set[Index]] = {}

    def search(self, query: Any, entity_class: str, namespace: str, **kwargs) -> Iterable[Mapping]:
        """
        search across all indices.

        :param query: passed to the search method of each index
        :param entity_class: used to restrict the search space to certain indices - see
            entity_class_to_ontology_mappings in constructor
        :param namespace: the namespace of the calling step (added to mapping metadata)
        :param kwargs: any other kwargs to pass to the search method of each index
        :return:
        """
        indices_to_use = self.entity_class_to_indices.get(entity_class, set())
        for index in indices_to_use:
            mappings = index.search(query=query, **kwargs)
            for mapping in mappings:
                mapping.metadata[NAMESPACE] = namespace
                yield mapping

    def load(self):
        """
        loads the cached version of the indices from disk
        """
        for cache_manager in self.cache_managers:
            all_indices = {
                index.name: index for index in cache_manager.get_or_create_ontology_indices()
            }

            for entity_class, ontologies in self.entity_class_to_ontology_mappings.items():
                current_indices = set()
                for ontology_name in ontologies:
                    index = all_indices.get(ontology_name)
                    if index is None:
                        logger.warning(f"No index found for {ontology_name}")
                    else:
                        current_indices.add(index)

                if not current_indices:
                    logger.warning(f"No indices loaded for entity class {entity_class}")
                self.entity_class_to_indices[entity_class] = current_indices
