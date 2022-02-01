import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, Dict, Any, Iterable, Tuple, List, Set, Generator

import pandas as pd
import torch
from cachetools import LFUCache
from pytorch_lightning import Trainer

from kazu.data.data import Entity, LINK_SCORE, NAMESPACE
from kazu.data.data import Mapping
from kazu.modelling.linking.sapbert.train import (
    PLSapbertModel,
)
from kazu.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
    SOURCE,
)
from kazu.modelling.ontology_preprocessing.base import OntologyParser, MAPPING_TYPE
from kazu.utils.link_index import (
    Index,
    MatMulTensorEmbeddingIndex,
    FaissEmbeddingIndex,
    CDistTensorEmbeddingIndex,
    DictionaryIndex,
)
from kazu.utils.utils import (
    get_cache_dir,
)
from kazu.utils.utils import get_match_entity_class_hash

logger = logging.getLogger(__name__)


def select_index_type(index_class_name) -> Type[Index]:
    """
    select a index type based on its string name
    :param index_class_name:
    :return:
    """
    if index_class_name == MatMulTensorEmbeddingIndex.__name__:
        return MatMulTensorEmbeddingIndex
    elif index_class_name == FaissEmbeddingIndex.__name__:
        return FaissEmbeddingIndex
    elif index_class_name == CDistTensorEmbeddingIndex.__name__:
        return CDistTensorEmbeddingIndex
    elif index_class_name == DictionaryIndex.__name__:
        return DictionaryIndex
    else:
        raise NotImplementedError(f"{index_class_name} not implemented in factory")


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
                    ent.add_mapping(mapping)
        return cache_misses


class IndexCacheManager(ABC):
    """
    An IndexCacheManager is responsible for creating, saving and loading a set of :class:`kazu.utils.caching.Index`.
    It's useful to use this class, instead of an instance of :class:`kazu.utils.caching.Index` directly, as this class
    will automatically cache any expensive index creation aspects (such as embedding generation for sapbert)
    """

    def __init__(self, index_type: str, parsers: List[OntologyParser], rebuild_cache: bool = False):
        self.parsers = parsers
        self.index_type = select_index_type(index_type)
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


class DictionaryIndexCacheManager(IndexCacheManager):
    """
    implementation for use with :class:`kazu.steps.linking.dictionary.DictionaryEntityLinkingStep`
    """

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        logger.info(f"creating index for {parser.in_path}")
        index = self.index_type(name=parser.name)
        ontology_df = parser.get_ontology_metadata()
        synonym_df = parser.get_ontology_synonyms()
        index.add(synonym_df=synonym_df, metadata_df=ontology_df)
        index_path = index.save(str(cache_dir))
        logger.info(f"saved {index.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {index.name} is {len(index)}")
        return index


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

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:

        logger.info(f"creating index for {parser.in_path}")

        index = self.index_type(parser.name)
        ontology_df = parser.get_ontology_metadata()
        for (
            partition_number,
            metadata_df,
            ontology_embeddings,
        ) in self.predict_ontology_embeddings(ontology_dataframe=ontology_df):
            logger.info(f"processing partition {partition_number} ")
            index.add(embeddings=ontology_embeddings, metadata=metadata_df)
            logger.info(f"index size is now {len(index)}")
        index_path = index.save(cache_dir)
        logger.info(f"saved {parser.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {parser.name} is {len(index)}")
        return index

    @staticmethod
    def enumerate_dataframe_chunks(
        df: pd.DataFrame, chunk_size: int = 100000
    ) -> Iterable[Tuple[int, pd.DataFrame]]:
        """
        generator to split up a dataframe into partitions
        :param df:
        :param chunk_size: size of partittions to create
        :return:
        """
        for i in range(0, len(df), chunk_size):
            yield i, df[i : i + chunk_size]

    def predict_ontology_embeddings(
        self, ontology_dataframe: pd.DataFrame
    ) -> Generator[Tuple[int, pd.DataFrame, torch.Tensor], None, None]:
        """
        since embeddings are memory hungry, we use a generator to partition an input dataframe into manageable chucks,
        and add them to the index sequentially
        :param ontology_dataframe:
        :return: partition number, metadata dataframe and embeddings
        """

        for partition_number, df in self.enumerate_dataframe_chunks(
            ontology_dataframe, self.ontology_partition_size
        ):
            len_df = len(df)
            if len_df == 0:
                return
            logger.info(f"creating partitions for partition {partition_number}")
            logger.info(f"read {len_df} rows from ontology")
            default_labels = df[DEFAULT_LABEL].tolist()
            logger.info(f"predicting embeddings for default_labels. Examples: {default_labels[:3]}")
            results = self.model.get_embeddings_for_strings(
                texts=default_labels, trainer=self.trainer, batch_size=self.batch_size
            )
            yield partition_number, df, results


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

    def search(self, query: Any, entity_class: str, namespace: str, **kwargs) -> List[Mapping]:
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
        results = []
        for index in indices_to_use:
            index_results = index.search(query=query, **kwargs)
            index_results[SOURCE] = index.name
            results.append(index_results)

        mappings = []
        len_results = len(results)
        if len_results > 0:
            if len_results == 1:
                # results contains a single, already-sorted DataFrame
                # so avoid sorting again
                results_df = results[0]
            else:
                results_df = pd.concat(results).sort_values(by=LINK_SCORE, ascending=False)
            for ontology_id, row in results_df.iterrows():
                row_dict: Dict = row.to_dict()
                mapping_type = [str(x) for x in row_dict.pop(MAPPING_TYPE)]
                source = row_dict.pop(SOURCE)
                default_label = row_dict.pop(DEFAULT_LABEL)
                row_dict[NAMESPACE] = namespace
                new_mapping = Mapping(
                    default_label=str(default_label),
                    source=str(source),
                    idx=str(ontology_id),
                    mapping_type=mapping_type,
                    metadata=row_dict,
                )
                mappings.append(new_mapping)
        return mappings

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
