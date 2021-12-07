import logging
from pathlib import Path
from typing import Type, Dict, Any

from cachetools import LFUCache

from azner.data.data import Entity, LINK_SCORE, NAMESPACE
from azner.modelling.ontology_preprocessing.base import OntologyParser, IDX, MAPPING_TYPE
from azner.utils.link_index import (
    Index,
    MatMulTensorEmbeddingIndex,
    FaissEmbeddingIndex,
    CDistTensorEmbeddingIndex,
    DictionaryIndex,
)
from azner.utils.utils import get_match_entity_class_hash


import os
import shutil
from typing import List, Tuple

import pandas as pd
import torch
from pytorch_lightning import Trainer

from azner.data.data import Mapping
from azner.modelling.linking.sapbert.train import (
    PLSapbertModel,
)
from azner.modelling.ontology_preprocessing.base import (
    DEFAULT_LABEL,
    SOURCE,
)
from azner.utils.utils import (
    get_cache_dir,
)

logger = logging.getLogger(__name__)


def select_index_type(embedding_index_class_name) -> Type:
    if embedding_index_class_name == MatMulTensorEmbeddingIndex.__name__:
        return MatMulTensorEmbeddingIndex
    elif embedding_index_class_name == FaissEmbeddingIndex.__name__:
        return FaissEmbeddingIndex
    elif embedding_index_class_name == CDistTensorEmbeddingIndex.__name__:
        return CDistTensorEmbeddingIndex
    elif embedding_index_class_name == DictionaryIndex.__name__:
        return DictionaryIndex
    else:
        raise NotImplementedError(f"{embedding_index_class_name} not implemented in factory")


class EntityLinkingLookupCache:
    """
    A simple wrapper around LFUCache to reduce calls to expensive processes (e.g. bert)
    """

    def __init__(self, lookup_cache_size: int = 5000):
        self.lookup_cache = LFUCache(lookup_cache_size)

    def update_lookup_cache(self, entity: Entity, mapping: Mapping):
        hash_val = get_match_entity_class_hash(entity)
        if hash_val not in self.lookup_cache:
            self.lookup_cache[hash_val] = [mapping]
        else:
            cache_hit = self.lookup_cache[hash_val]
            self.lookup_cache[hash_val] = cache_hit + [mapping]

    def check_lookup_cache(self, entities: List[Entity]) -> List[Entity]:
        """
        checks the cache for mappings. If relevant mappings are found for an entity, update it's mappings
        accordingly. If not return as a list of cache misses (e.g. for further processing)

        :param entities:
        :return:
        """
        cache_misses = []
        for ent in entities:
            hash_val = get_match_entity_class_hash(ent)
            maybe_mappings = self.lookup_cache.get(hash_val, None)
            if maybe_mappings is None:
                cache_misses.append(ent)
            else:
                for mapping in maybe_mappings:
                    ent.add_mapping(mapping)
        return cache_misses


class OntologyCacheManager:
    def __init__(self, index_type: str, rebuild_cache: bool = False):
        self.index_type = select_index_type(index_type)
        self.rebuild_cache = rebuild_cache

    def get_or_create_ontology_index(self, parser: OntologyParser) -> Index:
        """
        populate self.ontology_ids and self.ontology_index_dict either by calculating them afresh or loading from a
        cached version on disk
        """
        cache_dir = get_cache_dir(
            parser.in_path, prefix=self.index_type.__name__, create_if_not_exist=False
        )
        if self.rebuild_cache:
            logger.info("forcing a rebuild of the ontology cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            get_cache_dir(parser.in_path, prefix=self.index_type.__name__, create_if_not_exist=True)
            return self.build_ontology_cache(cache_dir, parser)
        elif os.path.exists(cache_dir):
            logger.info(f"loading cached ontology file from {cache_dir}")
            return self.load_ontology_from_cache(cache_dir, parser)
        else:
            logger.info("No ontology cache file found. Building a new one")
            get_cache_dir(parser.in_path, prefix=self.index_type.__name__, create_if_not_exist=True)
            return self.build_ontology_cache(cache_dir, parser)

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        raise NotImplementedError()

    def load_ontology_from_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        return Index.load(str(cache_dir), parser.name)


class DictionaryOntologyCacheManager(OntologyCacheManager):
    def __init__(self, index_type: str, rebuild_cache: bool = False):
        super().__init__(index_type, rebuild_cache)

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:
        logger.info(f"creating index for {parser.in_path}")
        index = self.index_type(name=parser.name)
        ontology_df = parser.format_default_labels()
        synonym_df = parser.synonym_table
        index.add(synonym_df, ontology_df)
        index_path = index.save(str(cache_dir))
        logger.info(f"saved {index.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {index.name} is {len(index)}")
        return index


class EmbeddingOntologyCacheManager(OntologyCacheManager):
    def __init__(
        self,
        model: PLSapbertModel,
        batch_size: int,
        trainer: Trainer,
        dl_workers: int,
        ontology_partition_size: int,
        index_type: str,
        rebuild_cache: bool = False,
    ):

        super().__init__(index_type, rebuild_cache)
        self.ontology_partition_size = ontology_partition_size
        self.dl_workers = dl_workers
        self.batch_size = batch_size
        self.model = model
        self.trainer = trainer

    def build_ontology_cache(self, cache_dir: Path, parser: OntologyParser) -> Index:

        logger.info(f"creating index for {parser.in_path}")

        index = self.index_type(parser.name)
        ontology_df = parser.format_default_labels()
        for (
            partition_number,
            metadata_df,
            ontology_embeddings,
        ) in self.predict_ontology_embeddings(ontology_dataframe=ontology_df):
            logger.info(f"processing partition {partition_number} ")
            index.add(ontology_embeddings, metadata_df)
            logger.info(f"index size is now {len(index)}")
        index_path = index.save(cache_dir)
        logger.info(f"saved {parser.name} index to {index_path.absolute()}")
        logger.info(f"final index size for {parser.name} is {len(index)}")
        return index

    def split_dataframe(self, df: pd.DataFrame, chunk_size: int = 100000):
        """
        generator to split up a dataframe into partitions
        :param df:
        :param chunk_size: size of partittions to create
        :return:
        """
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            yield df[i * chunk_size : (i + 1) * chunk_size]

    def predict_ontology_embeddings(
        self, ontology_dataframe: pd.DataFrame
    ) -> Tuple[List[str], pd.DataFrame, torch.Tensor]:
        """
        based on the value of self.ontology_path, this returns a Tuple[List[str],np.ndarray]. The strings are the
        iri's, and the torch.Tensor are the embeddings to be queried against
        :return: partition number, dataframe for metadata, 2d tensor of embeddings
        """

        for partition_number, df in enumerate(
            self.split_dataframe(ontology_dataframe, self.ontology_partition_size)
        ):
            df = df.copy()
            if df.shape[0] == 0:
                return
            logger.info(f"creating partitions for partition {partition_number}")
            logger.info(f"read {df.shape[0]} rows from ontology")
            default_labels = df[DEFAULT_LABEL].tolist()
            logger.info(f"predicting embeddings for default_labels. Examples: {default_labels[:3]}")
            results = self.model.get_embeddings_for_strings(
                texts=default_labels, trainer=self.trainer, batch_size=self.batch_size
            )
            yield partition_number, df, results


class CachedIndexGroup:
    """
    Convenience class for building and managing a group of indexes, and querying against them
    """

    def __init__(
        self,
        entity_class_to_ontology_mappings: Dict[str, List[str]],
        parsers: List[OntologyParser],
        cache_manager: OntologyCacheManager,
    ):
        self.cache_manager = cache_manager
        self.parsers = parsers
        self.entity_class_to_ontology_mappings = entity_class_to_ontology_mappings
        self.ontology_index_dict: Dict[str, Index] = {}

    def search(self, query: Any, entity_class: str, namespace: str, **kwargs) -> List[Mapping]:
        ontologies_to_search = self.entity_class_to_ontology_mappings.get(entity_class, [])
        results = []
        for ontology_name in ontologies_to_search:
            index = self.ontology_index_dict.get(ontology_name)
            if index is None:
                logger.warning(f"tried to search indices for {ontology_name}, but none were found")
            else:
                index_results = index.search(query=query, **kwargs)
                index_results[SOURCE] = index.name
                results.append(index_results)

        mappings = []
        if len(results) > 0:
            results = pd.concat(results).sort_values(by=LINK_SCORE)
            for i, row in results.iterrows():
                row_dict = row.to_dict()
                ontology_id = row_dict.pop(IDX)
                mapping_type = row_dict.pop(MAPPING_TYPE)
                if not isinstance(mapping_type, list):
                    mapping_type = [mapping_type]
                source = row_dict.pop(SOURCE)
                row_dict[NAMESPACE] = namespace
                new_mapping = Mapping(
                    source=source,
                    idx=ontology_id,
                    mapping_type=mapping_type,
                    metadata=row_dict,
                )
                mappings.append(new_mapping)
        return mappings

    def load(self):
        """
        loads the cached version of the embedding indices from disk
        :return: ontology:Index dict
        """
        for parser in self.parsers:
            index = self.cache_manager.get_or_create_ontology_index(parser)
            self.ontology_index_dict[index.name] = index
