import logging
import os
import shutil
import traceback
from typing import List, Tuple, Dict

import pandas as pd
import pydash
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from azner.data.data import Document, Mapping, PROCESSING_EXCEPTION, NAMESPACE
from azner.modelling.linking.sapbert.train import (
    PLSapbertModel,
    get_embedding_dataloader_from_strings,
)
from azner.steps import BaseStep
from azner.utils.caching import EntityLinkingLookupCache
from azner.utils.link_index import (
    EmbeddingIndexFactory,
    EmbeddingIndex,
    IDX,
    DEFAULT_LABEL,
    SOURCE,
)
from azner.utils.utils import (
    filter_entities_with_ontology_mappings,
    get_cache_dir,
)

logger = logging.getLogger(__name__)


class SapBertForEntityLinkingStep(BaseStep):
    """
    This step wraps Sapbert: Self Alignment pretraining for biomedical entity representation.
    We make use of two caches here:
    1) :class:`azner.utils.link_index.EmbeddingIndex` Since these are static and numerous, it makes sense to
    precompute them once and reload them each time. This is done automatically if no cache file is detected.
    2) :class:`azner.utils.caching.EntityLinkingLookupCache` Since certain entities will come up more frequently, we
    cache the result mappings rather than call bert repeatedly.

    Original paper https://aclanthology.org/2021.naacl-main.334.pdf
    """

    def __init__(
        self,
        depends_on: List[str],
        model: PLSapbertModel,
        ontology_path: str,
        batch_size: int,
        trainer: Trainer,
        dl_workers: int,
        ontology_partition_size: int,
        embedding_index_factory: EmbeddingIndexFactory,
        entity_class_to_ontology_mappings: Dict[str, str],
        process_all_entities: bool = False,
        rebuild_ontology_cache: bool = False,
        lookup_cache_size: int = 5000,
    ):
        """

        :param depends_on: namespaces of dependency stes
        :param model: a pretrained Sapbert Model
        :param ontology_path: path to file to generate embeddings from. See :meth:`azner.modelling.
            ontology_preprocessing.base.OntologyParser.OntologyParser.write_default_labels` for format
        :param batch_size: for inference with Pytorch
        :param trainer: a pytorch lightning Trainer to handle the inference for us
        :param dl_workers: number fo dataloader workers
        :param ontology_partition_size: when generating embeddings, process n in a partition before serialising to disk.
            (reduce if memory is an issue)
        :param embedding_index_factory: For creating Embedding Indexes
        :param entity_class_to_ontology_mappings: A Dict[str,str] that maps an entity class to the Ontology it should be
            processed against
        :param process_all_entities: if False, ignore entities that already have a mapping
        :param rebuild_ontology_cache: Force rebuild of embedding cache
        :param lookup_cache_size: size of lookup cache to maintain
        """
        super().__init__(depends_on=depends_on)
        self.embedding_index_factory = embedding_index_factory
        self.entity_class_to_ontology_mappings = entity_class_to_ontology_mappings
        self.ontology_partition_size = ontology_partition_size
        self.dl_workers = dl_workers
        self.rebuild_cache = rebuild_ontology_cache
        self.process_all_entities = process_all_entities
        self.batch_size = batch_size
        self.ontology_path = ontology_path
        self.model = model
        self.trainer = trainer
        self.get_or_create_ontology_index_dict()
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    def get_or_create_ontology_index_dict(self):
        """
        populate self.ontology_ids and self.ontology_index_dict either by calculating them afresh or loading from a
        cached version on disk
        """
        cache_dir = get_cache_dir(self.ontology_path, create_if_not_exist=False)
        if self.rebuild_cache:
            logger.info("forcing a rebuild of the ontology cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            get_cache_dir(self.ontology_path, create_if_not_exist=True)
            self.ontology_index_dict = self.cache_ontology_embeddings()
        elif os.path.exists(cache_dir):
            logger.info(f"loading cached ontology file from {cache_dir}")
            self.ontology_index_dict = self.load_ontology_index_dict_from_cache()
        else:
            logger.info("No ontology cache file found. Building a new one")
            get_cache_dir(self.ontology_path, create_if_not_exist=True)
            self.ontology_index_dict = self.cache_ontology_embeddings()

    def get_ontology_slices_from_full_dataframe(self) -> Tuple[str, pd.DataFrame]:
        full_df = pd.read_parquet(self.ontology_path)
        sources = full_df["source"].unique()
        logger.info(f"detected the following sources in the ontology file: {sources}")
        for source in sources:
            yield source, full_df[full_df[SOURCE] == source]

    def cache_ontology_embeddings(self) -> Dict[str, EmbeddingIndex]:
        """
        since the generation of the ontology embeddings is slow, we cache this to disk after this is done once.
        :return: ontology_name:Index dict
        """
        ontology_index_dict = {}
        for ontology_name, ontology_df in self.get_ontology_slices_from_full_dataframe():
            logger.info(f"creating index for {ontology_name}")
            index = self.embedding_index_factory.create_index(ontology_name)
            for (
                partition_number,
                metadata_df,
                ontology_embeddings,
            ) in self.predict_ontology_embeddings(ontology_dataframe=ontology_df):
                logger.info(f"processing partition {partition_number} ")
                index.add(ontology_embeddings, metadata_df)
                logger.info(f"index size is now {len(index)}")

            index_cache_dir = get_cache_dir(self.ontology_path, create_if_not_exist=True)
            index_path = index.save(index_cache_dir)
            logger.info(f"saved {ontology_name} index to {index_path.absolute()}")
            ontology_index_dict[ontology_name] = index
            logger.info(f"final index size for {ontology_name} is {len(index)}")
        return ontology_index_dict

    def load_ontology_index_dict_from_cache(
        self,
    ) -> Dict[str, EmbeddingIndex]:
        """
        loads the cached version of the embedding indices from disk
        :return: ontology:Index dict
        """
        ontology_index_dict: Dict[str, EmbeddingIndex] = {}

        index_cache_dir = get_cache_dir(self.ontology_path, create_if_not_exist=False)
        index_dirs = os.listdir(index_cache_dir)

        for filename in index_dirs:
            # skip hidden files created by os
            if not filename.startswith("."):
                index = self.embedding_index_factory.create_index()
                index.load(index_cache_dir.joinpath(filename))
                ontology_index_dict[index.name] = index

        return ontology_index_dict

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
            results = self.get_embeddings_for_strings(default_labels)
            yield partition_number, df, results

    def get_embeddings_for_strings(self, texts: List[str]) -> torch.Tensor:
        """
        for a list of strings, get the associated embeddings
        :param texts:
        :return: a 2d tensor of embeddings
        """
        loader = get_embedding_dataloader_from_strings(
            texts, self.model.tokeniser, self.batch_size, self.dl_workers
        )
        results = self.get_embeddings_from_dataloader(loader)
        return results

    def get_embeddings_from_dataloader(self, loader: DataLoader) -> torch.Tensor:
        """
        get the cls token output from all data in a dataloader as a 2d tensor
        :param loader:
        :return: 2d tensor of cls  output
        """
        results = self.trainer.predict(
            model=self.model, dataloaders=loader, return_predictions=True
        )
        results = self.model.get_embeddings(results)
        return results

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1) first obtain an entity list from all docs
        2) check the lookup LRUCache to see if it's been recently processed
        3) generate embeddings for the entities based on the value of Entity.match
        4) query this embedding against self.ontology_index_dict to determine the best matches based on cosine distance
        5) generate a new Mapping with the queried iri, and update the entity information
        :param docs:
        :return:
        """
        failed_docs = []
        try:
            entities = pydash.flatten([x.get_entities() for x in docs])
            if not self.process_all_entities:
                entities = filter_entities_with_ontology_mappings(entities)

            entities = self.lookup_cache.check_lookup_cache(entities)
            if len(entities) > 0:
                results = self.get_embeddings_for_strings([x.match for x in entities])
                results = torch.unsqueeze(results, 1)
                for i, result in enumerate(results):
                    entity = entities[i]
                    cache_missed_entities = self.lookup_cache.check_lookup_cache([entity])
                    if len(cache_missed_entities) == 0:
                        continue
                    ontology_name = self.entity_class_to_ontology_mappings[entity.entity_class]
                    index = self.ontology_index_dict.get(ontology_name, None)
                    if index is not None:
                        metadata_df = index.search(result)
                        for i, row in metadata_df.iterrows():
                            metadata_dict = row.to_dict()
                            metadata_dict[NAMESPACE] = self.namespace()
                            ontology_id = metadata_dict.pop(IDX)
                            # note, we set mapping type to inferred as sapbert doesn't really have the concept
                            new_mapping = Mapping(
                                source=ontology_name,
                                idx=ontology_id,
                                mapping_type=["inferred"],
                                metadata=metadata_dict,
                            )
                            entity.add_mapping(new_mapping)
                            self.lookup_cache.update_lookup_cache(entity, new_mapping)
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)

        return docs, failed_docs
