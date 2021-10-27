import logging
import os
import pickle
import shutil
from collections import defaultdict
from typing import List, Tuple, Dict

import pandas as pd
import pydash
import torch
from cachetools import LFUCache
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from azner.data.data import Document, Entity, Mapping
from azner.modelling.linking.sapbert.train import (
    PLSapbertModel,
    get_embedding_dataloader_from_strings,
)
from azner.steps import BaseStep
from azner.steps.utils.embedding_index import (
    EmbeddingIndexFactory,
    EmbeddingIndex,
)
from azner.steps.utils.utils import (
    filter_entities_with_ontology_mappings,
    get_match_entity_class_hash,
    get_cache_dir,
    get_cache_path,
)

logger = logging.getLogger(__name__)

IDENTIFIERS_CACHE_ID = "identifiers"


class SapBertForEntityLinkingStep(BaseStep):
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
        This step wraps Sapbert: Self Alignment pretraining for biomedical entity representation.


        We make use of two caches here:
        1) Ontology embeddings
            Since these are static and numerous, it makes sense to precompute them once and reload them each time
            This is done automatically if no cache file is detected. A cache directory is generated with the prefix
            'cache_' alongside the location of the original ontology file

        2) Runtime LFUCache
            Since certain entities will come up more frequently, we cache the result of the embedding check rather than
            call bert repeatedly. This cache is maintained on the basis of the get_match_entity_class_hash(Entity)
            function


        Note, the ontology to link against is held in memory as an ndarray. For very large KB's we should use
        faiss (see Nemo example via SapBert github reference)

        Original paper
        https://aclanthology.org/2021.naacl-main.334.pdf

        :param model: path to HF SAPBERT model, config and tokenizer. A good pretrained default is available at
                            https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
                            This is passed to HF Automodel.from_pretrained()
        :param depends_on: namespaces of dependency stes
        :param ontology_path: path to parquet of labels to map to. This should have three columns: 'source',
                            'iri' and 'default_label'. The default_label will be used to create an embedding, and the
                            source and iri will be used to create a Mapping for the entity.
                            See SapBert paper for further info
        :param batch_size: batch size for dataloader
        :param dl_workers: number of workers for the dataloader
        :param trainer: an instance of pytorch lightning trainer
        :param embedding_index_factory: an instance of EmbeddingIndexFactory, used to instantiate embedding indexes
        :param ontology_partition_size: number of rows to process before saving results, when building initial ontology cache
        :param process_all_entities: bool flag. Since SapBert involves expensive bert calls, this flag controls whether
                                        it should be used on all entities, or only entities that have no mappings (i.e.
                                        entites that have already been linked via a less expensive method, such as
                                        dictionary lookup). This flag check for the presence of at least one entry in
                                        Entity.metadata.mappings
        :param rebuild_ontology_cache: should the ontology embedding cache be rebuilt?
        :param lookup_cache_size: this step maintains a cache of {hash(Entity.match,Entity.entity_class):Mapping}, to reduce bert calls. This dictates the size
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
        self.lookup_cache = LFUCache(lookup_cache_size)

    def get_or_create_ontology_index_dict(self):
        """
        populate self.ontology_ids and self.ontology_index_dict either by calculating them afresh or loading from a cached
        version on disk
        :return:
        """
        cache_dir = get_cache_dir(self.ontology_path, create_if_not_exist=False)
        if self.rebuild_cache:
            logger.info("forcing a rebuild of the ontology cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            get_cache_dir(self.ontology_path, create_if_not_exist=True)
            self.ontology_ids, self.ontology_index_dict = self.cache_ontology_embeddings()
        elif os.path.exists(cache_dir):
            logger.info(f"loading cached ontology file from {cache_dir}")
            (
                self.ontology_ids,
                self.ontology_index_dict,
            ) = self.load_ontology_ids_and_ontology_index_dict_from_cache()
        else:
            logger.info("No ontology cache file found. Building a new one")
            get_cache_dir(self.ontology_path, create_if_not_exist=True)
            self.ontology_ids, self.ontology_index_dict = self.cache_ontology_embeddings()

    def get_ontology_slices_from_full_dataframe(self) -> Tuple[str, pd.DataFrame]:
        full_df = pd.read_parquet(self.ontology_path)
        sources = full_df["source"].unique()
        for source in sources:
            yield source, full_df[full_df["source"] == source]

    def cache_ontology_embeddings(self) -> Tuple[Dict[str, List[str]], Dict[str, EmbeddingIndex]]:
        """
        since the generation of the ontology embeddings is slow, we cache this to disk after this is done once.
        :return: tuple of the ontology_ids lookup dict and the ontology:Index dict
        """
        ontology_ids = defaultdict(list)
        ontology_index_dict = {}
        for ontology, ontology_df in self.get_ontology_slices_from_full_dataframe():
            logger.info(f"creating index for {ontology}")
            index = self.embedding_index_factory.create_index()
            for (
                partition_number,
                ontology_ids_list,
                ontology_embeddings,
            ) in self.predict_ontology_embeddings(ontology_dataframe=ontology_df):
                logger.info(f"processing partition {partition_number} ")
                index.add(ontology_embeddings)
                logger.info(f"index size is now {len(index)}")
                ontology_ids[ontology].extend(ontology_ids_list)

            logger.info("saving index metadata ")
            cache_path = get_cache_path(
                self.ontology_path, cache_id=f"{IDENTIFIERS_CACHE_ID}_{ontology}"
            )
            with open(cache_path, "wb") as f:
                pickle.dump(
                    (
                        ontology_ids[ontology],
                        ontology,
                    ),
                    f,
                )

            index_path = get_cache_path(
                self.ontology_path, cache_id=f"{index.__class__.__name__}_{ontology}"
            )
            index.save(str(index_path.absolute()))
            logger.info(f"saved {ontology} index to {index_path.absolute()}")
            ontology_index_dict[ontology] = index
            logger.info(f"final index size for {ontology} is {len(index)}")
            return ontology_ids, ontology_index_dict

    def load_ontology_ids_and_ontology_index_dict_from_cache(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, EmbeddingIndex]]:
        """
        loads the cached version of ontology ids and associated embedding index from disk
        :return: tuple of the ontology_ids lookup dict and the ontology:Index dict
        """
        ontology_ids: Dict[str, List[str]] = defaultdict(list)
        ontology_index_dict: Dict[str, EmbeddingIndex] = {}

        identifiers_cache_dir = get_cache_dir(self.ontology_path, create_if_not_exist=False)
        ontology_metadata_filenames = [
            x for x in os.listdir(identifiers_cache_dir) if IDENTIFIERS_CACHE_ID in x
        ]
        for filename in ontology_metadata_filenames:
            with open(identifiers_cache_dir.joinpath(filename), "rb") as f:
                ontology_ids_list, ontology_name = pickle.load(f)
                ontology_ids[ontology_name].extend(ontology_ids_list)
                index = self.embedding_index_factory.create_index()
                index_cache_path = get_cache_path(
                    self.ontology_path, cache_id=f"{index.__class__.__name__}_{ontology_name}"
                )

                index.load(str(index_cache_path.absolute()))
                ontology_index_dict[ontology_name] = index

        return ontology_ids, ontology_index_dict

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
    ) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        based on the value of self.ontology_path, this returns a Tuple[List[str],np.ndarray]. The strings are the
        iri's, and the torch.Tensor are the embeddings to be queried against
        :return: partition number, list of iri's, 2d tensor of embeddings
        """

        for partition_number, df in enumerate(
            self.split_dataframe(ontology_dataframe, self.ontology_partition_size)
        ):
            if df.shape[0] == 0:
                return
            logger.info(f"creating partitions for partition {partition_number}")
            logger.info(f"read {df.shape[0]} rows from ontology")
            df.columns = ["source", "iri", "default_label"]

            default_labels = df["default_label"].tolist()
            logger.info(f"predicting embeddings for default_labels. Examples: {default_labels[:3]}")
            results = self.get_embeddings_for_strings(default_labels)
            yield partition_number, df["iri"].tolist(), results

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
        results = self.model.get_embeddings(results, as_index=False)
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
        entities = pydash.flatten([x.get_entities() for x in docs])
        if not self.process_all_entities:
            entities = filter_entities_with_ontology_mappings(entities)

        entities = self.check_lookup_cache(entities)
        if len(entities) > 0:
            results = self.get_embeddings_for_strings([x.match for x in entities])
            results = torch.unsqueeze(results, 1)
            for i, result in enumerate(results):
                entity = entities[i]
                ontology_name = self.entity_class_to_ontology_mappings[entity.entity_class]
                distances, neighbors = self.ontology_index_dict[ontology_name].search(result)
                for ontology_id, dist in zip(neighbors, distances):
                    ontology_id = self.ontology_ids[ontology_name][ontology_id]
                    new_mapping = Mapping(
                        source=ontology_name,
                        idx=ontology_id,
                        mapping_type="direct",
                        metadata={"distance": dist},
                    )
                    entity.add_mapping(new_mapping)
                    self.update_lookup_cache(entity, new_mapping)

        return docs, []

    def update_lookup_cache(self, entity: Entity, mapping: Mapping):
        hash_val = get_match_entity_class_hash(entity)
        if hash_val not in self.lookup_cache:
            self.lookup_cache[hash_val] = mapping

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
            maybe_mapping = self.lookup_cache.get(hash_val, None)
            if maybe_mapping is None:
                cache_misses.append(ent)
            else:
                ent.add_mapping(maybe_mapping)
        return cache_misses
