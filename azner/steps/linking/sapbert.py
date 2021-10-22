import logging
import os
import pickle
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd
import pydash
import torch
from azner.data.data import Document, Entity, Mapping
from azner.data.pytorch import HFDataset
from azner.modelling.hf_lightning_wrappers import PLAutoModel
from azner.steps import BaseStep
from cachetools import LFUCache
from pytorch_lightning import Trainer
from scipy.spatial.distance import cdist
from azner.steps.utils.utils import (
    filter_entities_with_kb_mappings,
    get_match_entity_class_hash,
    update_mappings,
    get_cache_dir,
    get_cache_partition_path,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModel,
)
from transformers.file_utils import PaddingStrategy

logger = logging.getLogger(__name__)


class SapBertForEntityLinkingStep(BaseStep):
    def __init__(
        self,
        depends_on: List[str],
        model_path: str,
        knowledgebase_path: str,
        batch_size: int,
        trainer: Trainer,
        dl_workers: int,
        kb_partition_size: int,
        process_all_entities: bool = False,
        rebuild_kb_cache: bool = False,
        lookup_cache_size: int = 5000,
    ):
        """
        This step wraps Sapbert: Self Alignment pretraining for biomedical entity representation.

        Note, the knowledgebase to link against is held in memory as an ndarray. For very large KB's we should use
        faiss (see Nemo example via SapBert github reference)

        Original paper
        https://aclanthology.org/2021.naacl-main.334.pdf

        :param model_path: path to HF SAPBERT model, config and tokenizer. A good pretrained default is available at
                            https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
                            This is passed to HF Automodel.from_pretrained()
        :param depends_on: namespaces of dependency stes
        :param knowledgebase_path: path to parquet of labels to map to. This should have three columns: 'source',
                            'iri' and 'default_label'. The default_label will be used to create an embedding, and the
                            source and iri will be used to create a Mapping for the entity.
                            See SapBert paper for further info
        :param batch_size: batch size for dataloader
        :param dl_workers: number of workers for the dataloader
        :param trainer: an instance of pytorch lightning trainer
        :param kb_partition_size: number of rows to process before saving results, when building initial kb cache
        :param process_all_entities: bool flag. Since SapBert involves expensive bert calls, this flag controls whether
                                        it should be used on all entities, or only entities that have no mappings (i.e.
                                        entites that have already been linked via a less expensive method, such as
                                        dictionary lookup). This flag check for the presence of at least one entry in
                                        Entity.metadata.mappings
        :param rebuild_kb_cache: should the kb embedding cache be rebuilt?
        :param lookup_cache_size: this step maintains a cache of {hash(Entity.match,Entity.entity_class):Mapping}, to reduce bert calls. This dictates the size
        """
        super().__init__(depends_on=depends_on)
        self.dl_workers = dl_workers
        self.rebuild_cache = rebuild_kb_cache
        self.process_all_entities = process_all_entities
        self.batch_size = batch_size
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokeniser = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.knowledgebase_path = knowledgebase_path
        self.model = AutoModel.from_pretrained(model_path, config=self.config)
        self.model = PLAutoModel(self.model)
        self.trainer = trainer
        self.get_or_create_kb_embedding_cache()
        self.lookup_cache = LFUCache(lookup_cache_size)

    def get_or_create_kb_embedding_cache(self):
        """
        populate self.kb_ids and self.kb_embeddings either by calculating them afresh or loading from a cached version
        on disk
        :return:
        """
        cache_dir = get_cache_dir(self.knowledgebase_path, create_if_not_exist=False)
        if self.rebuild_cache:
            logger.info("forcing a rebuild of the kb cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            get_cache_dir(self.knowledgebase_path, create_if_not_exist=True)
            self.cache_kb_embeddings()
        elif os.path.exists(cache_dir):
            logger.info(f"loading cached kb file from {cache_dir}")
            self.load_kb_cache()
        else:
            logger.info("No kb cache file found. Building a new one")
            get_cache_dir(self.knowledgebase_path, create_if_not_exist=True)
            self.cache_kb_embeddings()

    def cache_kb_embeddings(self):
        """
        since the generation of the knowledgebase embeddings is slow, we cache this to disk after this is done once
        :return:
        """
        self.sources, self.kb_ids, self.kb_embeddings = [], [], []
        for partition_number, sources, kb_ids, kb_embeddings in self.predict_kb_embeddings():
            logger.info(f"saving partition {partition_number}")
            cache_path = get_cache_partition_path(
                self.knowledgebase_path, partition_number=partition_number
            )
            with open(cache_path, "wb") as f:
                pickle.dump(
                    (
                        sources,
                        kb_ids,
                        kb_embeddings,
                    ),
                    f,
                )
            self.sources.extend(sources)
            self.kb_embeddings.extend(kb_embeddings)
            self.kb_ids.extend(kb_ids)

    def load_kb_cache(self):
        self.sources, self.kb_ids, self.kb_embeddings = [], [], []
        cache_path = get_cache_dir(self.knowledgebase_path, create_if_not_exist=False)
        cache_files = os.listdir(cache_path)
        for file in cache_files:
            with open(cache_path.joinpath(file), "rb") as f:
                sources, kb_ids, kb_embeddings = pickle.load(f)
                self.sources.extend(sources)
                self.kb_ids.extend(kb_ids)
                self.kb_embeddings.extend(kb_embeddings)

    def split_dataframe(self, df: pd.DataFrame, chunk_size=100000):
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            yield df[i * chunk_size : (i + 1) * chunk_size]

    def predict_kb_embeddings(self) -> Tuple[List[str], List[str], np.ndarray]:
        """
        based on the value of self.knowledgebase_path, this returns a Tuple[List[str],np.ndarray]. The strings are the
        iri's, and the ndarray are the embeddings to be queried against
        :return:
        """

        full_df = pd.read_parquet(self.knowledgebase_path)

        for partition_number, df in enumerate(self.split_dataframe(full_df, 50000)):
            logger.info(f"creating partitions for partition {partition_number}")
            logger.info(f"read {df.shape[0]} rows from kb")
            df.columns = ["source", "iri", "default_label"]

            default_labels = df["default_label"].tolist()
            batch_encodings = self.tokeniser(default_labels)
            dataset = HFDataset(batch_encodings)
            collate_func = DataCollatorWithPadding(
                tokenizer=self.tokeniser, padding=PaddingStrategy.LONGEST
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=collate_func,
                num_workers=self.dl_workers,
            )

            results = self.trainer.predict(
                model=self.model, dataloaders=loader, return_predictions=True
            )
            results = torch.cat([x.pooler_output for x in results]).cpu().detach().numpy()
            logger.info("knowledgebase embedding generation successful")
            yield partition_number, df["source"].tolist(), df["iri"].tolist(), results

    def get_dataloader_for_entities(self, entities: List[Entity]) -> DataLoader:
        """
        get a dataloader and entity list from a List of Document. Collation is handled via DataCollatorWithPadding
        :param docs:
        :return:
        """

        batch_encoding = self.tokeniser([x.match for x in entities])
        dataset = HFDataset(batch_encoding)
        collate_func = DataCollatorWithPadding(
            tokenizer=self.tokeniser, padding=PaddingStrategy.LONGEST
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=collate_func,
            num_workers=self.dl_workers,
        )
        return loader

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1) first obtain an entity list from all docs
        2) check the cache
        3) generate embeddings for the entities based on the value of Entity.match
        4) query this embedding against self.kb_embeddings to determine the best match based on cosine distance
        5) generate a new Mapping with the queried iri, and update the entity information
        :param docs:
        :return:
        """
        entities = pydash.flatten([x.get_entities() for x in docs])
        if not self.process_all_entities:
            entities = filter_entities_with_kb_mappings(entities)

        entities = self.check_lookup_cache(entities)
        if len(entities) > 0:
            loader = self.get_dataloader_for_entities(entities)
            results = self.trainer.predict(
                model=self.model, dataloaders=loader, return_predictions=True
            )
            results = (
                torch.unsqueeze(torch.cat([x.pooler_output for x in results]), 1)
                .cpu()
                .detach()
                .numpy()
            )

            for i, result in enumerate(results):
                dist = cdist(result, self.kb_embeddings)
                nn_index = np.argmin(dist)
                entity = entities[i]
                new_mapping = Mapping(
                    source=self.sources[nn_index], idx=self.kb_ids[nn_index], mapping_type="direct"
                )
                update_mappings(entity, new_mapping)
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
                update_mappings(ent, maybe_mapping)
        return cache_misses
