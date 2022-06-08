import itertools
import logging
import sys
import traceback
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Dict, Set

import pydash
import torch
from pytorch_lightning import Trainer

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity, SearchRanks, Hit
from kazu.modelling.linking.sapbert.train import PLSapbertModel
from kazu.steps import BaseStep
from kazu.utils.caching import EntityLinkingLookupCache
from kazu.utils.link_index import EmbeddingIndex

logger = logging.getLogger(__name__)


class SapBertForEntityLinkingStep(BaseStep):
    """
    This step wraps Sapbert: Self Alignment pretraining for biomedical entity representation.
    We make use of two caches here:
    1) :class:`kazu.utils.caching.CachedIndexGroup` Since we need to calculate embeddings for all labels in an ontology
    , it makes sense to precompute them once and reload them each time. This is done automatically if no cache file is
    detected.
    2) :class:`kazu.utils.caching.EntityLinkingLookupCache` Since certain entities will come up more frequently, we
    cache the result mappings rather than call bert repeatedly.

    Original paper https://aclanthology.org/2021.naacl-main.334.pdf
    """

    def __init__(
        self,
        depends_on: List[str],
        indices: List[EmbeddingIndex],
        embedding_model: PLSapbertModel,
        trainer: Trainer,
        entity_class_to_ontology_mappings: Dict[str, List[str]],
        min_string_length_to_trigger: Optional[Dict[str, int]] = None,
        ignore_high_conf: bool = True,
        lookup_cache_size: int = 5000,
        top_n: int = 20,
        score_cutoffs: Tuple[float, float] = (99.9940, 99.995),
        batch_size: int = 16,
    ):
        """

        :param depends_on:
        :param index_group: an instance of CachedIndexGroup constructed with a list of EmbeddingIndexCacheManager
        :param min_string_length_to_trigger: a per entity class mapping that signals sapbert will not run on matches
            shorter than this. (sapbert is less good at symbolic matching than string processing techniques)
        :param ignore_high_conf: don't process ents that already have at least one high confidence hit(a.k.a perfect match)
            i.e. if we've scored a perfect match already, sapbert unlikely to improve
        :param lookup_cache_size: the size of the Least Recently Used lookup cache to maintain
        :param top_n: keep up to the top_n hits of the query
        :param score_cutoffs: min score for a hit to be considered. first is lower bound for medium confidence,
            second is upper bound for med high confidence
        """

        super().__init__(depends_on=depends_on)
        self.batch_size = batch_size
        self.entity_class_to_ontology_mappings = entity_class_to_ontology_mappings
        self.trainer = trainer
        self.embedding_model = embedding_model
        self.ignore_high_conf = ignore_high_conf
        self.min_string_length_to_trigger = min_string_length_to_trigger
        self.score_cutoffs = score_cutoffs
        self.top_n = top_n
        self.indices = indices
        self.entity_class_to_indices: Dict[str, Set[EmbeddingIndex]] = {}
        self.load_or_build_caches()

        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    def load_or_build_caches(self):
        for index in self.indices:
            index.set_embedding_model(self.embedding_model)
            index.load_or_build_cache()
        all_indices = {index.parser.name: index for index in self.indices}

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

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1) first obtain an entity list from all docs
        2) check the lookup LRUCache to see if it's been recently processed
        3) generate embeddings for the entities based on the value of Entity.match
        4) query this embedding against self.index_group to determine the best matches based on cosine distance
        5) generate a new Mapping, and update the entity information/LRUCache
        :param docs:
        :return:
        """
        failed_docs = []
        try:
            entities_to_process = []
            for ent in pydash.flatten([x.get_entities() for x in docs]):
                if ent.entity_class not in self.entity_class_to_indices.keys():
                    continue
                if self.ignore_high_conf:
                    # check every parser namespace has a high conf hit
                    hits_by_parser_name = itertools.groupby(
                        sorted(ent.hits, key=lambda x: x.parser_name), key=lambda x: x.parser_name
                    )
                    needs_sapbert = False
                    for parser_name, hits_iter in hits_by_parser_name:
                        if not any(hit.confidence == SearchRanks.EXACT_MATCH for hit in hits_iter):
                            needs_sapbert = True
                            break
                    if needs_sapbert:
                        entities_to_process.append(ent)

            self.process_entities(entities_to_process)
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)

        return docs, failed_docs

    def process_entities(self, entities: Iterable[Entity]):
        entities = self.lookup_cache.check_lookup_cache(entities)

        if len(entities) > 0:
            entity_string_to_ent_list = defaultdict(list)
            # group entities by their match string, so we only need to process one string for all matches in a set
            # of documents
            for x in entities:
                if self.min_string_length_to_trigger and self.min_string_length_to_trigger.get(
                    x.entity_class, sys.maxsize
                ) > len(x.match):
                    continue
                entity_string_to_ent_list[x.match].append(x)
            if len(entity_string_to_ent_list) > 0:

                results = self.embedding_model.get_embeddings_for_strings(
                    list(entity_string_to_ent_list.keys()),
                    trainer=self.trainer,
                    batch_size=self.batch_size,
                )
                results = torch.unsqueeze(results, 1)
                # look over the matched string and associated entities, updating the cache as we go
                for entities_grouped, result in zip(entity_string_to_ent_list.values(), results):
                    for entity in entities_grouped:
                        cache_missed_entities = self.lookup_cache.check_lookup_cache([entity])
                        if not len(cache_missed_entities) == 0:
                            indices_to_search = self.entity_class_to_indices.get(
                                entity.entity_class
                            )
                            if indices_to_search:
                                all_hits: List[Hit] = []
                                for index in indices_to_search:
                                    all_hits.extend(index.search(result, self.top_n))
                                for hit in all_hits:
                                    hit.namespace = self.namespace()
                                entity.hits.extend(all_hits)
                                self.lookup_cache.update_hits_lookup_cache(entity, all_hits)
