import logging
import sys
import traceback
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Dict

import pydash
import torch
from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity, LINK_CONFIDENCE, LinkRanks
from kazu.steps import BaseStep
from kazu.utils.caching import (
    EntityLinkingLookupCache,
    CachedIndexGroup,
    EmbeddingIndexCacheManager,
)
from kazu.utils.utils import HitResolver

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
        index_group: CachedIndexGroup,
        min_string_length_to_trigger: Optional[Dict[str, int]] = None,
        lookup_cache_size: int = 5000,
        top_n: int = 20,
        score_cutoffs: Tuple[float, float] = (99.9940, 99.995),
    ):
        """

        :param depends_on:
        :param index_group: an instance of CachedIndexGroup constructed with a list of EmbeddingIndexCacheManager
        :param min_string_length_to_trigger: a per entity class mapping that signals sapbert will not run on matches
            shorter than this. (sapbert is less good at symbolic matching than string processing techniques)
        :param lookup_cache_size: the size of the Least Recently Used lookup cache to maintain
        :param top_n: keep up to the top_n hits of the query
        :param score_cutoffs: min score for a hit to be considered. first is lower bound for medium confidence,
            second is upper bound for med high confidence
        """

        super().__init__(depends_on=depends_on)
        self.min_string_length_to_trigger = min_string_length_to_trigger
        self.score_cutoffs = score_cutoffs
        if not all([isinstance(x, EmbeddingIndexCacheManager) for x in index_group.cache_managers]):
            raise RuntimeError(
                "The CachedIndexGroup must be configured with an EmbeddingIndexCacheManager to work"
                "correctly with the Sapbert Step"
            )

        if len(index_group.cache_managers) > 1:
            logger.warning(
                f"multiple cache managers detected for {self.namespace()}. This may mean you are loading"
                f"multiple instances of a model, which is memory inefficient. In addition, this instance will"
                f" reuse the model data associated with the first detected cache manager. This may have "
                f"unintended consequences."
            )

        self.top_n = top_n
        self.index_group = index_group
        self.index_group.load()
        # we reuse the instance of the model associated with the cache manager, so we don't have to instantiate it twice
        reusable_cache_manager = index_group.cache_managers[0]
        if isinstance(reusable_cache_manager, EmbeddingIndexCacheManager):
            self.dl_workers = reusable_cache_manager.dl_workers
            self.batch_size = reusable_cache_manager.batch_size
            self.model = reusable_cache_manager.model
            self.trainer = reusable_cache_manager.trainer
        else:
            raise ValueError(
                "the Sapbert step requires a CachedIndexGroup of type  EmbeddingIndexCacheManager"
            )
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

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
            entities = pydash.flatten([x.get_entities() for x in docs])
            # filter entities that have no mapping in cache manager
            entities = filter(
                lambda x: x.entity_class in self.index_group.entity_class_to_indices.keys(),
                entities,
            )
            self.process_entities(entities)
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

                results = self.model.get_embeddings_for_strings(
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
                            hits = list(
                                self.index_group.search(
                                    query=result,
                                    entity_class=entity.entity_class,
                                    namespace=self.namespace(),
                                    top_n=self.top_n,
                                    score_cutoffs=self.score_cutoffs,
                                    original_string=entity.match,
                                )
                            )
                            entity.hits.extend(hits)
                            self.lookup_cache.update_hits_lookup_cache(entity, hits)
