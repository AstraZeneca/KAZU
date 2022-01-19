import logging
import traceback
from collections import defaultdict
from typing import List, Tuple

import pydash
import torch

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.utils.caching import (
    EntityLinkingLookupCache,
    CachedIndexGroup,
    EmbeddingIndexCacheManager,
)

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
        lookup_cache_size: int = 5000,
        top_n: int = 20,
        score_cutoff: float = 99.0,
    ):
        """

        :param depends_on:
        :param index_group: an instance of CachedIndexGroup constructed with a list of EmbeddingIndexCacheManager
        :param lookup_cache_size: the size of the Least Recently Used lookup cache to maintain
        :param top_n: keep up to the top_n hits of the query
        :param score_cutoff: min score for a hit to be considered
        """

        super().__init__(depends_on=depends_on)

        self.score_cutoff = score_cutoff
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
        self.dl_workers = index_group.cache_managers[0].dl_workers
        self.batch_size = index_group.cache_managers[0].batch_size
        self.model = index_group.cache_managers[0].model
        self.trainer = index_group.cache_managers[0].trainer
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
            entities = self.lookup_cache.check_lookup_cache(entities)
            entity_string_to_ent_list = defaultdict(list)
            # group entities by their match string, so we only need to process one string for all matches in a set
            # of documents
            [entity_string_to_ent_list[x.match].append(x) for x in entities]

            if len(entities) > 0:
                results = self.model.get_embeddings_for_strings(
                    list(entity_string_to_ent_list.keys()),
                    trainer=self.trainer,
                    batch_size=self.batch_size,
                )
                results = torch.unsqueeze(results, 1)
                # look over the matched string and associated entities, updating the cache as we go
                for match_key, result in zip(entity_string_to_ent_list, results):
                    for entity in entity_string_to_ent_list[match_key]:
                        cache_missed_entities = self.lookup_cache.check_lookup_cache([entity])
                        if not len(cache_missed_entities) == 0:
                            mappings = self.index_group.search(
                                query=result,
                                entity_class=entity.entity_class,
                                namespace=self.namespace(),
                                top_n=self.top_n,
                                score_cutoff=self.score_cutoff,
                            )
                            for mapping in mappings:
                                entity.add_mapping(mapping)
                                self.lookup_cache.update_lookup_cache(entity, mapping)
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)

        return docs, failed_docs
