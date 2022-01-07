import logging
import traceback
from typing import List, Tuple

import pydash

from azner.data.data import Document, PROCESSING_EXCEPTION
from azner.steps import BaseStep
from azner.utils.caching import CachedIndexGroup, DictionaryIndexCacheManager
from azner.utils.caching import EntityLinkingLookupCache
from azner.utils.utils import (
    find_document_from_entity,
)

logger = logging.getLogger(__name__)


class DictionaryEntityLinkingStep(BaseStep):
    """
    Uses synonym lists to match entities to ontologies.
    """

    def __init__(
        self,
        depends_on: List[str],
        index_group: CachedIndexGroup,
        lookup_cache_size: int = 5000,
        fuzzy: bool = True,
        top_n: int = 20,
        score_cutoff: float = 90.0,
    ):
        """

        :param depends_on:
        :param index_group: A CachedIndexGroup constructed with List[DictionaryIndexCacheManager]
        :param lookup_cache_size: the size of the Least Recently Used lookup cache to maintain
        :param fuzzy: query the CachedIndexGroup with fuzzy string matching
        :param top_n: keep the top_n hits of the query
        :param score_cutoff: min score for a hit to be considered
        """
        super().__init__(depends_on=depends_on)
        self.score_cutoff = score_cutoff
        if not all(
            [isinstance(x, DictionaryIndexCacheManager) for x in index_group.cache_managers]
        ):
            raise RuntimeError(
                "The CachedIndexGroup must be configured with an DictionaryIndexCacheManager to work"
                "correctly with the DictionaryEntityLinkingStep"
            )
        self.top_n = top_n
        self.fuzzy = fuzzy
        self.index_group = index_group
        self.index_group.load()
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1) first obtain an entity list from all docs
        2) check the lookup LRUCache to see if an entity has been recently processed
        3) if the cache misses, run a string similarity search upon the CachedIndexGroup
        :param docs:
        :return:
        """
        failed_docs = []
        entities = pydash.flatten([x.get_entities() for x in docs])
        if len(entities) > 0:
            for entity in entities:
                cache_missed_entities = self.lookup_cache.check_lookup_cache([entity])
                if len(cache_missed_entities) == 0:
                    continue
                else:
                    try:
                        mappings = self.index_group.search(
                            query=entity.match,
                            entity_class=entity.entity_class,
                            fuzzy=self.fuzzy,
                            top_n=self.top_n,
                            namespace=self.namespace(),
                            score_cutoff=self.score_cutoff,
                        )
                        for mapping in mappings:
                            entity.add_mapping(mapping)
                            self.lookup_cache.update_lookup_cache(entity, mapping)
                    except Exception:
                        doc = find_document_from_entity(docs, entity)
                        doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                        failed_docs.append(doc)

        return docs, failed_docs
