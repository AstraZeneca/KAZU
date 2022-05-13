import copy
import itertools
import logging
import traceback
from typing import List, Tuple

import pydash

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.utils.caching import CachedIndexGroup, DictionaryIndexCacheManager
from kazu.utils.caching import EntityLinkingLookupCache
from kazu.utils.utils import (
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
        top_n: int = 20,
    ):
        """

        :param depends_on:
        :param index_group: A CachedIndexGroup constructed with List[DictionaryIndexCacheManager]
        :param lookup_cache_size: the size of the Least Recently Used lookup cache to maintain
        :param top_n: keep the top_n hits of the query
        """
        super().__init__(depends_on=depends_on)
        if not all(
            [isinstance(x, DictionaryIndexCacheManager) for x in index_group.cache_managers]
        ):
            raise RuntimeError(
                "The CachedIndexGroup must be configured with an DictionaryIndexCacheManager to work"
                "correctly with the DictionaryEntityLinkingStep"
            )
        self.top_n = top_n
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
        ents_by_match_and_class = {
            k: list(v)
            for k, v in itertools.groupby(
                sorted(
                    entities,
                    key=lambda x: (
                        x.match,
                        x.entity_class,
                    ),
                ),
                key=lambda x: (
                    x.match,
                    x.entity_class,
                ),
            )
        }
        if len(ents_by_match_and_class) > 0:
            for ent_match_and_class, ents_this_match in ents_by_match_and_class.items():

                cache_missed_entities = self.lookup_cache.check_lookup_cache(ents_this_match)
                if len(cache_missed_entities) == 0:
                    continue
                else:
                    try:
                        hits = list(
                            self.index_group.search(
                                query=ent_match_and_class[0],
                                entity_class=ent_match_and_class[1],
                                top_n=self.top_n,
                                namespace=self.namespace(),
                            )
                        )
                        for ent in ents_this_match:
                            ent.hits.extend(copy.deepcopy(hits))
                        self.lookup_cache.update_hits_lookup_cache(ents_this_match[0], hits)

                    except Exception:
                        failed_docs_set = set()
                        for ent in ents_this_match:
                            doc = find_document_from_entity(docs, ent)
                            doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                            failed_docs_set.add(doc)
                        failed_docs.extend(list(failed_docs_set))

        return docs, failed_docs
