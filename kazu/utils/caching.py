import copy
import logging
from typing import Iterable, List

from cachetools import LFUCache
from kazu.data.data import Entity, SynonymTermWithMetrics
from kazu.utils.utils import get_match_entity_class_hash

logger = logging.getLogger(__name__)


class EntityLinkingLookupCache:
    """
    A simple wrapper around LFUCache to reduce calls to expensive processes (e.g. bert)
    """

    def __init__(self, lookup_cache_size: int = 5000):
        self.terms_lookup_cache: LFUCache = LFUCache(lookup_cache_size)

    def update_terms_lookup_cache(self, entity: Entity, terms: Iterable[SynonymTermWithMetrics]):
        hash_val = get_match_entity_class_hash(entity)
        cache_hit = self.terms_lookup_cache.get(hash_val)
        if cache_hit is None:
            self.terms_lookup_cache[hash_val] = set(terms)

    def check_lookup_cache(self, entities: Iterable[Entity]) -> List[Entity]:
        """
        checks the cache for mappings and hits. If relevant mappings are found for an entity, update its mappings
        accordingly. If not return as a list of cache misses (e.g. for further processing)

        :param entities:
        :return:
        """
        cache_misses = []
        for ent in entities:
            hash_val = get_match_entity_class_hash(ent)
            hits_cache_hits = self.terms_lookup_cache.get(hash_val, set())
            if not hits_cache_hits:
                cache_misses.append(ent)
            else:
                ent.update_terms(copy.deepcopy(hits_cache_hits))
        return cache_misses
