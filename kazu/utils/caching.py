import copy
import logging
from typing import Iterable, List, Set

from cachetools import LFUCache

from kazu.data.data import Entity, Hit
from kazu.data.data import Mapping
from kazu.utils.utils import get_match_entity_class_hash

logger = logging.getLogger(__name__)


class EntityLinkingLookupCache:
    """
    A simple wrapper around LFUCache to reduce calls to expensive processes (e.g. bert)
    """

    def __init__(self, lookup_cache_size: int = 5000):
        self.mappings_lookup_cache: LFUCache = LFUCache(lookup_cache_size)
        self.hits_lookup_cache: LFUCache = LFUCache(lookup_cache_size)

    def update_mappings_lookup_cache(self, entity: Entity, mappings: List[Mapping]):
        hash_val = get_match_entity_class_hash(entity)
        cache_hit = self.mappings_lookup_cache.get(hash_val)
        if cache_hit is None:
            self.mappings_lookup_cache[hash_val] = mappings

    def update_hits_lookup_cache(self, entity: Entity, hits: Set[Hit]):
        hash_val = get_match_entity_class_hash(entity)
        cache_hit = self.hits_lookup_cache.get(hash_val)
        if cache_hit is None:
            self.hits_lookup_cache[hash_val] = hits

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
            mappings_cache_hits = self.mappings_lookup_cache.get(hash_val, [])
            hits_cache_hits = self.hits_lookup_cache.get(hash_val, [])
            if not mappings_cache_hits and not hits_cache_hits:
                cache_misses.append(ent)
            else:
                for mapping in mappings_cache_hits:
                    ent.mappings.add(copy.deepcopy(mapping))
                for hit in hits_cache_hits:
                    ent.update_hits([copy.deepcopy(hit)])
        return cache_misses
