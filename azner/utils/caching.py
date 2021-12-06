from typing import List

from cachetools import LFUCache

from azner.data.data import Entity, Mapping
from azner.utils.utils import get_match_entity_class_hash


class EntityLinkingLookupCache:
    """
    A simple wrapper around LFUCache to reduce calls to expensive processes (e.g. bert)
    """

    def __init__(self, lookup_cache_size: int = 5000):
        self.lookup_cache = LFUCache(lookup_cache_size)

    def update_lookup_cache(self, entity: Entity, mapping: Mapping):
        hash_val = get_match_entity_class_hash(entity)
        if hash_val not in self.lookup_cache:
            self.lookup_cache[hash_val] = [mapping]
        else:
            cache_hit = self.lookup_cache[hash_val]
            self.lookup_cache[hash_val] = cache_hit + [mapping]

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
            maybe_mappings = self.lookup_cache.get(hash_val, None)
            if maybe_mappings is None:
                cache_misses.append(ent)
            else:
                for mapping in maybe_mappings:
                    ent.add_mapping(mapping)
        return cache_misses
