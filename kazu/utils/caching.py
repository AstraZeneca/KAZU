import logging
import os
import sys
import tempfile
from collections.abc import Iterable, Callable
from pathlib import Path
from typing import Any, TypeVar, Protocol, Optional, Union

from cachetools import LFUCache
from diskcache import Cache
from kazu.data import Entity, CandidatesToMetrics
from kazu.utils.utils import get_match_entity_class_hash

logger = logging.getLogger(__name__)
kazu_model_pack_dir = os.getenv("KAZU_MODEL_PACK")
KAZU_DISK_CACHE_NAME = "kazu_disk_cache"


Ret = TypeVar("Ret", covariant=True)
"""A TypeVar to represent a return value.

Used in :class:`~.Memoization` to encode that a memorized function
returns the same type as the original function it 'memoized'.
"""


class Memoization(Protocol[Ret]):
    def __cache_key__(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Ret:
        raise NotImplementedError


class CacheProtocol(Protocol):
    def memoize(
        self,
        name: Optional[str] = None,
        typed: bool = False,
        expire: Optional[float] = None,
        tag: Optional[str] = None,
        ignore: set[Union[str, int]] = set(),
    ) -> Callable[[Callable[..., Ret]], Memoization[Ret]]:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def delete(self, key: Any) -> bool:
        raise NotImplementedError

    def __enter__(self) -> Cache:
        raise NotImplementedError

    def __exit__(self, *exception: Exception) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Any:
        raise NotImplementedError


kazu_disk_cache: CacheProtocol
"""We use the :class:`diskcache.Cache` concept to cache expensive to produce resources
to disk. Methods and functions can be decorated with kazu_disk_cache.memoize() to use
this feature. Note, when used with a class method, the default behaviour of this
function is to generate a key based on the constructor arguments of the class instance.
Since these can be large (e.g. OntologyParser), we sometimes use the ignore argument to
override this behaviour.

e.g.

.. code-block:: python

    @kazu_disk_cache.memoize(ignore={0})
    def method_of_class_with_lots_of_args(self):
        ...
"""
if kazu_model_pack_dir is None:
    kazu_disk_cache_path_str = tempfile.mkdtemp(suffix=KAZU_DISK_CACHE_NAME)
    logger.warning(
        "KAZU_MODEL_PACK env variable not set. Using %s for caching. "
        "This will not be reused if the process is exited",
        kazu_disk_cache_path_str,
    )
else:
    kazu_disk_cache_path_str = str(Path(kazu_model_pack_dir).joinpath(KAZU_DISK_CACHE_NAME))
    logger.info("using disk cache at %s", kazu_disk_cache_path_str)

kazu_disk_cache = Cache(
    directory=kazu_disk_cache_path_str,
    eviction_policy="none",
    size_limit=sys.maxsize,
    cull_limit=0,
)


class EntityLinkingLookupCache:
    """A simple wrapper around LFUCache to reduce calls to expensive processes (e.g.
    bert)"""

    def __init__(self, lookup_cache_size: int = 5000):
        self.candidate_lookup_cache: LFUCache[int, CandidatesToMetrics] = LFUCache(
            lookup_cache_size
        )

    def update_candidates_lookup_cache(
        self, entity: Entity, candidates: CandidatesToMetrics
    ) -> None:
        hash_val = get_match_entity_class_hash(entity)
        cache_hit = self.candidate_lookup_cache.get(hash_val)
        if cache_hit is None:
            self.candidate_lookup_cache[hash_val] = candidates

    def check_lookup_cache(self, entities: Iterable[Entity]) -> list[Entity]:
        """Checks the cache for linking candidates. If relevant candidates are found for
        an entity, update it accordingly. If not return as a list of cache misses (e.g.
        for further processing)

        :param entities:
        :return:
        """
        cache_misses = []
        for ent in entities:
            hash_val = get_match_entity_class_hash(ent)
            candidates_from_cache = self.candidate_lookup_cache.get(hash_val)
            if not candidates_from_cache:
                cache_misses.append(ent)
            else:
                ent.add_or_update_linking_candidates(candidates_from_cache)
        return cache_misses
