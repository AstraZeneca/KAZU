import logging
import traceback
from typing import List, Tuple

import pydash

from azner.data.data import Document, PROCESSING_EXCEPTION
from azner.steps import BaseStep
from azner.utils.caching import CachedIndexGroup
from azner.utils.caching import EntityLinkingLookupCache
from azner.utils.utils import (
    filter_entities_with_ontology_mappings,
    find_document_from_entity,
)

logger = logging.getLogger(__name__)


class DictionaryEntityLinkingStep(BaseStep):
    """
    A simple dictionary lookup step, using RapidFuzz for string matching
    """

    def __init__(
        self,
        depends_on: List[str],
        index_group: CachedIndexGroup,
        process_all_entities: bool = False,
        lookup_cache_size: int = 5000,
        fuzzy: bool = True,
        top_n: int = 20,
    ):
        """

        :param depends_on:
        :param entity_class_to_ontology_mappings: Dict mapping entity class to ontology names
        :param ontology_dictionary_index: Dict of ontology name: DictionaryIndex
        :param process_all_entities: if false, will ignore any entities with mappings already associated
        :param lookup_cache_size: cache size to prevent repeated calls to the index
        """
        super().__init__(depends_on=depends_on)
        self.top_n = top_n
        self.fuzzy = fuzzy
        self.process_all_entities = process_all_entities
        self.index_group = index_group
        self.index_group.load()
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1) first obtain an entity list from all docs
        2) check the lookup LRUCache to see if it's been recently processed
        3) if the cache misses, run a string similarity search based upon DictionaryIndex
        :param docs:
        :return:
        """
        failed_docs = []
        entities = pydash.flatten([x.get_entities() for x in docs])
        if not self.process_all_entities:
            entities = filter_entities_with_ontology_mappings(entities)

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
                        )
                        for mapping in mappings:
                            entity.add_mapping(mapping)
                            self.lookup_cache.update_lookup_cache(entity, mapping)
                    except Exception:
                        doc = find_document_from_entity(docs, entity)
                        doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                        failed_docs.append(doc)

        return docs, failed_docs
