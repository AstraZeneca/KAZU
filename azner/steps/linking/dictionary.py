import logging
import traceback
from typing import List, Tuple, Dict

import pydash

from azner.data.data import Document, Mapping, PROCESSING_EXCEPTION, NAMESPACE
from azner.steps import BaseStep
from azner.utils.caching import EntityLinkingLookupCache
from azner.utils.dictionary_index import DictionaryIndex
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
        entity_class_to_ontology_mappings: Dict[str, str],
        ontology_dictionary_index: Dict[str, DictionaryIndex],
        process_all_entities: bool = False,
        lookup_cache_size: int = 5000,
    ):
        """

        :param depends_on:
        :param entity_class_to_ontology_mappings: Dict mapping entity class to ontology names
        :param ontology_dictionary_index: Dict of ontology name: DictionaryIndex
        :param process_all_entities: if false, will ignore any entities with mappings already associated
        :param lookup_cache_size: cache size to prevent repeated calls to the index
        """
        super().__init__(depends_on=depends_on)
        self.entity_class_to_ontology_mappings = entity_class_to_ontology_mappings
        self.process_all_entities = process_all_entities
        self.ontology_index_dict = ontology_dictionary_index
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
                        ontology_name = self.entity_class_to_ontology_mappings[entity.entity_class]
                        index = self.ontology_index_dict.get(ontology_name, None)
                        if index is not None:
                            metadata_df = index.search(entity.match)
                            for i, row in metadata_df.iterrows():
                                row_dict = row.to_dict()
                                ontology_id = row_dict.pop("iri")
                                row_dict[NAMESPACE] = self.namespace()
                                new_mapping = Mapping(
                                    source=ontology_name,
                                    idx=ontology_id,
                                    mapping_type="direct",
                                    metadata=row_dict,
                                )
                                entity.add_mapping(new_mapping)
                                self.lookup_cache.update_lookup_cache(entity, new_mapping)
                    except Exception:
                        doc = find_document_from_entity(docs, entity)
                        doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                        failed_docs.append(doc)

        return docs, failed_docs
