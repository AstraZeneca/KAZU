import logging
import traceback
from typing import List, Tuple, Dict, Set, Optional

from kazu.data.data import Document, PROCESSING_EXCEPTION, SynonymTermWithMetrics
from kazu.steps import Step
from kazu.utils.caching import EntityLinkingLookupCache
from kazu.utils.grouping import sort_then_group
from kazu.utils.link_index import DictionaryIndex
from kazu.utils.utils import find_document_from_entity

logger = logging.getLogger(__name__)


class DictionaryEntityLinkingStep(Step):
    """
    Uses :class:`kazu.utils.link_index.DictionaryIndex` to match entities to ontologies.
    """

    def __init__(
        self,
        indices: List[DictionaryIndex],
        lookup_cache_size: int = 5000,
        top_n: int = 20,
        skip_ner_namespaces: Optional[Set[str]] = None,
    ):
        """

        :param indices: indices to query
        :param lookup_cache_size: the size of the Least Recently Used lookup cache to maintain
        :param top_n: keep the top_n results for the query (passed to :class:`kazu.utils.link_index.DictionaryIndex`)
        :param skip_ner_namespaces: set of NER-step namespaces -- linking will be skipped for entities generated by
            these namespaces
        """
        self.entity_class_to_indices: Dict[str, Set[DictionaryIndex]] = {}
        self.top_n = top_n
        self.skip_ner_namespaces = skip_ner_namespaces if skip_ner_namespaces is not None else set()
        self.indices = indices
        self.load_or_build_caches()
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    def load_or_build_caches(self):
        for index in self.indices:
            index.load_or_build_cache()
            indices_for_current_class = self.entity_class_to_indices.setdefault(
                index.parser.entity_class, set()
            )
            indices_for_current_class.add(index)

    def __call__(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1. first obtain an entity list from all docs
        2. check the lookup LRUCache to see if an entity has been recently processed
        3. if the cache misses, run a string similarity search using the configured :class:`kazu.utils.link_index.DictionaryIndex` 's

        :param docs:
        :return:
        """
        failed_docs: List[Document] = []
        entities = (
            ent
            for doc in docs
            for ent in doc.get_entities()
            if ent.namespace not in self.skip_ner_namespaces
        )
        ents_by_match_and_class = {
            k: list(v) for k, v in sort_then_group(entities, lambda x: (x.match, x.entity_class))
        }
        if len(ents_by_match_and_class) > 0:
            for ent_match_and_class, ents_this_match in ents_by_match_and_class.items():

                cache_missed_entities = self.lookup_cache.check_lookup_cache(ents_this_match)
                if len(cache_missed_entities) == 0:
                    continue
                else:
                    try:
                        indices_to_search = self.entity_class_to_indices.get(ent_match_and_class[1])
                        if indices_to_search:
                            terms: List[SynonymTermWithMetrics] = []
                            for index in indices_to_search:
                                terms.extend(index.search(ent_match_and_class[0], self.top_n))

                            for ent in ents_this_match:
                                ent.update_terms(terms)

                            self.lookup_cache.update_terms_lookup_cache(
                                entity=next(iter(ents_this_match)), terms=terms
                            )

                    except Exception:
                        failed_docs_set: Set[Document] = set()
                        for ent in ents_this_match:
                            doc = find_document_from_entity(docs, ent)
                            doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                            failed_docs_set.add(doc)
                        failed_docs.extend(failed_docs_set)

        return docs, failed_docs
