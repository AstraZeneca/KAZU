import logging
from collections import defaultdict
from typing import List, Set, Optional

from kazu.data.data import Document, SynonymTermWithMetrics
from kazu.steps.step import Step, document_batch_step
from kazu.utils.caching import EntityLinkingLookupCache
from kazu.utils.grouping import sort_then_group
from kazu.utils.link_index import DictionaryIndex

logger = logging.getLogger(__name__)


class DictionaryEntityLinkingStep(Step):
    """Uses :class:`kazu.utils.link_index.DictionaryIndex` to match entities to ontologies.

    Note, this is not an instance of :class:`kazu.steps.step.ParserDependentStep`, as
    this logic would duplicate the work of :class:`kazu.utils.link_index.DictionaryIndex`
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
        self.entity_class_to_indices = defaultdict(set)
        for index in indices:
            self.entity_class_to_indices[index.entity_class].add(index)
        self.top_n = top_n
        self.skip_ner_namespaces = skip_ner_namespaces if skip_ner_namespaces is not None else set()
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    @document_batch_step
    def __call__(self, docs: List[Document]) -> None:
        """
        logic of entity linker:

        1. first obtain an entity list from all docs
        2. check the lookup LRUCache to see if an entity has been recently processed
        3. if the cache misses, run a string similarity search using the configured :class:`kazu.utils.link_index.DictionaryIndex` 's

        :param docs:
        :return:
        """
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
