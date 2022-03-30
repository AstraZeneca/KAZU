import copy
import itertools
import logging
import traceback
from typing import List, Tuple, Optional

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import BaseStep

logger = logging.getLogger(__name__)


class NerDeDuplicationStep(BaseStep):
    """
    With multiple NER pipelines running, it's expected that the same entity will be picked up over the same span
    of text multiple times. This step removes redundant Entity instances by setting a priority order over NER step
    namespaces. Only considers contiguous entities (e.g. where len(Entity.spans) == 1), as the semantics of
    overlapping of non-contiguous entities are undefined
    """

    def __init__(self, depends_on: Optional[List[str]], namespace_preferred_order: List[str]):
        """

        :param depends_on:
        :param namespace_preferred_order: order of namespaces to prefer. Any exactly overlapped entities are eliminated
        according to this ordering (first = higher priority).
        """

        super().__init__(depends_on)
        # store as dict for lookup speed
        self.namespace_preferred_order = {
            namespace: i for i, namespace in enumerate(namespace_preferred_order)
        }

    def filter_ents(self, ents: List[Entity]):
        """
        choose the best entities for a given location
        :param ents:
        :return:
        """
        remove_lst, keep_lst = [], []
        detected_classes = {ent.entity_class for ent in ents}
        other_namespaces_that_found_this_ent = set()
        for ent in ents:
            # if ent.metadata.get(USE_EXACT_MATCHING,False) and len(ent.mappings)==0:
            #     remove_lst.append(ent)
            #     other_namespaces_that_found_this_ent.add(ent.namespace)
            if ent.entity_class == "entity" and len(detected_classes) > 1:
                other_namespaces_that_found_this_ent.add(ent.namespace)
                remove_lst.append(ent)
            else:
                keep_lst.append(ent)

        for ent in keep_lst:
            ent.metadata["also_detected_by"] = copy.copy(other_namespaces_that_found_this_ent)
        return keep_lst, remove_lst

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for section in doc.sections:
                    ents_to_consider = (x for x in section.entities if len(x.spans) == 1)
                    groups = itertools.groupby(
                        sorted(
                            ents_to_consider,
                            key=lambda x: (x.start, x.end, x.entity_class, x.namespace),
                        ),
                        key=lambda x: (x.start, x.end, x.entity_class),
                    )
                    for _, ent_iter in groups:
                        ents_to_keep, ents_to_remove = self.filter_ents(list(ent_iter))

                        for other_ent in ents_to_remove:
                            section.entities.remove(other_ent)
                        for ent in ents_to_keep[1:]:
                            section.entities.remove(ent)

            except Exception:
                message = f"doc failed: affected ids: {doc.idx}\n" + traceback.format_exc()
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
                # docs.remove(doc)
        return docs, failed_docs
