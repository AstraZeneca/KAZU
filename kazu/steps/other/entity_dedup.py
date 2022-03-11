import itertools
import logging
import traceback
from typing import List, Tuple, Optional

from kazu.data.data import Document, PROCESSING_EXCEPTION
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

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for section in doc.sections:
                    ents_to_consider = (x for x in section.entities if len(x.spans) == 1)
                    groups = itertools.groupby(
                        sorted(
                            ents_to_consider,
                            key=lambda x: (
                                x.start,
                                x.end,
                                self.namespace_preferred_order[x.namespace],
                            ),
                        ),
                        key=lambda x: (x.start, x.end),
                    )
                    for _, ent_iter in groups:
                        next_ent = next(ent_iter)
                        lowest_pref_order = self.namespace_preferred_order[next_ent.namespace]
                        for ent in ent_iter:
                            current_pref_order = self.namespace_preferred_order[ent.namespace]
                            if current_pref_order >= lowest_pref_order:
                                logger.debug("removing redundant entity: {}".format(ent))
                                section.entities.remove(ent)
                                lowest_pref_order = current_pref_order
            except Exception:
                message = f"doc failed: affected ids: {doc.idx}\n" + traceback.format_exc()
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
                docs.remove(doc)
        return docs, failed_docs
