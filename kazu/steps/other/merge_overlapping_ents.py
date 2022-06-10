import logging
import traceback
from collections import defaultdict
from typing import List, Tuple, Optional, DefaultDict, Set

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import BaseStep

logger = logging.getLogger(__name__)


class MergeOverlappingEntsStep(BaseStep):
    """
    This step deals with overlapping entities.
    the strategy is as follows:

    1) the final result should not allow any overlapped entities
    2) the longest span should always be kept, assuming a Mapping is attached
    3) if no Mapping attached, choose the next longest span with a Mapping
    4) in the case of equal length, Mapped spans, keep according to a prescribed order


    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        namespace_preferred_order: List[str],
        ent_class_preferred_order: List[str],
    ):
        """

        :param depends_on:
        :param namespace_preferred_order: order of namespaces to prefer. Any exactly overlapped entities are eliminated
        according to this ordering (first = higher priority).
        """

        super().__init__(depends_on)
        # store as dict for lookup speed
        self.ent_class_preferred_order = {
            namespace: i for i, namespace in enumerate(reversed(ent_class_preferred_order))
        }
        self.namespace_preferred_order = {
            namespace: i for i, namespace in enumerate(reversed(namespace_preferred_order))
        }

    def prefer_longest_mapped_then_class(self, ents: Set[Entity]):
        ents_with_mappings = list(filter(lambda x: len(x.mappings) > 0, ents))
        if len(ents_with_mappings) == 0:
            # No mapped ents this location. just take longest
            arranged = list(
                sorted(
                    ents,
                    key=lambda x: (
                        (x.end - x.start),
                        self.ent_class_preferred_order.get(x.entity_class, 0),
                    ),
                    reverse=True,
                )
            )
        else:
            # take longest mapped
            arranged = list(
                sorted(
                    ents_with_mappings,
                    key=lambda x: (
                        (x.end - x.start),
                        self.ent_class_preferred_order.get(x.entity_class, 0),
                    ),
                    reverse=True,
                )
            )
        return arranged[0], arranged[1:]

    def filter_ents_across_class(self, ents: DefaultDict[Tuple[int, int], Set[Entity]]):
        """
        choose the best entities for a given location
        :param ents:
        :return:
        """
        to_keep = []
        for location, ents_set in ents.items():
            if len(ents_set) > 1:
                to_keep_this_location, to_drop = self.prefer_longest_mapped_then_class(ents_set)
                to_keep.append(to_keep_this_location)
                logger.debug("dropping ents %s", to_drop)
            else:
                to_keep.extend(list(ents_set))
        return to_keep

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        algorithm:
        find all longest non-overlapping spans -> locations to consider
        for each location, find all ents exactly matching and within
        select class of span from entity label
        prefer dictionary hits within this span
        keep longest dictionary hit, eject all others
        if no dictionary hit found, keep longest span regardless of namespace


        :param docs:
        :return:
        """

        failed_docs = []
        for doc in docs:
            try:
                for section in doc.sections:
                    if len(section.entities) > 0:
                        locations_by_start = sorted(section.entities, key=lambda x: x.start)
                        locations_overlapped: DefaultDict[
                            Tuple[int, int], Set[Entity]
                        ] = defaultdict(set)
                        ents_this_group = set()
                        start = locations_by_start[0].start
                        end = locations_by_start[0].end
                        for entity in locations_by_start:
                            if entity.start >= start and entity.end <= end:
                                ents_this_group.add(entity)
                            elif entity.start >= start and entity.start < end and entity.end > end:
                                end = entity.end
                                ents_this_group.add(entity)
                            elif entity.start >= end:
                                locations_overlapped[(start, end)] = ents_this_group

                                ents_this_group = set([entity])
                                start = entity.start
                                end = entity.end
                            else:
                                raise RuntimeError("SHOULD NOT HAPPEN!")
                        else:
                            locations_overlapped[(start, end)] = ents_this_group

                        section.entities = self.filter_ents_across_class(locations_overlapped)

            except Exception:
                message = f"doc failed: affected ids: {doc.idx}\n" + traceback.format_exc()
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
                # docs.remove(doc)
        return docs, failed_docs
