import logging
import traceback
from collections import defaultdict
from typing import List, Tuple, Optional, DefaultDict, Set, Dict

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity, Section
from kazu.steps import BaseStep

logger = logging.getLogger(__name__)


class MergeOverlappingEntsStep(BaseStep):
    """
    This step merges overlapping and nested entities. The final result should not allow any overlapped entities
    1) the longest Entity with at least one Mapping affixed should always be kept
    2) if no Mapping affixed, choose the next longest span with a Mapping
    3) in the case of equal length Entities, keep according to a prescribed entity class order. If the prescribed
        entity class order is also equal, the preferred entity is selected on the basis of the entity class name (
        alphabetically ordered). Since this final ordering is random, it's recommended to specify an
        ent_class_preferred_order in the config, which guarantees a consistent ordering.
    4) if no mappings are found, simply keep the longest entity
    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        ent_class_preferred_order: List[str],
    ):
        """

        :param depends_on:
        :param ent_class_preferred_order: order of namespaces to prefer. Any exactly overlapped entities are eliminated
        according to this ordering (first = higher priority). If an entity class is not specified, it's assumed to have
         a priority of 0 (a.k.a lowest)
        """

        super().__init__(depends_on)
        # store as dict for lookup speed
        self.ent_class_preferred_order = {
            namespace: i for i, namespace in enumerate(reversed(ent_class_preferred_order))
        }

    def prefer_longest_mapped_then_class(self, ents: Set[Entity]):
        ents_with_mappings = list(filter(lambda x: len(x.mappings) > 0, ents))
        if len(ents_with_mappings) == 0:
            # No mapped ents this location. just take longest
            preferred_ents = list(
                sorted(
                    ents,
                    key=lambda x: (
                        (x.end - x.start),
                        self.ent_class_preferred_order.get(x.entity_class, 0),
                        x.entity_class,
                    ),
                    reverse=True,
                )
            )
        else:
            # take longest mapped
            preferred_ents = list(
                sorted(
                    ents_with_mappings,
                    key=lambda x: (
                        (x.end - x.start),
                        self.ent_class_preferred_order.get(x.entity_class, 0),
                        x.entity_class,
                    ),
                    reverse=True,
                )
            )
        return preferred_ents[0], preferred_ents[1:]

    def filter_ents_across_class(self, ents: Dict[Tuple[int, int], Set[Entity]]) -> List[Entity]:
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
                        locations_overlapped = self.group_entities_by_location(section)
                        section.entities = self.filter_ents_across_class(locations_overlapped)

            except Exception:
                message = f"doc failed: affected ids: {doc.idx}\n" + traceback.format_exc()
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
                # docs.remove(doc)
        return docs, failed_docs

    def group_entities_by_location(self, section: Section) -> Dict[Tuple[int, int], Set[Entity]]:
        """
        in this context, a location is a span of text represented by a start and end char index tuple. A location
        represents all contiguous and non-contiguous entities that in someway overlap, even if not directly. E.g.

        A overlaps B but not C. B overlaps C.

        entities A, B and C are all considered to be part of the same location

        :param section:
        :return: dict of locations to Set[Entity]
        """
        locations_by_start = sorted(section.entities, key=lambda x: x.start)
        locations_overlapped: DefaultDict[Tuple[int, int], Set[Entity]] = defaultdict(set)
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

                ents_this_group = {entity}
                start = entity.start
                end = entity.end
            else:
                raise RuntimeError(
                    "Entities overlap in an undefined fashion. This should be impossible?"
                )
        else:
            locations_overlapped[(start, end)] = ents_this_group
        return dict(locations_overlapped)
