import logging
import traceback
from collections import defaultdict
from typing import List, Tuple, Optional, DefaultDict, Set, Dict

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import Step

logger = logging.getLogger(__name__)


class MergeOverlappingEntsStep(Step):
    """
    This step merges overlapping and nested entities. The final result should not allow any overlapped entities
    see algorithm description below
    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        ent_class_preferred_order: List[str],
        ignore_non_contiguous: bool = True,
    ):
        """

        The algorithm for selecting an entity span is as follows:

        1. group entities by location

           In this context, a location is a span of text represented by a start and end char index tuple. A location
           represents all contiguous and non-contiguous entities that in some way overlap, even if not directly. E.g.

           A overlaps B but not C. B overlaps C.

           entities A, B and C are all considered to be part of the same location

        2. sort entities within each location, picking the best according to the following sort logic:

           1. prefer entities with mappings
           2. prefer longest spans
           3. prefer entities as configured by ent_class_preferred_order (see param description below)
           4. If the proscribed entity class order is also equal, the preferred entity is selected on the basis of
              the entity class name (reverse alphabetically ordered). Warning: This last sort criteria is arbitrary


        :param depends_on:
        :param ent_class_preferred_order: order of namespaces to prefer. Any partially overlapped entities are
            eliminated according to this ordering (first = higher priority). If an entity class is not specified, it's
            assumed to have a priority of 0 (a.k.a lowest)
        :param ignore_non_contiguous: should non-contiguous entities be excluded from the merge process?
        """

        super().__init__(depends_on)
        # store as dict for lookup speed
        self.ignore_non_contiguous = ignore_non_contiguous
        self.ent_class_preferred_order = {
            namespace: i for i, namespace in enumerate(reversed(ent_class_preferred_order))
        }

    def select_preferred_entity(self, ents: Set[Entity]) -> Tuple[Entity, List[Entity]]:
        """
        :param ents:
        :return: tuple of Entity<preferred> ,List[Entity]<other entities at this location>
        """
        preferred_ents = sorted(
            ents,
            key=lambda x: (
                len(x.mappings) > 0,
                (x.end - x.start),
                self.ent_class_preferred_order.get(x.entity_class, 0),
                x.entity_class,
            ),
            reverse=True,
        )
        return preferred_ents[0], preferred_ents[1:]

    def filter_ents_across_class(self, ents: Dict[Tuple[int, int], Set[Entity]]) -> List[Entity]:
        """
        choose the best entities per location

        :param ents:
        :return:
        """
        to_keep = []
        for location, ents_set in ents.items():
            if len(ents_set) > 1:
                to_keep_this_location, to_drop = self.select_preferred_entity(ents_set)
                to_keep.append(to_keep_this_location)
                logger.debug("dropping ents %s", to_drop)
            else:
                to_keep.extend(ents_set)
        return to_keep

    def __call__(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for section in doc.sections:
                    if len(section.entities) > 0:
                        ents_to_merge, non_contig_ents = [], []
                        if self.ignore_non_contiguous:
                            for ent in section.entities:
                                if len(ent.spans) == 1:
                                    ents_to_merge.append(ent)
                                else:
                                    non_contig_ents.append(ent)
                        else:
                            ents_to_merge = section.entities

                        locations_overlapped = self.group_entities_by_location(ents_to_merge)
                        section.entities = self.filter_ents_across_class(locations_overlapped)
                        section.entities.extend(non_contig_ents)

            except Exception:
                message = f"doc failed: affected ids: {doc.idx}\n" + traceback.format_exc()
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
        return docs, failed_docs

    def group_entities_by_location(
        self, entities: List[Entity]
    ) -> Dict[Tuple[int, int], Set[Entity]]:
        """

        :param entities:
        :return: dict of locations to Set[Entity]
        """
        if len(entities) == 0:
            return {}
        locations_by_start = sorted(entities, key=lambda x: x.start)
        locations_overlapped: DefaultDict[Tuple[int, int], Set[Entity]] = defaultdict(set)
        ents_this_group = set()
        start = locations_by_start[0].start
        end = locations_by_start[0].end
        for entity in locations_by_start:
            if start <= entity.start < end:
                # we're still in the same location
                if entity.end > end:
                    # expand the location to new end
                    end = entity.end
                ents_this_group.add(entity)
            else:
                assert (
                    entity.start >= end
                ), "Entities overlap in an undefined fashion. This should be impossible?"
                # we've gone beyond the 'current' location, 'submit' it to locations_overlapped
                locations_overlapped[(start, end)] = ents_this_group
                # set up new location
                ents_this_group = {entity}
                start = entity.start
                end = entity.end

        locations_overlapped[(start, end)] = ents_this_group
        return dict(locations_overlapped)
