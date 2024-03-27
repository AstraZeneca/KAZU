import dataclasses
import logging
import urllib
from collections.abc import Iterable, Callable
from typing import Protocol, Optional

from kazu.data import (
    Document,
    Entity,
    Mapping,
    StringMatchConfidence,
    DisambiguationConfidence,
    MentionConfidence,
    KazuConfigurationError,
)
from kazu.steps import Step, document_iterating_step
from kazu.utils.grouping import sort_then_group

EntityFilterFn = Callable[[Entity], bool]
MappingFilterFn = Callable[[Mapping], bool]

logger = logging.getLogger(__name__)


class CleanupAction(Protocol):
    def cleanup(self, doc: Document) -> None:
        raise NotImplementedError


class MappingFilterCleanupAction:
    def __init__(self, filter_fns: list[MappingFilterFn]):
        self.filter_fns = filter_fns

    def cleanup(self, doc: Document) -> None:
        for entity in doc.get_entities():
            entity.mappings = {
                mapping
                for mapping in entity.mappings
                if not any(f(mapping) for f in self.filter_fns)
            }


class EntityFilterCleanupAction:
    def __init__(self, filter_fns: list[EntityFilterFn]):
        self.filter_fns = filter_fns

    def cleanup(self, doc: Document) -> None:
        for section in doc.sections:
            section.entities = [
                entity for entity in section.entities if not any(f(entity) for f in self.filter_fns)
            ]


class DropMappingsByConfidenceMappingFilter:
    def __init__(
        self,
        string_match_ranks_to_drop: Iterable[StringMatchConfidence],
        disambiguation_ranks_to_drop: Iterable[DisambiguationConfidence],
    ):
        self.string_match_ranks_to_drop = set(string_match_ranks_to_drop)
        self.disambiguation_ranks_to_drop = set(disambiguation_ranks_to_drop)

    def __call__(self, mapping: Mapping) -> bool:

        return (
            mapping.string_match_confidence in self.string_match_ranks_to_drop
            or mapping.disambiguation_confidence in self.disambiguation_ranks_to_drop
        )


class DropUnmappedEntityFilter:
    def __init__(
        self,
        from_ent_namespaces: Optional[Iterable[str]] = None,
        min_confidence_level: Optional[MentionConfidence] = MentionConfidence.PROBABLE,
    ):
        self.min_confidence_level = min_confidence_level
        self.from_ent_namespaces = None if from_ent_namespaces is None else set(from_ent_namespaces)

    def __call__(self, ent: Entity) -> bool:
        relevant_namespace = (
            self.from_ent_namespaces is None or ent.namespace in self.from_ent_namespaces
        )

        if self.min_confidence_level is None:
            return relevant_namespace and len(ent.mappings) == 0
        else:
            return (
                relevant_namespace
                and len(ent.mappings) == 0
                and ent.mention_confidence < self.min_confidence_level
            )


class LinkingCandidateRemovalCleanupAction:
    def __init__(self):
        pass

    def cleanup(self, doc: Document) -> None:
        for section in doc.sections:
            for ent in section.entities:
                ent.linking_candidates.clear()


class StripMappingURIsAction:
    """Strip the IDs in :class:`kazu.data.Mapping` to just the final part of the URI.

    For example, this will turn
    http://purl.obolibrary.org/obo/MONDO_0004979
    into just MONDO_004979.

    If you don't want URI stripping at all, don't use this Action as part of the CleanupStep/in the pipeline.
    """

    def __init__(self, parsers_to_strip: Optional[Iterable[str]] = None):
        """

        :param parsers_to_strip: if you only want to strip URIs for some parsers and not others,
            provide the parsers to strip here. Otherwise, all parsers will have their IDs stripped.
            This prevents having to keep the full list of parsers in sync here.
        """
        self.parsers_to_strip = parsers_to_strip

    @staticmethod
    def _strip_uri(idx):
        url = urllib.parse.urlparse(idx)
        if url.scheme == "":
            # not a url
            new_idx = idx
        else:
            new_idx = url.path.split("/")[-1]
        return new_idx

    def cleanup(self, doc: Document) -> None:
        for entity in doc.get_entities():
            new_mappings = set()
            for mapping in entity.mappings:
                if (
                    self.parsers_to_strip is not None
                    and mapping.parser_name not in self.parsers_to_strip
                ):
                    new_mappings.add(mapping)
                else:
                    new_mappings.add(dataclasses.replace(mapping, idx=self._strip_uri(mapping.idx)))
            entity.mappings = new_mappings


class DropMappingsByParserNameRankAction(CleanupAction):
    """Removes instances of :class:`.Mapping` based upon some preferential order of
    parsers.

    Useful if you want to filter results based upon some predefined hierarchy of importance,
    for entity classes mapping to multiple parsers. For instance, you may prefer Meddra
    entities over Mondo ones, but will accept Mondo ones if Meddra mappings aren't
    available.

    .. caution::
       To ensure this class is configured correctly, ensure that all
       the parsers you intend to use with it have populated the metadata
       database first. See :meth:`~.OntologyParser.populate_databases`\\.
    """

    def __init__(
        self,
        entity_class_to_parser_name_rank: dict[str, list[str]],
    ):
        """

        :param entity_class_to_parser_name_rank: For a given entity class, only retain the mappings from the first parser
            that an entity has mappings for, based on list ordering (first is preferred).
        """
        self.entity_class_to_parser_name_rank = entity_class_to_parser_name_rank

    def cleanup(self, doc: Document) -> None:
        for entity in doc.get_entities():
            ranks_to_consider = self.entity_class_to_parser_name_rank.get(entity.entity_class)
            if ranks_to_consider is not None:
                try:
                    for _, mappings in sort_then_group(
                        entity.mappings, key_func=lambda x: ranks_to_consider.index(x.parser_name)
                    ):
                        entity.mappings = set(mappings)
                        # only consider the top rank
                        break
                except IndexError:
                    raise KazuConfigurationError(
                        f"Tried to cleanup mappings for {entity}, but at least one mapping has a parser_name that has not been configured with a rank. Mappings: {entity.mappings}"
                    )


class CleanupStep(Step):
    def __init__(self, cleanup_actions: list[CleanupAction]):
        self.cleanup_actions = cleanup_actions

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for cleanup_action in self.cleanup_actions:
            cleanup_action.cleanup(doc)
