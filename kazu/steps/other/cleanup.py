import dataclasses
from typing import List, Protocol, Callable, Iterable, Optional
import urllib

from kazu.data.data import (
    Document,
    Entity,
    Mapping,
    StringMatchConfidence,
    DisambiguationConfidence,
    MentionConfidence,
)
from kazu.steps import Step, document_iterating_step

EntityFilterFn = Callable[[Entity], bool]
MappingFilterFn = Callable[[Mapping], bool]


class CleanupAction(Protocol):
    def cleanup(self, doc: Document):
        raise NotImplementedError


class MappingFilterCleanupAction:
    def __init__(self, filter_fns: List[MappingFilterFn]):
        self.filter_fns = filter_fns

    def cleanup(self, doc: Document):
        for entity in doc.get_entities():
            entity.mappings = {
                mapping
                for mapping in entity.mappings
                if not any(f(mapping) for f in self.filter_fns)
            }


class EntityFilterCleanupAction:
    def __init__(self, filter_fns: List[EntityFilterFn]):
        self.filter_fns = filter_fns

    def cleanup(self, doc: Document):
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
        from_ent_namespaces: Iterable[str],
        min_confidence_level: Optional[MentionConfidence] = MentionConfidence.PROBABLE,
    ):
        self.min_confidence_level = min_confidence_level
        self.from_ent_namespaces = set(from_ent_namespaces)

    def __call__(self, ent: Entity) -> bool:
        if self.min_confidence_level is None:
            return ent.namespace in self.from_ent_namespaces and len(ent.mappings) == 0
        else:
            return (
                ent.namespace in self.from_ent_namespaces
                and len(ent.mappings) == 0
                and ent.mention_confidence < self.min_confidence_level
            )


class StripMappingURIsAction:
    """Strip the IDs in :class:`kazu.data.data.Mapping` to just the final part of the URI.

    For example, this will turn http://purl.obolibrary.org/obo/MONDO_0004979 into just MONDO_004979.

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

    def cleanup(self, doc: Document):
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


class CleanupStep(Step):
    def __init__(self, cleanup_actions: List[CleanupAction]):
        self.cleanup_actions = cleanup_actions

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for cleanup_action in self.cleanup_actions:
            cleanup_action.cleanup(doc)
