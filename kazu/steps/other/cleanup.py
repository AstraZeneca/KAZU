from typing import List, Protocol, Callable, Iterable
from kazu.data.data import (
    Document,
    Entity,
    Mapping,
    StringMatchConfidence,
    DisambiguationConfidence,
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
    def __init__(self, from_ent_namespaces: Iterable[str]):
        self.from_ent_namespaces = set(from_ent_namespaces)

    def __call__(self, ent: Entity) -> bool:
        return ent.namespace in self.from_ent_namespaces and len(ent.mappings) == 0


class CleanupStep(Step):
    def __init__(self, cleanup_actions: List[CleanupAction]):
        self.cleanup_actions = cleanup_actions

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for cleanup_action in self.cleanup_actions:
            cleanup_action.cleanup(doc)
