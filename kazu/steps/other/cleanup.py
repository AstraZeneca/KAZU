import traceback
from typing import List, Tuple, Protocol, Callable, Iterable
from kazu.data.data import Document, Entity, PROCESSING_EXCEPTION, Mapping, LinkRanks
from kazu.steps import Step

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
    def __init__(self, ranks_to_drop: Iterable[LinkRanks]):
        self.ranks_to_drop = set(ranks_to_drop)

    def __call__(self, mapping: Mapping) -> bool:
        return mapping.confidence in self.ranks_to_drop


class DropUnmappedEntityFilter:
    def __init__(self, from_ent_namespaces: Iterable[str]):
        self.from_ent_namespaces = set(from_ent_namespaces)

    def __call__(self, ent: Entity) -> bool:
        return ent.namespace in self.from_ent_namespaces and len(ent.mappings) == 0


class CleanupStep(Step):
    def __init__(self, cleanup_actions: List[CleanupAction]):
        self.cleanup_actions = cleanup_actions

    def __call__(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for cleanup_action in self.cleanup_actions:
                    cleanup_action.cleanup(doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)

        return docs, failed_docs
