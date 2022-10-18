import traceback
from typing import List, Tuple, Optional, Protocol, Callable, Iterable
from kazu.data.data import Document, Entity, PROCESSING_EXCEPTION
from kazu.steps import BaseStep

EntityFilterFn = Callable[[Entity], bool]


class CleanupAction(Protocol):
    def cleanup(self, doc: Document):
        raise NotImplementedError


class EntityFilterFnProvider(Protocol):
    def filter_fns(self) -> List[EntityFilterFn]:
        raise NotImplementedError


class EntityFilterCleanupAction:
    def __init__(self, filter_fns: List[EntityFilterFn]):
        self.filter_fns = filter_fns

    def cleanup(self, doc: Document):
        for section in doc.sections:
            section.entities = [
                entity for entity in section.entities if not any(f(entity) for f in self.filter_fns)
            ]


class DropUnmappedEntityFilter:
    def __init__(self, from_ent_namespaces: Iterable[str]):
        self.from_ent_namespaces = set(from_ent_namespaces)

    def __call__(self, ent: Entity) -> bool:
        return ent.namespace in self.from_ent_namespaces and len(ent.mappings) == 0


class CleanupStep(BaseStep):
    def __init__(self, depends_on: Optional[List[str]], cleanup_actions: List[CleanupAction]):
        super().__init__(depends_on=depends_on)
        self.cleanup_actions = cleanup_actions

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for cleanup_action in self.cleanup_actions:
                    cleanup_action.cleanup(doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)

        return docs, failed_docs
