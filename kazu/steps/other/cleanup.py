import traceback
from typing import List, Tuple, Optional, Protocol

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep


class CleanupAction(Protocol):
    def __call__(self, doc: Document):
        raise NotImplementedError


class DropUnmappedEnts:
    def __init__(self, from_ent_namespaces: List[str]):
        self.from_ent_namespaces = set(from_ent_namespaces)

    def __call__(self, doc: Document):
        for section in doc.sections:
            section_entities = set(section.entities)
            drop_entities = set(
                ent
                for ent in section_entities
                if ent.namespace in self.from_ent_namespaces and len(ent.mappings) == 0
            )
            section_entities.difference_update(drop_entities)
            section.entities = list(section_entities)


class CleanupStep(BaseStep):
    def __init__(self, depends_on: Optional[List[str]], cleanup_actions: List[CleanupAction]):
        super().__init__(depends_on=depends_on)
        self.cleanup_actions = cleanup_actions

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                for cleanup_action in self.cleanup_actions:
                    cleanup_action(doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)

        return docs, failed_docs
