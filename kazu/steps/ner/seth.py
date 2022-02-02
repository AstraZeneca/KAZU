import logging
import traceback
from typing import Optional, List, Tuple

from py4j.java_gateway import JavaGateway

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import BaseStep

logger = logging.getLogger(__name__)


class SethStep(BaseStep):
    entity_class = "GENERAL_VARIANT"

    def __init__(self, depends_on: Optional[List[str]]):
        super().__init__(depends_on)
        self.gateway = JavaGateway()
        self.seth = self.gateway.entry_point

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            for section in doc.sections:
                try:
                    mutation_lst = self.seth.findMutations(section.get_text())
                    entities = []
                    for mutation_dict in mutation_lst:
                        entities.append(
                            Entity.from_spans(
                                start=mutation_dict["start"],
                                end=mutation_dict["end"],
                                match=mutation_dict["match"],
                                entity_class=SethStep.entity_class,
                                metadata={
                                        "found_with": mutation_dict.get("found_with"),
                                        "protein_mutation": mutation_dict.get("protein_mutation"),
                                        "hgvs": mutation_dict["hgvs"],
                                    }
                                ),
                            )

                except Exception:
                    doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                    failed_docs.append(doc)
        return docs, failed_docs
