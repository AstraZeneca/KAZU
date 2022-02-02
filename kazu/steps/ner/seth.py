import logging
import os
import traceback
from typing import Optional, List, Tuple

from py4j.java_gateway import JavaGateway

from kazu.data.data import Document, PROCESSING_EXCEPTION, Entity
from kazu.steps import BaseStep

logger = logging.getLogger(__name__)


class SethStep(BaseStep):
    def __init__(self, depends_on: Optional[List[str]], entity_class: str, seth_fatjar_path: str):
        super().__init__(depends_on)
        if not os.path.exists(seth_fatjar_path):
            raise RuntimeError(f"required jar: {seth_fatjar_path} not found")

        self.gateway = JavaGateway.launch_gateway(classpath=seth_fatjar_path, die_on_exit=True)
        self.seth = self.gateway.jvm.com.astrazeneca.kazu.SethRunner()
        self.entity_class = entity_class

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
                                spans=[
                                    (
                                        int(mutation_dict["start"]),
                                        int(mutation_dict["end"]),
                                    )
                                ],
                                text=section.get_text(),
                                entity_class=self.entity_class,
                                namespace=self.namespace,
                                metadata={
                                    "found_with": mutation_dict.get("found_with"),
                                    "protein_mutation": mutation_dict.get("protein_mutation"),
                                    "hgvs": mutation_dict["hgvs"],
                                },
                            ),
                        )
                    section.entities.extend(entities)
                except Exception:
                    doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                    failed_docs.append(doc)
        return docs, failed_docs
