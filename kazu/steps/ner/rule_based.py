import traceback
from typing import List, Tuple

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep


class RuleBasedNerAndLinkingStep(BaseStep):
    """
    A wrapper for the explosion ontology-based entity matcher and linker
    """

    def __init__(
        self,
        depends_on: List[str],
    ):
        super().__init__(depends_on=depends_on)

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        # this may be moved inside a loop over the docs
        try:
            pass
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)

        # make sure we're not returning all docs as successful if some have failed
        return docs, failed_docs
