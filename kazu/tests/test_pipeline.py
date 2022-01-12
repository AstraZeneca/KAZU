import os
import traceback
from typing import List, Tuple
import tempfile
from kazu.steps import BaseStep
from kazu.data.data import Document, PROCESSING_EXCEPTION, SimpleDocument
from kazu.pipeline.pipeline import FailedDocsFileHandler, Pipeline


class BrokenStep(BaseStep):
    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                this_causes_an_exception = {{"cant": "hash a dict"}}
                this_causes_an_exception.clear()
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return [], failed_docs


def test_pipeline_error_handling():
    with tempfile.TemporaryDirectory() as f:
        step = BrokenStep([])
        pipeline = Pipeline([step], [FailedDocsFileHandler(f)])

        docs = [SimpleDocument("hello") for _ in range(5)]
        pipeline(docs)
        error_files = os.listdir(os.path.join(f, step.namespace()))
        # should be two files per doc - one with exception, one with doc contents
        assert len(error_files) == 2 * len(docs)

        # should flush docs between calls
        assert len(pipeline.failed_docs) == 0

        more_docs = [SimpleDocument("hello") for _ in range(5)]

        pipeline(more_docs)
        error_files = os.listdir(os.path.join(f, step.namespace()))
        # should be two files per doc - one with exception, one with doc contents
        assert len(error_files) == 2 * (len(docs) + len(more_docs))
