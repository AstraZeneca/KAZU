import logging
import traceback
from typing import List, Optional, Tuple

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.steps.linking.post_processing.strategy_runner import StrategyRunner

logger = logging.getLogger(__name__)


class MappingStep(BaseStep):
    """
    A wrapper for :class:`.StrategyRunner`, so it can be used in a pipeline.
    """

    def __init__(self, depends_on: Optional[List[str]], strategy_runner: StrategyRunner):
        """

        :param depends_on:
        """
        super().__init__(depends_on)
        self.strategy_runner = strategy_runner

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []
        for doc in docs:
            try:
                self.strategy_runner(doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs
