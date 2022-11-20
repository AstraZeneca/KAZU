import logging

from kazu.data.data import Document
from kazu.steps import Step, iterating_step
from kazu.steps.linking.post_processing.strategy_runner import StrategyRunner

logger = logging.getLogger(__name__)


class MappingStep(Step):
    """
    A wrapper for :class:`.StrategyRunner`, so it can be used in a pipeline.
    """

    def __init__(self, strategy_runner: StrategyRunner):
        self.strategy_runner = strategy_runner

    @iterating_step
    def __call__(self, doc: Document) -> None:
        self.strategy_runner(doc)
