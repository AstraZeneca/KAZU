import logging
from typing import Iterable

from kazu.data.data import Document
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.steps import document_iterating_step
from kazu.steps.linking.post_processing.strategy_runner import StrategyRunner
from kazu.steps.step import ParserDependentStep

logger = logging.getLogger(__name__)


class MappingStep(ParserDependentStep):
    """A wrapper for :class:`.StrategyRunner`\\, so it can be used in a pipeline."""

    def __init__(
        self,
        parsers: Iterable[OntologyParser],
        strategy_runner: StrategyRunner,
    ):
        super().__init__(parsers)
        self.strategy_runner = strategy_runner

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        self.strategy_runner(doc)
