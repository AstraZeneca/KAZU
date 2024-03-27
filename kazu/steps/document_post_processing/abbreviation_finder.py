from typing import Optional

from kazu.data import Document
from kazu.steps import Step, document_iterating_step
from kazu.utils.abbreviation_detector import KazuAbbreviationDetector


class AbbreviationFinderStep(Step):
    """Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    Uses a modified version of the scispacy abbreviation finder rules, to expand abbreviations (see
    :class:`kazu.utils.abbreviation_detector.KazuAbbreviationDetector`\\ ). In this implementation,
    abbreviations learnt in one section will be applied throughout the others.
    """

    def __init__(self, exclude_abbrvs: Optional[list[str]] = None):
        self.detector = KazuAbbreviationDetector(
            namespace=self.namespace(), exclude_abbrvs=exclude_abbrvs
        )

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        """

        :param doc:
        :return:
        """
        self.detector(doc)
