import traceback
from typing import Tuple, List, Optional

from spacy.lang.en import English

from kazu.data.data import Document, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.utils.abbreviation_detector import KazuAbbreviationDetector


class AbbreviationFinderStep(BaseStep):
    """
    Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).
    Uses a modified version of the scispacy abbreviation finder rules, to expand abbreviations (see
    :class:`kazu.utils.abbreviation_detector.KazuAbbreviationDetector`). In this implementation, abbreviations learnt in
     one section will be applied throughout the others
    """

    def __init__(self, depends_on: List[str], exclude_abbrvs: Optional[List[str]] = None):
        super().__init__(depends_on)
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 100000000
        self.detector = KazuAbbreviationDetector(
            self.nlp, namespace=self.namespace(), exclude_abbrvs=exclude_abbrvs
        )

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        we need to override _run, as we need to calculate abbreviations over all sections in a document

        :param docs:
        :return:
        """
        failed_docs = []
        for doc in docs:
            try:
                self.detector(doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs
