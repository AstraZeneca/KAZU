import traceback
from typing import Tuple, List, Dict

from spacy.lang.en import English

from kazu.data.data import Document, Section, CharSpan, PROCESSING_EXCEPTION
from .string_preprocessing_step import StringPreprocessorStep
from kazu.utils.abbreviation_detector import AbbreviationDetector


class SciSpacyAbbreviationExpansionStep(StringPreprocessorStep):
    """
    Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).
    Uses a modified version of the scispacy abbreviation finder rules, to expand abbreviations. In this implementation,
    it's possible to apply abbreviations across the multiple sections in a :class:`kazu.data.data.Document`.
    For instance, abbreviations learnt in an abstract will also be applied throughout the body of the text

    """

    def __init__(self, depends_on: List[str]):
        super().__init__(depends_on)
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 100000000
        self.abbreviation_pipe = AbbreviationDetector(self.nlp)

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        we need to override _run, as we need to calculate abbreviations over all sections in a document

        :param docs:
        :return:
        """
        failed_docs = []
        for doc in docs:
            try:
                self.expand_abbreviations_section(doc)
            except Exception:
                doc.metadata[PROCESSING_EXCEPTION] = traceback.format_exc()
                failed_docs.append(doc)
        return docs, failed_docs

    def expand_abbreviations(self, section: Section) -> Tuple[str, Dict[CharSpan, CharSpan]]:
        """
        processes a document for abbreviations, returning a new string with all abbreviations expanded

        :param text: input document
        :return: expanded input document
        """

        doc = self.nlp(section.get_text())
        doc = self.abbreviation_pipe.run_rules(doc)
        abbreviations_sorted = sorted(doc._.abbreviations, key=lambda x: x.start_char, reverse=True)
        abbreviations_sorted_tuples = [
            (CharSpan(start=x.start_char, end=x.end_char), str(x._.long_form))
            for x in abbreviations_sorted
        ]

        result_doc, recalc_offset_map = self.modify_string(
            section=section, modifications=abbreviations_sorted_tuples
        )
        return result_doc, recalc_offset_map

    def expand_abbreviations_section(self, document: Document) -> Document:

        all = " ".join(section.get_text() for section in document.sections)
        doc = self.nlp(all)
        try:
            self.abbreviation_pipe.find_abbreviations(doc)
            expanded_text_and_offset_maps = [
                self.expand_abbreviations(section) for section in document.sections
            ]
            for (expanded_text, offset_map), section in zip(
                expanded_text_and_offset_maps, document.sections
            ):

                section.preprocessed_text = expanded_text
                section.offset_map = offset_map

        finally:
            self.abbreviation_pipe.reset()

        return document
