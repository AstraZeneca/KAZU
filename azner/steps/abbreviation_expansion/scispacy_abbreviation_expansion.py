import logging
from collections import defaultdict
from typing import Tuple, List, Optional, Set, Dict

from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokens import Span, Doc

from azner.data.data import Document, Section, CharSpan
from azner.steps import StringPreprocessorStep

logger = logging.getLogger(__name__)


def find_abbreviation(
    long_form_candidate: Span, short_form_candidate: Span
) -> Tuple[Span, Optional[Span]]:
    """
    Implements the abbreviation detection algorithm in "A simple algorithm
    for identifying abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    The algorithm works by enumerating the characters in the short form of the abbreviation,
    checking that they can be matched against characters in a candidate text for the long form
    in order, as well as requiring that the first letter of the abbreviated form matches the
    _beginning_ letter of a word.

    Parameters
    ----------
    long_form_candidate: Span, required.
        The spaCy span for the long form candidate of the definition.
    short_form_candidate: Span, required.
        The spaCy span for the abbreviation candidate.

    Returns
    -------
    A Tuple[Span, Optional[Span]], representing the short form abbreviation and the
    span corresponding to the long form expansion, or None if a match is not found.
    """
    long_form = " ".join([x.text for x in long_form_candidate])
    short_form = " ".join([x.text for x in short_form_candidate])

    long_index = len(long_form) - 1
    short_index = len(short_form) - 1

    while short_index >= 0:
        current_char = short_form[short_index].lower()
        # We don't check non alpha-numeric characters.
        if not current_char.isalnum():
            short_index -= 1
            continue

            # Does the character match at this position? ...
            # .... or if we are checking the first character of the abbreviation, we enforce
            # to be the _starting_ character of a span.
        while (long_index >= 0 and long_form[long_index].lower() != current_char) or (
            short_index == 0 and long_index > 0 and long_form[long_index - 1].isalnum()
        ):
            long_index -= 1

        if long_index < 0:
            return short_form_candidate, None

        long_index -= 1
        short_index -= 1

    # The last subtraction will either take us on to a whitespace character, or
    # off the front of the string (i.e. long_index == -1). Either way, we want to add
    # one to get back to the start character of the long form
    long_index += 1

    # Now we know the character index of the start of the character span,
    # here we just translate that to the first token beginning after that
    # value, so we can return a spaCy span instead.
    word_lengths = 0
    starting_index = None
    for i, word in enumerate(long_form_candidate):
        # need to add 1 for the space characters
        word_lengths += len(word.text_with_ws)
        if word_lengths > long_index:
            starting_index = i
            break

    return short_form_candidate, long_form_candidate[starting_index:]


def filter_matches(matcher_output: List[Tuple[int, int, int]], doc: Doc) -> List[Tuple[Span, Span]]:
    # Filter into two cases:
    # 1. <Short Form> ( <Long Form> )
    # 2. <Long Form> (<Short Form>) [this case is most common].
    candidates = []
    for match in matcher_output:
        start = match[1]
        end = match[2]
        # Ignore spans with more than 8 words in them, and spans at the start of the doc
        if end - start > 8 or start == 1:
            continue
        if end - start > 3:
            # Long form is inside the parens.
            # Take one word before.
            short_form_candidate = doc[start - 2 : start - 1]
            long_form_candidate = doc[start:end]
        else:
            # Normal case.
            # Short form is inside the parens.
            short_form_candidate = doc[start:end]

            # Sum character lengths of contents of parens.
            abbreviation_length = sum([len(x) for x in short_form_candidate])
            max_words = min(abbreviation_length + 5, abbreviation_length * 2)
            # Look up to max_words backwards
            long_form_candidate = doc[max(start - max_words - 1, 0) : start - 1]

        # add candidate to candidates if candidates pass filters
        if short_form_filter(short_form_candidate):
            candidates.append((long_form_candidate, short_form_candidate))

    return candidates


def short_form_filter(span: Span) -> bool:
    # All words are between length 2 and 10
    if not all([2 <= len(x) < 10 for x in span]):
        return False

    # At least 50% of the short form should be alpha
    if (sum([c.isalpha() for c in span.text]) / len(span.text)) < 0.5:
        return False

    # The first character of the short form should be alpha
    if not span.text[0].isalpha():
        return False
    return True


class AbbreviationDetector:
    """
    Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    This class sets the `._.abbreviations` attribute on spaCy Doc.

    The abbreviations attribute is a `List[Span]` where each Span has the `Span._.long_form`
    attribute set to the long form definition of the abbreviation.

    Note that this class does not replace the spans, or merge them.

    Original Credit:
    https://github.com/allenai/scispacy
    @inproceedings{neumann-etal-2019-scispacy,
    title = "{S}cispa{C}y: {F}ast and {R}obust {M}odels for {B}iomedical {N}atural {L}anguage {P}rocessing",
    author = "Neumann, Mark  and
      King, Daniel  and
      Beltagy, Iz  and
      Ammar, Waleed",
    booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5034",
    doi = "10.18653/v1/W19-5034",
    pages = "319--327",
    eprint = {arXiv:1902.07669},
    }
    """

    def __init__(self, nlp) -> None:
        Doc.set_extension("abbreviations", default=[], force=True)
        Span.set_extension("long_form", default=None, force=True)
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("parenthesis", [[{"ORTH": "("}, {"OP": "+"}, {"ORTH": ")"}]])
        self.global_matcher = Matcher(nlp.vocab)
        self.rules = {}
        self.to_remove = set()

    def find_abbreviations(self, doc: Doc) -> None:
        """
        add matcher rules based on abbreviations found
        :param doc:
        :return:
        """
        matches = self.matcher(doc)
        matches_no_brackets = [(x[0], x[1] + 1, x[2] - 1) for x in matches]
        filtered = filter_matches(matches_no_brackets, doc)
        all_occurences: Dict[Span, Set[Span]] = defaultdict(set)
        already_seen_long: Set[str] = set()
        already_seen_short: Set[str] = set()
        for (long_candidate, short_candidate) in filtered:
            short, long = find_abbreviation(long_candidate, short_candidate)
            # We need the long and short form definitions to be unique, because we need
            # to store them so we can look them up later. This is a bit of a
            # pathalogical case also, as it would mean an abbreviation had been
            # defined twice in a document. There's not much we can do about this,
            # but at least the case which is discarded will be picked up below by
            # the global matcher. So it's likely that things will work out ok most of the time.
            new_long = str(long) not in already_seen_long if long else False
            new_short = str(short) not in already_seen_short
            if long is not None and new_long and new_short:
                already_seen_long.add(str(long))
                already_seen_short.add(str(short))
                all_occurences[long].add(short)
                self.rules[str(long)] = long
                # Add a rule to a matcher to find exactly this substring.
                self.global_matcher.add(str(long), [[{"ORTH": x.text}] for x in short])

    def run_rules(self, doc: Doc) -> Doc:
        """
        apply abbreviation matcher rules to a spacy doc
        :param doc:
        :return:
        """
        all_occurences: Dict[Span, Set[Span]] = defaultdict(set)
        global_matches = self.global_matcher(doc)
        for match, start, end in global_matches:
            string_key = self.global_matcher.vocab.strings[match]
            self.to_remove.add(string_key)
            all_occurences[self.rules[string_key]].add(doc[start:end])

        occurences = list((k, v) for k, v in all_occurences.items())
        for (long_form, short_forms) in occurences:
            for short in short_forms:
                short._.long_form = long_form
                doc._.abbreviations.append(short)
        return doc

    def reset(self):
        """
        in this implementation, the matcher rules retains state, allowing the rules learnt from one section to be
        applied to another. Calling reset removes this state, allowing new rules to be learnt (i.e. when processing
        another document)
        :return:
        """
        keys = [self.to_remove.pop() for _ in range(len(self.to_remove))]
        for key in keys:
            # Clean up the global matcher.
            self.global_matcher.remove(key)
        self.rules.clear()
        self.global_matcher = Matcher(self.nlp.vocab)


class SciSpacyAbbreviationExpansionStep(StringPreprocessorStep):
    """
    Use the scispacy abbreviation finder rules, to expand abbreviations ahead of NER
    """

    def __init__(self, depends_on: List[str]):
        """

        :param depends_on:
        :param override_original_section_text: if True, original section text is moved to the section metadata under
                "original_text", and Section.text becomes expanded text. If false, expanded text is added to metadata
                under "expanded_abbreviations". In all cases, a map of offsets from modifications to abbreviations is
                created under "abbreviations_offset_mappings"
        """
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
        for doc in docs:
            self.expand_abbreviations_section(doc)
        return docs, []

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

        all = " ".join([section.get_text() for section in document.sections])
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
