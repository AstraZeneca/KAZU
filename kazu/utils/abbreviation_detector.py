import copy
import itertools
import logging

from kazu.data.data import Document, Entity, Section
from kazu.utils.grouping import sort_then_group

"""
Original Credit:
https://github.com/allenai/scispacy
https://github.com/allenai/scispacy/blob/main/scispacy/abbreviation.py
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

from typing import Tuple, List, Optional, Set, Dict
from collections import defaultdict
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
from spacy.language import Language

logger = logging.getLogger(__name__)

SectionAndLongToShortCandidates = Tuple[Section, Span, Span]
SectionToSpacyDoc = Dict[Section, Doc]
SectionToCharacterIndexedEntities = Dict[Section, Dict[Tuple[int, int], Set[Entity]]]


def find_abbreviation(
    long_form_candidate: Span, short_form_candidate: Span
) -> Tuple[Span, Optional[Span]]:
    """
    from https://github.com/allenai/scispacy/blob/main/scispacy/abbreviation.py

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
        while (
            (long_index >= 0 and long_form[long_index].lower() != current_char)
            or (short_index == 0 and long_index > 0 and long_form[long_index - 1].isalnum())
            # .... or if we are checking the first character of the abbreviation, we enforce
            # to be the _starting_ character of a span.
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


def filter_matches(
    section: Section, matcher_output: List[Tuple[int, int, int]], doc: Doc
) -> List[Tuple[Section, Span, Span]]:
    """
    from https://github.com/allenai/scispacy/blob/main/scispacy/abbreviation.py

    :param section:
    :param matcher_output:
    :param doc:
    :return:
    """
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
            candidates.append((section, long_form_candidate, short_form_candidate))

    return candidates


def short_form_filter(span: Span) -> bool:
    """
    from https://github.com/allenai/scispacy/blob/main/scispacy/abbreviation.py

    :param span:
    :return:
    """

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


class KazuAbbreviationDetector:
    """
    Modified version of https://github.com/allenai/scispacy/blob/main/scispacy/abbreviation.py

    see top of file for original implementation credit

    Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    If an abbreviation is detected, a new instance of :class:`Entity` is generated, copying information from the
    originating long span. If the original long span was not an entity, the abbreviation entity is removed. In the latter
    case, you can force the class to not delete entities by providing a list of strings to exclude_abbrvs.
    For instance, this might be wise for abbreviations that are very common and therefore not defined (e.g. 'NSCLC').
    Note, however, that the abbreviation detection is always preferred, so if a long form entity is detected, that
    will always be chosen

    """

    def __init__(
        self,
        nlp: Language,
        namespace: str,
        exclude_abbrvs: Optional[List[str]] = None,
    ) -> None:
        """

        :param nlp: spacy model to use
        :param namespace: the namespace to give any generated entities
        :param exclude_abbrvs: detected abbreviations matching this list will not be removed, even if no source
            entities are found
        """
        self.namespace = namespace
        self.exclude_abbrvs: Set[str] = set(exclude_abbrvs) if exclude_abbrvs is not None else set()
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("parenthesis", [[{"ORTH": "("}, {"OP": "+"}, {"ORTH": ")"}]])

    def __call__(self, document: Document):
        (
            section_and_long_to_short_candidates,
            section_to_ents_by_char_index,
            section_to_spacy_doc,
        ) = self._find_candidates_and_index_sections(document)

        (
            global_matcher,
            long_form_string_to_source_ents,
        ) = self._build_matcher_and_identify_source_entities(
            section_and_long_to_short_candidates, section_to_ents_by_char_index
        )
        self._find_abbreviations_and_override_entities(
            global_matcher,
            long_form_string_to_source_ents,
            section_to_ents_by_char_index,
            section_to_spacy_doc,
        )

    def _find_abbreviations_and_override_entities(
        self,
        global_matcher: Matcher,
        long_form_string_to_source_ents: Dict[str, Set[Entity]],
        section_to_ents_by_char_index: SectionToCharacterIndexedEntities,
        section_to_spacy_doc: SectionToSpacyDoc,
    ):
        for section, spacy_doc in section_to_spacy_doc.items():
            global_matches = global_matcher(spacy_doc)
            for match, start, end in global_matches:
                string_key = global_matcher.vocab.strings[match]
                abbrv_span: Span = spacy_doc[start:end]
                abbrv_char_index_key = (
                    abbrv_span.start_char,
                    abbrv_span.end_char,
                )
                for source_ent in long_form_string_to_source_ents.get(string_key, set()):
                    new_ent = self._create_ent_from_span_and_source_ent(
                        abbrv_span, section, source_ent
                    )
                    section.entities.append(new_ent)
                    if abbrv_char_index_key in section_to_ents_by_char_index[section]:
                        section_to_ents_by_char_index[section][abbrv_char_index_key].add(new_ent)

                abbrv_ents_at_index_by_namespace = {
                    k: set(v)
                    for k, v in sort_then_group(
                        section_to_ents_by_char_index[section].get(abbrv_char_index_key, set()),
                        key_func=lambda x: x.namespace,
                    )
                }

                # multiple abbrv ents at index. if one is generated by self.namespace, keep that one.
                if self.namespace in abbrv_ents_at_index_by_namespace:
                    abbrv_ents_at_index_by_namespace.pop(self.namespace)
                    for ent_to_remove in itertools.chain.from_iterable(
                        abbrv_ents_at_index_by_namespace.values()
                    ):
                        section.entities.remove(ent_to_remove)
                else:
                    # Else, remove others unless they are in exclude list
                    for ent_to_remove in itertools.chain.from_iterable(
                        abbrv_ents_at_index_by_namespace.values()
                    ):
                        if ent_to_remove.match not in self.exclude_abbrvs:
                            section.entities.remove(ent_to_remove)

    def _create_ent_from_span_and_source_ent(
        self, abbrv_span: Span, section: Section, source_ent: Entity
    ) -> Entity:
        new_ent_data: Dict = copy.deepcopy(source_ent.__dict__)
        new_ent_data.pop("start")
        new_ent_data.pop("end")
        new_ent_data.pop("spans")
        new_ent_data.pop("match_norm")
        new_ent_data.pop("match")
        new_ent_data.pop("namespace")
        new_ent = Entity.from_spans(
            text=section.get_text(),
            spans=[
                (
                    abbrv_span.start_char,
                    abbrv_span.end_char,
                )
            ],
            namespace=self.namespace,
            join_str="",
            **new_ent_data,
        )
        return new_ent

    def _build_matcher_and_identify_source_entities(
        self,
        section_and_long_to_short_candidates: List[SectionAndLongToShortCandidates],
        section_to_ents_by_char_index: SectionToCharacterIndexedEntities,
    ) -> Tuple[Matcher, Dict[str, Set[Entity]]]:
        global_matcher = Matcher(self.nlp.vocab)
        all_occurences: Dict[Span, Set[Span]] = defaultdict(set)
        already_seen_long: Set[str] = set()
        already_seen_short: Set[str] = set()
        long_form_string_to_source_ents: Dict[str, Set[Entity]] = {}
        for (section, long_candidate, short_candidate) in section_and_long_to_short_candidates:
            short, long = find_abbreviation(long_candidate, short_candidate)
            # We need the long and short form definitions to be unique, because we need
            # to store them so we can look them up later. This is a bit of a
            # pathalogical case also, as it would mean an abbreviation had been
            # defined twice in a document. There's not much we can do about this,
            # but at least the case which is discarded will be picked up below by
            # the global matcher. So it's likely that things will work out ok most of the time.
            new_long = long.text not in already_seen_long if long else False
            new_short = short.text not in already_seen_short
            if long is not None and new_long and new_short:
                char_index_key = (
                    long.start_char,
                    long.end_char,
                )
                already_seen_long.add(long.text)
                already_seen_short.add(short.text)
                all_occurences[long].add(short)
                # Add a rule to a matcher to find exactly this substring.
                global_matcher.add(long.text, [[{"ORTH": x.text} for x in short]])
                long_form_string_to_source_ents[long.text] = section_to_ents_by_char_index[
                    section
                ].get(char_index_key, set())
        return global_matcher, long_form_string_to_source_ents

    def _find_candidates_and_index_sections(
        self, document: Document
    ) -> Tuple[
        List[SectionAndLongToShortCandidates], SectionToCharacterIndexedEntities, SectionToSpacyDoc
    ]:
        long_to_short_candidates: List[SectionAndLongToShortCandidates] = []
        section_to_spacy_doc = {}
        section_to_ents_by_char_index = {}
        for section in document.sections:
            spacy_doc = self.nlp(section.get_text())
            matches = self.matcher(spacy_doc)
            matches_no_brackets = [(x[0], x[1] + 1, x[2] - 1) for x in matches]

            long_to_short_candidates.extend(filter_matches(section, matches_no_brackets, spacy_doc))
            section_to_spacy_doc[section] = spacy_doc
            section_to_ents_by_char_index[section] = {
                k: set(v)
                for k, v in sort_then_group(
                    filter(lambda x: len(x.spans) == 1, section.entities),
                    key_func=lambda x: (
                        x.start,
                        x.end,
                    ),
                )
            }
        return long_to_short_candidates, section_to_ents_by_char_index, section_to_spacy_doc
