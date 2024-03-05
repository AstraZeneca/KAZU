import dataclasses
from typing import Iterable, Protocol

from kazu.data.data import CuratedTerm, MentionConfidence, CuratedTermBehaviour
from kazu.utils.string_normalizer import StringNormalizer


class AutoCurationAction(Protocol):
    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        raise NotImplementedError


class SymbolicToCaseSensitiveAction(AutoCurationAction):
    def __init__(self, entity_class: str):
        self.entity_class = entity_class

    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        all_symbolic = all(
            StringNormalizer.classify_symbolic(form.string, entity_class=self.entity_class)
            for form in curated_term.original_forms
        )
        if all_symbolic:
            return dataclasses.replace(
                curated_term,
                original_forms=frozenset(
                    dataclasses.replace(form, case_sensitive=True)
                    for form in curated_term.original_forms
                ),
                alternative_forms=frozenset(
                    dataclasses.replace(form, case_sensitive=True)
                    for form in curated_term.alternative_forms
                ),
            )
        else:
            return curated_term


class IsCommmonWord(AutoCurationAction):
    def __init__(self, path: str):
        with open(path, mode="r") as inf:
            self.common_words = {line.rstrip() for line in inf.readlines()}

    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        found_common = (
            all(word in self.common_words for word in form.string.lower().split())
            for form in curated_term.original_forms
        )

        if any(found_common):
            return dataclasses.replace(
                curated_term,
                original_forms=frozenset(
                    dataclasses.replace(form, mention_confidence=MentionConfidence.POSSIBLE)
                    for form in curated_term.original_forms
                ),
                alternative_forms=frozenset(
                    dataclasses.replace(form, mention_confidence=MentionConfidence.POSSIBLE)
                    for form in curated_term.alternative_forms
                ),
            )
        else:
            return curated_term


class MinLength(AutoCurationAction):
    def __init__(self, min_len: int = 2):
        self.min_len = min_len

    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        for form in curated_term.original_forms:
            if len(form.string) < self.min_len:
                return dataclasses.replace(
                    curated_term, behaviour=CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING
                )

        return curated_term


class MaxLength(AutoCurationAction):
    """Drop terms that exceed a maximum string length."""

    def __init__(self, max_len: int = 60):
        self.max_len = max_len

    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        if any(len(form.string) > self.max_len for form in curated_term.original_forms):
            return dataclasses.replace(
                curated_term, behaviour=CuratedTermBehaviour.DROP_SYNONYM_TERM_FOR_LINKING
            )
        else:
            return curated_term


def is_upper_case_word_to_case_insensitive(curated_term: CuratedTerm) -> CuratedTerm:
    """Make curations where all original forms are all uppercase alphabetical characters
    case-insensitive.

    Some data sources use all-caps strings for nouns that can be considered case-
    insensitive (e.g. Chembl).

    :param curated_term:
    :return:
    """

    if all(form.string.isupper() and form.string.isalpha() for form in curated_term.original_forms):
        return dataclasses.replace(
            curated_term,
            original_forms=frozenset(
                dataclasses.replace(form, case_sensitive=False)
                for form in curated_term.original_forms
            ),
            alternative_forms=frozenset(
                dataclasses.replace(form, case_sensitive=False)
                for form in curated_term.alternative_forms
            ),
        )
    else:
        return curated_term


def initial_lowercase_then_upper_to_case_sensitive(curated_term: CuratedTerm) -> CuratedTerm:
    """If a form starts with a lowercase character followed by an uppercase character,
    then all forms should be case-sensitive.

    E.g. "eGFR" vs "EGFR".

    :param curated_term:
    :return:
    """
    if any(
        len(form.string) >= 2 and form.string[0].islower() and form.string[1].isupper()
        for form in curated_term.original_forms
    ):
        return dataclasses.replace(
            curated_term,
            original_forms=frozenset(
                dataclasses.replace(form, case_sensitive=True)
                for form in curated_term.original_forms
            ),
            alternative_forms=frozenset(
                dataclasses.replace(form, case_sensitive=True)
                for form in curated_term.alternative_forms
            ),
        )
    else:
        return curated_term


class AutoCurator:
    def __init__(self, actions: list[AutoCurationAction]):
        self.actions = actions

    def __call__(self, curations: set[CuratedTerm]) -> Iterable[CuratedTerm]:
        for curation in curations:
            for action in self.actions:
                curation = action(curation)

            yield curation
