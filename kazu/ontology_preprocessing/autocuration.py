import dataclasses
from typing import Iterable, Protocol

import pandas as pd
from kazu.data.data import CuratedTerm, MentionConfidence, CuratedTermBehaviour
from kazu.utils.string_normalizer import StringNormalizer


class AutoCurationAction(Protocol):
    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        pass


class SymbolicToCaseSensitiveAction(AutoCurationAction):
    def __init__(self, entity_class: str):
        self.entity_class = entity_class

    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        is_symbolic = set()
        new_forms = set()
        for form in curated_term.original_forms:
            is_symbolic.add(
                StringNormalizer.classify_symbolic(form.string, entity_class=self.entity_class)
            )
            new_forms.add(dataclasses.replace(form, case_sensitive=True))
        if all(is_symbolic):
            return dataclasses.replace(curated_term, original_forms=frozenset(new_forms))
        else:
            return curated_term


class IsCommmonWord(AutoCurationAction):
    def __init__(self, path: str):
        self.common_words = set(pd.read_csv(path, sep="\t", header=None).values.flatten())

    def __call__(self, curated_term: CuratedTerm) -> CuratedTerm:
        is_common = set()
        new_forms = set()
        for form in curated_term.original_forms:
            is_common.add(all(word in self.common_words for word in form.string.lower().split(" ")))
            new_forms.add(dataclasses.replace(form, mention_confidence=MentionConfidence.POSSIBLE))

        if any(is_common):
            return dataclasses.replace(curated_term, original_forms=frozenset(new_forms))
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


class AutoCurator:
    def __init__(self, actions: list[AutoCurationAction]):
        self.actions = actions

    def __call__(self, curations: set[CuratedTerm]) -> Iterable[CuratedTerm]:
        for curation in curations:
            for action in self.actions:
                curation = action(curation)

            yield curation
