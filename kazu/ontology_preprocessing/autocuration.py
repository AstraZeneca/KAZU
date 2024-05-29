import dataclasses
from typing import Iterable, Protocol

from kazu.data import OntologyStringResource, MentionConfidence, OntologyStringBehaviour
from kazu.utils.string_normalizer import StringNormalizer


class AutoCurationAction(Protocol):
    def __call__(self, resource: OntologyStringResource) -> OntologyStringResource:
        raise NotImplementedError


class SymbolicToCaseSensitiveAction(AutoCurationAction):
    def __init__(self, entity_class: str):
        self.entity_class = entity_class

    def __call__(self, resource: OntologyStringResource) -> OntologyStringResource:
        all_symbolic = all(
            StringNormalizer.classify_symbolic(syn.text, entity_class=self.entity_class)
            for syn in resource.original_synonyms
        )
        if all_symbolic:
            return dataclasses.replace(
                resource,
                original_synonyms=frozenset(
                    dataclasses.replace(syn, case_sensitive=True)
                    for syn in resource.original_synonyms
                ),
                alternative_synonyms=frozenset(
                    dataclasses.replace(syn, case_sensitive=True)
                    for syn in resource.alternative_synonyms
                ),
            )
        else:
            return resource


class IsCommmonWord(AutoCurationAction):
    def __init__(self, path: str):
        with open(path, mode="r") as inf:
            self.common_words = {line.rstrip() for line in inf.readlines()}

    def __call__(self, resource: OntologyStringResource) -> OntologyStringResource:
        found_common = (
            all(word in self.common_words for word in syn.text.lower().split())
            for syn in resource.original_synonyms
        )

        if any(found_common):
            return dataclasses.replace(
                resource,
                original_synonyms=frozenset(
                    dataclasses.replace(syn, mention_confidence=MentionConfidence.POSSIBLE)
                    for syn in resource.original_synonyms
                ),
                alternative_synonyms=frozenset(
                    dataclasses.replace(syn, mention_confidence=MentionConfidence.POSSIBLE)
                    for syn in resource.alternative_synonyms
                ),
            )
        else:
            return resource


class MinLength(AutoCurationAction):
    def __init__(self, min_len: int = 2):
        self.min_len = min_len

    def __call__(self, resource: OntologyStringResource) -> OntologyStringResource:
        for syn in resource.original_synonyms:
            if len(syn.text) < self.min_len:
                return dataclasses.replace(
                    resource, behaviour=OntologyStringBehaviour.DROP_FOR_LINKING
                )

        return resource


class MaxLength(AutoCurationAction):
    """Drop resources that exceed a maximum string length."""

    def __init__(self, max_len: int = 60):
        self.max_len = max_len

    def __call__(self, resource: OntologyStringResource) -> OntologyStringResource:
        if any(len(syn.text) > self.max_len for syn in resource.original_synonyms):
            return dataclasses.replace(resource, behaviour=OntologyStringBehaviour.DROP_FOR_LINKING)
        else:
            return resource


def is_upper_case_word_to_case_insensitive(
    resource: OntologyStringResource,
) -> OntologyStringResource:
    """Make Resources where all original synonyms are all uppercase alphabetical
    characters case-insensitive.

    Some data sources use all-caps strings for nouns that can be considered case-
    insensitive (e.g. Chembl).

    :param resource:
    :return:
    """

    if all(syn.text.isupper() and syn.text.isalpha() for syn in resource.original_synonyms):
        return dataclasses.replace(
            resource,
            original_synonyms=frozenset(
                dataclasses.replace(syn, case_sensitive=False) for syn in resource.original_synonyms
            ),
            alternative_synonyms=frozenset(
                dataclasses.replace(syn, case_sensitive=False)
                for syn in resource.alternative_synonyms
            ),
        )
    else:
        return resource


def initial_lowercase_then_upper_to_case_sensitive(
    resource: OntologyStringResource,
) -> OntologyStringResource:
    """If a synonym starts with a lowercase character followed by an uppercase
    character, then all synonyms should be case-sensitive.

    E.g. "eGFR" vs "EGFR".

    :param resource:
    :return:
    """
    if any(
        len(syn.text) >= 2 and syn.text[0].islower() and syn.text[1].isupper()
        for syn in resource.original_synonyms
    ):
        return dataclasses.replace(
            resource,
            original_synonyms=frozenset(
                dataclasses.replace(syn, case_sensitive=True) for syn in resource.original_synonyms
            ),
            alternative_synonyms=frozenset(
                dataclasses.replace(syn, case_sensitive=True)
                for syn in resource.alternative_synonyms
            ),
        )
    else:
        return resource


class LikelyAcronym(AutoCurationAction):
    """If all synonyms are less than or equal to the specified length, and are all upper
    case, give a confidence of POSSIBLE to all forms."""

    def __init__(self, max_len_to_consider: int = 5):
        self.max_len_to_consider = max_len_to_consider

    def __call__(self, resource: OntologyStringResource) -> OntologyStringResource:

        if all(
            len(syn.text) <= self.max_len_to_consider and syn.text.isupper()
            for syn in resource.original_synonyms
        ):
            return dataclasses.replace(
                resource,
                original_synonyms=frozenset(
                    dataclasses.replace(syn, mention_confidence=MentionConfidence.POSSIBLE)
                    for syn in resource.original_synonyms
                ),
                alternative_synonyms=frozenset(
                    dataclasses.replace(syn, mention_confidence=MentionConfidence.POSSIBLE)
                    for syn in resource.alternative_synonyms
                ),
            )
        else:
            return resource


class AutoCurator:
    def __init__(self, actions: list[AutoCurationAction]):
        self.actions = actions

    def __call__(self, resources: set[OntologyStringResource]) -> Iterable[OntologyStringResource]:
        for resource in resources:
            for action in self.actions:
                resource = action(resource)

            yield resource
