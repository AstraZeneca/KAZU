import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Union, overload, Any

from kazu.data import (
    Document,
    Entity,
    Section,
    OntologyStringResource,
    OntologyStringBehaviour,
    MentionConfidence,
    LinkingCandidate,
    Synonym,
)
from kazu.utils.grouping import sort_then_group
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

logger = logging.getLogger(__name__)


def linking_candidates_to_ontology_string_resources(
    candidates: Iterable[LinkingCandidate],
) -> set[OntologyStringResource]:
    """

    :param candidates:
    :return:
    """
    result = set()
    for syn_norm, candidates in sort_then_group(candidates, key_func=lambda x: x.synonym_norm):
        alts = set()
        for candidate in candidates:
            for raw_syn in candidate.raw_synonyms:
                alts.add(
                    Synonym(
                        text=raw_syn,
                        case_sensitive=False,
                        mention_confidence=MentionConfidence.PROBABLE,
                    )
                )
        result.add(
            OntologyStringResource(
                original_synonyms=frozenset(alts),
                behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
            )
        )
    return result


def find_document_from_entity(docs: list[Document], entity: Entity) -> Document:
    """For a given entity and a list of docs, find the doc the entity belongs to.

    :param docs:
    :param entity:
    :return:
    """
    for doc in docs:
        if entity in doc.get_entities():
            return doc
    raise RuntimeError(f"Error! Entity {entity} is not attached to a document")


def documents_to_id_section_map(docs: list[Document]) -> dict[int, Section]:
    """Return a map of documents, indexed by order of sections.

    :param docs:
    :return:
    """
    result = {}
    i = 0
    for doc in docs:
        for section in doc.sections:
            result[i] = section
            i += 1
    return result


def documents_to_document_section_batch_encodings_map(
    docs: list[Document],
    tokenizer: PreTrainedTokenizerBase,
    stride: int = 128,
    max_length: int = 512,
) -> tuple[BatchEncoding, dict[int, Section]]:
    """Convert documents into a BatchEncoding. Also returns a list of <int + section>
    for the resulting encoding.

    :param docs:
    :param tokenizer:
    :param stride:
    :param max_length:
    :return:
    """
    id_section_map = documents_to_id_section_map(docs)
    strings = [section.text for section in id_section_map.values()]

    batch_encodings = tokenizer(
        strings,
        stride=stride,
        max_length=max_length,
        truncation=TruncationStrategy.LONGEST_FIRST,
        return_overflowing_tokens=True,
        padding=PaddingStrategy.MAX_LENGTH,
    )
    return batch_encodings, id_section_map


def get_match_entity_class_hash(ent: Entity) -> int:
    return hash(
        (
            ent.match,
            ent.entity_class,
        )
    )


PathLike = Union[str, Path]


def as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


class EntityClassFilter:
    """A condition that returns True if a document has any entities that match the class
    of the required_entity_classes."""

    def __init__(self, required_entity_classes: Iterable[str]):
        """

        :param required_entity_classes: list of str, specifying entity classes to assess
        """
        self.required_entities = set(required_entity_classes)

    def __call__(self, document: Document) -> bool:
        return any(
            (entity.entity_class in self.required_entities for entity in document.get_entities())
        )


@overload
def _create_ngrams_iter(tokens: str, n: int) -> Iterable[str]:
    pass


@overload
def _create_ngrams_iter(tokens: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    pass


def _create_ngrams_iter(tokens, n=2):
    """Yields ngrams of the input as a sequence of strings.

    Tokens can be a single string where each token is a character, or an Iterable of
    words. If tokens is a str
    """
    num_tokens = len(tokens)
    for i in range(num_tokens):
        ngram_end_index = i + n
        if ngram_end_index > num_tokens:
            # ngram would extend beyond end of parts
            # it's ok for it to be the same number though as otherwise
            # we don't get the final word of ngrams at the right of parts
            break
        yield tokens[i : i + n]


def create_char_ngrams(s: str, n: int = 2) -> list[str]:
    """Return list of char ngrams as a string."""
    return list(_create_ngrams_iter(s, n))


def create_word_ngrams(s: str, n: int = 2) -> list[str]:
    """Return list of word ngrams as a single space-separated string."""
    words = s.split(" ")
    ngrams_iter = _create_ngrams_iter(words, n)
    return [" ".join(ngram) for ngram in ngrams_iter]


class Singleton(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    @staticmethod
    def clear_all():
        logger.warning(
            "When clearing singletons, check that instances of classes with metaclass=Singleton are "
            "not used as class fields, as this will cause unexpected behaviour. Also note that any existing "
            "classes that use a Singleton will need to be re-instantiated to work correctly!. Much pain "
            "can be avoided by observing this simple principle! Some people consider Singletons an "
            "anti-pattern, and this is a reason why!!!"
        )
        Singleton._instances.clear()
