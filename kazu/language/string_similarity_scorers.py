import re
from collections import Counter
from typing import Protocol

from rapidfuzz import fuzz
from torch import Tensor, cosine_similarity
from cachetools import LFUCache

from kazu.data import NumericMetric
from kazu.utils.utils import Singleton
from kazu.utils.sapbert import SapBertHelper


class StringSimilarityScorer(Protocol):
    """Calculates a NumericMetric based on a string match or a normalised string match
    and a normalised synonym."""

    def __call__(self, reference_term: str, query_term: str) -> NumericMetric:
        raise NotImplementedError


class BooleanStringSimilarityScorer(StringSimilarityScorer, Protocol):
    def __call__(self, reference_term: str, query_term: str) -> bool:
        raise NotImplementedError


class NumberMatchStringSimilarityScorer(BooleanStringSimilarityScorer):
    """Checks all numbers in reference_term are represented in query_term."""

    number_finder = re.compile("[0-9]+")

    @classmethod
    def __call__(cls, reference_term: str, query_term: str) -> bool:
        reference_term_number_count = Counter(cls.number_finder.findall(reference_term))
        query_term_number_count = Counter(cls.number_finder.findall(query_term))
        return reference_term_number_count == query_term_number_count


class EntitySubtypeStringSimilarityScorer(BooleanStringSimilarityScorer):
    """Checks all TYPE x mentions in match norm are represented in syn norm."""

    # need to handle I explicitly
    # other roman numerals get normalized to integers,
    # but not I as this would be problematic
    numeric_class_phrases = re.compile("|".join(["TYPE (?:I|[0-9]+)"]))

    @classmethod
    def __call__(cls, reference_term: str, query_term: str) -> bool:
        reference_term_numeric_phrase_count = Counter(
            cls.numeric_class_phrases.findall(reference_term)
        )
        query_term_numeric_phrase_count = Counter(cls.numeric_class_phrases.findall(query_term))

        # we don't want to just do reference_term_numeric_phrase_count == query_term_numeric_phrase_count
        # because e.g. if reference term is 'diabetes' that is an NER match we've picked up in some text,
        # we want to keep hold of all of 'diabetes type I', 'diabetes type II', 'diabetes', in case surrounding context
        # enables us to disambiguate which type of diabetes it is
        return all(
            numeric_class_phase in query_term_numeric_phrase_count
            and query_term_numeric_phrase_count[numeric_class_phase] >= count
            for numeric_class_phase, count in reference_term_numeric_phrase_count.items()
        )


class EntityNounModifierStringSimilarityScorer(BooleanStringSimilarityScorer):
    """Checks all modifier phrases in reference_term are represented in query_term."""

    def __init__(self, noun_modifier_phrases: list[str]):
        self.noun_modifier_phrases = noun_modifier_phrases

    def __call__(self, reference_term: str, query_term: str) -> bool:
        # the pattern should either be in both or neither
        return all(
            (pattern in reference_term) == (pattern in query_term)
            for pattern in self.noun_modifier_phrases
        )


class RapidFuzzStringSimilarityScorer(StringSimilarityScorer):
    """Uses rapid fuzz to calculate string similarity.

    Note, if the token count >4 and reference_term has more than 10 chars,
    token_sort_ratio is used. Otherwise, WRatio is used
    """

    @staticmethod
    def __call__(reference_term: str, query_term: str) -> NumericMetric:
        if len(reference_term) > 10 and len(reference_term.split(" ")) > 4:
            return fuzz.token_sort_ratio(reference_term, query_term)
        else:
            return fuzz.WRatio(reference_term, query_term)


class SapbertStringSimilarityScorer(metaclass=Singleton):
    """Note this is an implementation of the StringSimilarityScorer Protocol, but as a
    Singleton we can't inherit it."""

    def __init__(self, sapbert: SapBertHelper, cache_size: int = 1000):
        """

        :param sapbert: The sapbert model to use
        :param cache_size: cache size, to prevent repeated calls to sapbert for the same string
        """
        self.sapbert = sapbert
        self.embedding_cache: LFUCache[str, Tensor] = LFUCache(maxsize=cache_size)

    def __call__(self, reference_term: str, query_term: str) -> float:
        if reference_term == query_term:
            return 1.0

        ref_embedding = self.embedding_cache.get(reference_term)
        query_embedding = self.embedding_cache.get(query_term)
        if ref_embedding is None and query_embedding is None:
            embeddings = self.sapbert.get_embeddings_for_strings(
                [reference_term, query_term], batch_size=2
            )
            ref_embedding = embeddings[0]
            query_embedding = embeddings[1]
            self.embedding_cache[reference_term] = ref_embedding
            self.embedding_cache[query_term] = query_embedding
        elif ref_embedding is None:
            embeddings = self.sapbert.get_embeddings_for_strings([reference_term], batch_size=1)
            ref_embedding = embeddings[0]
            self.embedding_cache[reference_term] = ref_embedding
        elif query_embedding is None:
            embeddings = self.sapbert.get_embeddings_for_strings([query_term], batch_size=1)
            query_embedding = embeddings[0]
            self.embedding_cache[query_term] = query_embedding

        assert ref_embedding is not None
        assert query_embedding is not None
        return cosine_similarity(ref_embedding, query_embedding, dim=0).item()
