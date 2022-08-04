import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Protocol

from kazu.data.data import NumericMetric
from rapidfuzz import fuzz
from strsimpy import NGram


class StringSimilarityScorer(ABC):
    """
    calculates a NumericMetric based on a string match or a normalised string match and a normalised term
    """

    @abstractmethod
    def __call__(self, reference_term: str, query_term: str) -> NumericMetric:
        raise NotImplementedError()


class BooleanStringSimilarityScorer(Protocol):
    def __call__(self, reference_term: str, query_term: str) -> bool:
        ...


class NGramStringSimilarityScorer(StringSimilarityScorer):
    ngram = NGram(2)

    def __call__(self, reference_term: str, query_term: str) -> NumericMetric:
        return 2 / (self.ngram.distance(reference_term, query_term) + 1.0)


class NumberMatchStringSimilarityScorer(StringSimilarityScorer):
    """
    checks all numbers in reference_term are represented in term_norm
    """

    number_finder = re.compile("[0-9]+")

    def __call__(self, reference_term: str, query_term: str) -> bool:
        reference_term_number_count = Counter(re.findall(self.number_finder, reference_term))
        query_term_number_count = Counter(re.findall(self.number_finder, query_term))
        return reference_term_number_count == query_term_number_count


class EntitySubtypeStringSimilarityScorer(StringSimilarityScorer):
    """
    checks all TYPE x mentions in match norm are represented in term norm
    """

    # need to handle I explicitly
    # other roman numerals get normalized to integers,
    # but not I as this would be problematic
    numeric_class_phrases = [re.compile(x) for x in ["TYPE (I|[0-9])+"]]

    def __call__(self, reference_term: str, query_term: str) -> bool:
        for pattern in self.numeric_class_phrases:
            match = re.search(pattern, reference_term)
            if match:
                if match.group() in query_term:
                    continue
                else:
                    return False
            else:
                continue

        return True


class EntityNounModifierStringSimilarityScorer(StringSimilarityScorer):
    """
    checks all modifier phrased in reference_term are represented in term_norm
    """

    noun_modifier_phrases = ["LIKE", "SUBUNIT", "PSEUDOGENE", "RECEPTOR"]

    def __call__(self, reference_term: str, query_term: str) -> bool:
        for pattern in self.noun_modifier_phrases:
            required_match = True if pattern in reference_term else False
            pattern_in_query_term = True if pattern in query_term else False
            if not pattern_in_query_term and not required_match:
                continue
            elif pattern_in_query_term and required_match:
                continue
            elif pattern_in_query_term and not required_match:
                return False
            elif not pattern_in_query_term and required_match:
                return False
            else:
                raise RuntimeError("Impossible")
        return True


class RapidFuzzStringSimilarityScorer(StringSimilarityScorer):
    """
    uses rapid fuzz to calculate string similarity. Note, if the token count >4 and reference_term has
    more than 10 chars, token_sort_ratio is used. Otherwise WRatio is used
    """

    def __call__(self, reference_term: str, query_term: str) -> NumericMetric:
        if len(reference_term) > 10 and len(reference_term.split(" ")) > 4:
            return fuzz.token_sort_ratio(reference_term, query_term)
        else:
            return fuzz.WRatio(reference_term, query_term)
