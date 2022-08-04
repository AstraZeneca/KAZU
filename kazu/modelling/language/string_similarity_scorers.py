import re
from abc import ABC, abstractmethod
from collections import Counter

from kazu.data.data import NumericMetric
from rapidfuzz import fuzz
from strsimpy import NGram


class StringSimilarityScorer(ABC):
    @abstractmethod
    def calculate(self, match_norm: str, syn_norm: str) -> NumericMetric:
        raise NotImplementedError()

    def __call__(self, match: str, match_norm: str, term_norm: str) -> NumericMetric:
        return self.calculate(match_norm=match_norm, syn_norm=term_norm)


class BooleanStringSimilarityScorer(ABC):
    @abstractmethod
    def calculate(self, match_norm: str, syn_norm: str) -> bool:
        raise NotImplementedError()

    def __call__(self, match: str, match_norm: str, term_norm: str) -> bool:
        return self.calculate(match_norm=match_norm, syn_norm=term_norm)


class NGramStringSimilarityScorer(StringSimilarityScorer):
    ngram = NGram(2)

    def calculate(self, match_norm: str, syn_norm: str) -> NumericMetric:
        return 2 / (self.ngram.distance(match_norm, syn_norm) + 1.0)


class NumberMatchStringSimilarityScorer(BooleanStringSimilarityScorer):
    number_finder = re.compile("[0-9]+")

    def calculate(self, match_norm: str, syn_norm: str) -> bool:
        match_norm_number_count = Counter(re.findall(self.number_finder, match_norm))
        syn_norm_number_count = Counter(re.findall(self.number_finder, syn_norm))
        return match_norm_number_count == syn_norm_number_count


class EntitySubtypeStringSimilarityScorer(BooleanStringSimilarityScorer):
    numeric_class_phrases = [re.compile(x) for x in ["TYPE [0-9]+"]]

    def calculate(self, match_norm: str, syn_norm: str) -> bool:
        for pattern in self.numeric_class_phrases:
            match = re.search(pattern, match_norm)
            if match:
                if match.group() in syn_norm:
                    continue
                else:
                    return False
            else:
                continue

        return True


class EntityNounModifierStringSimilarityScorer(BooleanStringSimilarityScorer):
    noun_modifier_phrases = ["LIKE", "SUBUNIT", "PSEUDOGENE", "RECEPTOR"]

    def calculate(self, match_norm: str, syn_norm: str) -> bool:
        for pattern in self.noun_modifier_phrases:
            required_match = True if pattern in match_norm else False
            pattern_in_syn_norm = True if pattern in syn_norm else False
            if not pattern_in_syn_norm and not required_match:
                continue
            elif pattern_in_syn_norm and required_match:
                continue
            elif pattern_in_syn_norm and not required_match:
                return False
            elif not pattern_in_syn_norm and required_match:
                return False
            else:
                raise RuntimeError("Impossible")
        return True


class RapidFuzzStringSimilarityScorer(StringSimilarityScorer):
    def calculate(self, match_norm: str, syn_norm: str) -> NumericMetric:
        if len(match_norm) > 10 and len(match_norm.split(" ")) > 4:
            return fuzz.token_sort_ratio(match_norm, syn_norm)
        else:
            return fuzz.WRatio(match_norm, syn_norm)
