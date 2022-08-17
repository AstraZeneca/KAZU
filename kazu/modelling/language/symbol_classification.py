from abc import ABC

from gilda.process import replace_dashes
from kazu.utils.string_normalizer import StringNormalizer


class SymbolClassifier(ABC):
    """
    It's often useful to differentiate between symbolic and non-symbolic strings. Examples of symbolic strings
    include abbreviations, identifiers etc such as EGFR, ENSG000001.

    Since the rules that should be used for classification is somewhat linked to the underlying entity class, it's
    useful to have an interface
    """

    @staticmethod
    def is_symbolic(string: str) -> bool:
        raise NotImplementedError()


class DefaultSymbolClassifier(SymbolClassifier):
    @staticmethod
    def is_symbolic(string: str) -> bool:
        return StringNormalizer.is_probably_symbol_like(string)


class GeneSymbolClassifier(SymbolClassifier):
    @staticmethod
    def word_like_filter(word: str):
        if len(word) < 4:
            return False
        else:

            upper_count = 1
            lower_count = 1
            int_count = 1

            for char in word:
                if char.isalpha():
                    if char.isupper():
                        upper_count += 1
                    else:
                        lower_count += 1
                elif char.isnumeric():
                    int_count += 1

            upper_lower_ratio = float(upper_count) / float(lower_count)
            int_alpha_ratio = float(int_count) / (float(upper_count + lower_count - 1))
            if upper_lower_ratio > 1.0 or int_alpha_ratio > 1.0:
                return False
            else:
                return True

    @classmethod
    def count_word_like_tokens(cls, raw_str: str) -> int:
        raw_str = replace_dashes(raw_str, " ")
        tokens = raw_str.split(" ")
        if len(tokens) == 1:
            return 0
        else:
            return sum([1 for x in tokens if cls.word_like_filter(x)])

    @classmethod
    def is_symbolic(cls, string: str) -> bool:
        if cls.count_word_like_tokens(string) == 0:
            return True
        else:
            return False
