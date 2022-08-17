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
    def is_symbolic(string: str) -> bool:
        string = replace_dashes(string, " ")
        tokens = string.split(" ")
        if len(tokens) == 1:
            # single tokens are likely gene symbols - can't trust capitalisation
            # in part because of convention to camel case animal homologous gene symbols
            return True
        else:
            return all(len(x) < 4 or StringNormalizer.is_probably_symbol_like(x) for x in tokens)
