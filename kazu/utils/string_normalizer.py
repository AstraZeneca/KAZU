import re
from functools import lru_cache
from typing import Optional, Dict, Protocol

from gilda.process import depluralize, replace_dashes

from kazu.modelling.language.language_phenomena import GREEK_SUBS


def natural_text_symbol_classifier(original_string: str) -> bool:
    """
    a symbol classifier that is designed to improve recall on natural text, especially gene symbols
    looks at the ratio of upper case to lower case chars, and the ratio of integer to alpha chars. If the ratio of
    upper case or integers is higher, assume it's a symbol. Also if the first char is lower case, and any
    subsequent characters are upper case, it's probably a symbol (e.g. erbB2)
    :param original_string:
    :return:
    """

    upper_count = 0
    lower_count = 0
    numeric_count = 0
    first_char_is_lower = False

    for i, char in enumerate(original_string):
        if char.isalpha():
            if char.isupper():
                upper_count += 1
                if first_char_is_lower:
                    return True
            else:
                lower_count += 1
                if i == 0:
                    first_char_is_lower = True

        elif char.isnumeric():
            numeric_count += 1

    if upper_count > lower_count:
        return True
    elif numeric_count > (upper_count + lower_count):
        return True
    else:
        return False


class EntityClassNormalizer(Protocol):
    """
    protocol describing methods a normalizer should implement
    """

    def is_symbol_like(self, original_string: str) -> bool:
        ...

    def normalize_symbol(self, original_string: str) -> str:
        ...

    def normalize_noun_phrase(self, original_string: str) -> str:
        ...


class DefaultStringNormalizer(EntityClassNormalizer):
    """
    normalize a biomedical string for search. Suitable for most use cases
    """

    allowed_additional_chars = {" ", "(", ")", "+", "-", "‐"}
    greek_subs = GREEK_SUBS
    greek_subs_upper = {x: f" {y.upper()} " for x, y in greek_subs.items()}
    other_subs = {
        "(": " (",
        ")": ") ",
        ",": " ",
        "/": " ",
        "VIII": " 8 ",
        "VII": " 7 ",
        "XII": " 12 ",
        "III": " 3 ",
        "VI": " 6 ",
        "IV": " 4 ",
        "IX": " 9 ",
        "XI": " 11 ",
        "II": " 2 ",
    }
    re_subs = {
        re.compile(r"(?<!\()-(?!\))"): " ",  # minus not in brackets
        re.compile(r"(?<!\()‐(?!\))"): " ",  # hyphen not in brackets
        re.compile(r"\sI\s|\sI$"): " 1 ",
        re.compile(r"\sV\s|\sV$"): " 5 ",
        re.compile(r"\sX\s|\sX$"): " 10 ",
    }
    re_subs_2 = {
        re.compile(r"\sA\s|\sA$|^A\s"): " ALPHA ",
        re.compile(r"\sB\s|\sB$|^B\s"): " BETA ",
    }

    number_split_pattern = re.compile(r"(\d+)")

    symbol_number_split = re.compile(r"(\d+)$")
    trailing_lowercase_s_split = re.compile(r"(.*)(s)$")

    def is_symbol_like(self, original_string: str) -> bool:
        # True if all upper, all alphanum, no spaces,
        for char in original_string:
            if char.islower() or not char.isalnum():
                return False
        else:
            return True

    def normalize_symbol(self, original_string: str) -> str:
        return original_string.strip()

    def normalize_noun_phrase(self, original_string: str) -> str:
        string = self.replace_substrings(original_string)
        # split up numbers
        string = self.split_on_numbers(string)
        # replace greek
        string = self.replace_greek(string)

        # strip non alphanum
        string = self.replace_non_alphanum(string)

        string = self.depluralize(string)

        string = self.sub_greek_char_abbreviations(string)

        string = string.strip()
        return string.upper()

    @staticmethod
    def depluralize(string):
        if len(string) > 3:
            string = depluralize(string)[0]
        return string

    @classmethod
    def sub_greek_char_abbreviations(cls, string):
        for re_sub, replace in cls.re_subs_2.items():
            string = re.sub(re_sub, replace, string)
        return string

    @staticmethod
    def to_upper(string):
        string = string.upper()
        return string

    @staticmethod
    def handle_lower_case_prefixes(string):
        """
        preserve case only if first char of contiguous subsequence is lower case, and is alphanum, and upper
        case detected in rest of part. Currently unused as it causes problems with normalisation of e.g. erbB2, which
        is a commonly used form of the symbol
        :param debug:
        :param string:
        :return:
        """
        parts = string.split(" ")
        new_parts = []
        for part in parts:
            if part != "":
                if part.islower() and not len(part) == 1:
                    new_parts.append(part.upper())
                else:
                    first_char_case = part[0].islower()
                    if (first_char_case and part[0].isalnum()) or (
                        first_char_case and len(part) == 1
                    ):
                        new_parts.append(part)
                    else:
                        new_parts.append(part.upper())
        string = " ".join(new_parts)
        return string

    @classmethod
    def replace_non_alphanum(cls, string):
        string = "".join(x for x in string if (x.isalnum() or x in cls.allowed_additional_chars))

        return string

    @classmethod
    def replace_greek(cls, string):
        for substr, replace in cls.greek_subs_upper.items():
            if substr in string:
                string = string.replace(substr, replace)
        return string

    @classmethod
    def split_on_numbers(cls, string):
        splits = [x.strip() for x in re.split(cls.number_split_pattern, string)]
        string = " ".join(splits)
        return string

    @classmethod
    def replace_substrings(cls, original_string):
        string = original_string
        # replace substrings
        for substr, replace in cls.other_subs.items():
            if substr in string:
                string = string.replace(substr, replace)
        for re_sub, replace in cls.re_subs.items():
            string = re.sub(re_sub, replace, string)
        return string


class GeneStringNormalizer(EntityClassNormalizer):
    """
    contrary to other entity classes,  gene symbols require special handling because of their highly unusual nature
    :param original_string:
    :param debug:
    :param entity_class:
    :return:
    """

    default_normalizer = DefaultStringNormalizer()

    def is_symbol_like(self, original_string: str) -> bool:
        string = replace_dashes(original_string, " ")
        tokens = string.split(" ")
        if len(tokens) == 1:
            # single tokens are likely gene symbols, or can be easily classified as gene symbols - can't trust
            # capitalisation e.g. mTOR, erbBB2, egfr vs EGFR, Insulin
            # in part because of convention to camel case animal homologous gene symbols
            return True
        else:
            return all(len(x) < 4 or natural_text_symbol_classifier(x) for x in tokens)

    def remove_training_lowercase_s(self, string):
        if string[-1] == "s":
            return string[:-1]
        else:
            return string

    def normalize_symbol(self, original_string: str, entity_class: Optional[str] = None) -> str:

        string = self.remove_training_lowercase_s(original_string)
        string = self.default_normalizer.replace_substrings(string)
        # split up numbers
        string = self.default_normalizer.split_on_numbers(string)
        # replace greek
        string = self.default_normalizer.replace_greek(string)

        # strip non alphanum
        string = self.default_normalizer.replace_non_alphanum(string)

        string = self.default_normalizer.sub_greek_char_abbreviations(string)

        string = string.strip()
        return string.upper()

    def normalize_noun_phrase(
        self, original_string: str, entity_class: Optional[str] = None
    ) -> str:
        return self.default_normalizer.normalize_noun_phrase(original_string)


class StringNormalizer:
    """
    call custom entity class normalizers, or a default normalizer if none is available
    """

    default_normalizer = DefaultStringNormalizer()
    normalizers: Dict[Optional[str], EntityClassNormalizer] = {"gene": GeneStringNormalizer()}

    @staticmethod
    @lru_cache(maxsize=5000)
    def classify_symbolic(original_string: str, entity_class: Optional[str] = None) -> bool:
        return StringNormalizer.normalizers.get(
            entity_class, StringNormalizer.default_normalizer
        ).is_symbol_like(original_string)

    @staticmethod
    @lru_cache(maxsize=5000)
    def normalize(original_string: str, entity_class: Optional[str] = None) -> str:
        if StringNormalizer.normalizers.get(
            entity_class, StringNormalizer.default_normalizer
        ).is_symbol_like(original_string):
            return StringNormalizer.normalizers.get(
                entity_class, StringNormalizer.default_normalizer
            ).normalize_symbol(original_string)
        else:
            return StringNormalizer.normalizers.get(
                entity_class, DefaultStringNormalizer()
            ).normalize_noun_phrase(original_string)
