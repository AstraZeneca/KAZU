import re
from functools import lru_cache
from typing import Optional, Dict, Protocol, Type

from gilda.process import depluralize, replace_dashes

from kazu.modelling.language.language_phenomena import GREEK_SUBS


class EntityClassNormalizer(Protocol):
    """
    protocol describing methods a normalizer should implement
    """

    @staticmethod
    def is_symbol_like(original_string: str) -> bool:
        """
        method to determine whether a string is a symbol (e.g. "AD") or a noun phrase (e.g. "Alzheimers Disease")

        :param original_string:
        :return:
        """
        ...

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        """
        method for normalising a symbol

        :param original_string:
        :return:
        """
        ...

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        """
        method for normalising a noun phrase

        :param original_string:
        :return:
        """
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

    @staticmethod
    def is_symbol_like(original_string: str) -> bool:
        """
        checks for ratio of upper to lower case characters, and numeric to alpha characters.

        :param original_string:
        :return:
        """
        upper_count = 0
        lower_count = 0
        numeric_count = 0

        for char in original_string:
            if char.isalpha():
                if char.isupper():
                    upper_count += 1
                else:
                    lower_count += 1

            elif char.isnumeric():
                numeric_count += 1

        if upper_count > lower_count:
            return True
        elif numeric_count > (upper_count + lower_count):
            return True
        else:
            return False

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        return original_string.upper().strip()

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        string = DefaultStringNormalizer.replace_substrings(original_string)
        string = DefaultStringNormalizer.split_on_numbers(string)
        string = DefaultStringNormalizer.replace_greek(string)
        string = DefaultStringNormalizer.remove_non_alphanum(string)
        string = DefaultStringNormalizer.depluralize(string)
        string = DefaultStringNormalizer.sub_greek_char_abbreviations(string)
        string = string.strip()
        return string.upper()

    @staticmethod
    def depluralize(string):
        """
        apply some depluralisation rules

        :param string:
        :return:
        """
        if len(string) > 3:
            string = depluralize(string)[0]
        return string

    @staticmethod
    def sub_greek_char_abbreviations(string):
        """
        substitute single characters for alphanumeric representation - e.g. A -> ALPHA B--> BETA

        :param string:
        :return:
        """
        for re_sub, replace in DefaultStringNormalizer.re_subs_2.items():
            string = re.sub(re_sub, replace, string)
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

    @staticmethod
    def remove_non_alphanum(string):
        """
        removes all non alphanumeric characters

        :param string:
        :return:
        """
        return "".join(
            x
            for x in string
            if (x.isalnum() or x in DefaultStringNormalizer.allowed_additional_chars)
        )

    @staticmethod
    def replace_greek(string):
        """
        replaces greek characters with string representation

        :param string:
        :return:
        """
        for substr, replace in DefaultStringNormalizer.greek_subs_upper.items():
            if substr in string:
                string = string.replace(substr, replace)
        return string

    @staticmethod
    def split_on_numbers(string):
        """
        splits a string on numbers, for consistency

        :param string:
        :return:
        """
        return " ".join(
            x.strip() for x in re.split(DefaultStringNormalizer.number_split_pattern, string)
        )

    @staticmethod
    def replace_substrings(original_string):
        """
        replaces a range of other strings that might be confusing to a classifier, such as roman numerals

        :param original_string:
        :return:
        """
        string = original_string
        for substr, replace in DefaultStringNormalizer.other_subs.items():
            if substr in string:
                string = string.replace(substr, replace)
        for re_sub, replace in DefaultStringNormalizer.re_subs.items():
            string = re.sub(re_sub, replace, string)
        return string


class GeneStringNormalizer(EntityClassNormalizer):
    @staticmethod
    def is_symbol_like(original_string: str) -> bool:
        """
        a symbol classifier that is designed to improve recall on natural text, especially gene symbols
        looks at the ratio of upper case to lower case chars, and the ratio of integer to alpha chars. If the ratio of
        upper case or integers is higher, assume it's a symbol. Also if the first char is lower case, and any
        subsequent characters are upper case, it's probably a symbol (e.g. erbB2)

        :param original_string:
        :return:
        """
        string = replace_dashes(original_string, " ")
        tokens = string.split(" ")
        if len(tokens) == 1:
            # single tokens are likely gene symbols, or can be easily classified as gene symbols - can't trust
            # capitalisation e.g. mTOR, erbBB2, egfr vs EGFR, Insulin
            # in part because of convention to camel case animal homologous gene symbols
            return True
        else:
            return all(len(x) < 4 or GeneStringNormalizer.gene_token_classifier(x) for x in tokens)

    @staticmethod
    def gene_token_classifier(original_string):
        """
        slightly modified version of DefaultStringNormalizer.is_symbol_like, designed to work on single tokens. Checks
        if the casing of the symbol changes from lower to upper (if so, is likely to be symbolic, e.g. erbB2)

        :param original_string:
        :return:
        """
        upper_count = 0
        lower_count = 0
        numeric_count = 0
        first_char_is_lower = len(original_string) > 0 and original_string[0].islower()
        for char in original_string:
            if char.isalpha():
                if char.isupper():
                    upper_count += 1
                    if first_char_is_lower:
                        return True
                else:
                    lower_count += 1
            elif char.isnumeric():
                numeric_count += 1
        if upper_count > lower_count:
            return True
        elif numeric_count > (upper_count + lower_count):
            return True
        else:
            return False

    @staticmethod
    def remove_trailing_s_if_otherwise_capitalised(string: str):
        """
        frustratingly, some gene symbols are pluralised like ERBBs. we can't jsut remove trailing s as this breaks
        genuine symbols like 'MDH-s' and 'GASP10ps'. So, we only strip the trailing 's' if the char before is upper
        case

        :param string:
        :return:
        """
        if len(string) >= 3 and string[-2].isupper():
            return string.removesuffix("s")
        else:
            return string

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        """
        contrary to other entity classes, gene symbols require special handling because of their highly unusual
        nature

        :param original_string:
        :return:
        """
        string = GeneStringNormalizer.remove_trailing_s_if_otherwise_capitalised(original_string)
        string = DefaultStringNormalizer.replace_substrings(string)
        string = DefaultStringNormalizer.split_on_numbers(string)
        string = DefaultStringNormalizer.replace_greek(string)
        string = DefaultStringNormalizer.remove_non_alphanum(string)
        string = DefaultStringNormalizer.sub_greek_char_abbreviations(string)
        string = string.strip()
        return string.upper()

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_noun_phrase for non symbolic genes

        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_noun_phrase(original_string)


class StringNormalizer:
    """
    call custom entity class normalizers, or a default normalizer if none is available
    """

    normalizers: Dict[Optional[str], Type[EntityClassNormalizer]] = {"gene": GeneStringNormalizer}

    @staticmethod
    @lru_cache(maxsize=5000)
    def classify_symbolic(original_string: str, entity_class: Optional[str] = None) -> bool:
        return StringNormalizer.normalizers.get(
            entity_class, DefaultStringNormalizer
        ).is_symbol_like(original_string)

    @staticmethod
    @lru_cache(maxsize=5000)
    def normalize(original_string: str, entity_class: Optional[str] = None) -> str:
        normaliser_for_entity_class = StringNormalizer.normalizers.get(
            entity_class, DefaultStringNormalizer
        )
        if normaliser_for_entity_class.is_symbol_like(original_string):
            return normaliser_for_entity_class.normalize_symbol(original_string)
        else:
            return normaliser_for_entity_class.normalize_noun_phrase(original_string)
