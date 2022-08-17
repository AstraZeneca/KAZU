import re
from typing import Optional

from gilda.process import depluralize
from kazu.modelling.language.language_phenomena import GREEK_SUBS


class StringNormalizer:
    """
    normalise a biomedical string for search
    TODO: make configurable
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

    @classmethod
    def is_symbol_like(cls, debug, original_string) -> Optional[str]:
        # TODO: rename method
        # True if all upper, all alphanum, no spaces,

        for char in original_string:
            if char.islower() or not char.isalnum():
                return None
        else:
            splits = [x.strip() for x in re.split(cls.symbol_number_split, original_string)]
            string = " ".join(splits).strip()
            if debug:
                print(string)
            return string

    @staticmethod
    def is_probably_symbol_like(original_string: str) -> bool:
        """
        a more forgiving version of is_symbol_like, designed to improve symbol recall on natural text
        looks at the ratio of upper case to lower case chars, and the ratio of integer to alpha chars. If the ratio of
        upper case or integers is higher, assume it's a symbol
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

    @classmethod
    def normalize(cls, original_string: str, debug: bool = False):
        original_string = original_string.strip()
        symbol_like = cls.is_symbol_like(debug, original_string)
        if symbol_like:
            return symbol_like
        else:
            string = cls.replace_substrings(debug, original_string)

            # split up numbers
            string = cls.split_on_numbers(debug, string)
            # replace greek
            string = cls.replace_greek(debug, string)

            # strip non alphanum
            string = cls.replace_non_alphanum(debug, string)

            string = cls.depluralize(debug, string)
            # strip modifying lowercase prefixes
            string = cls.handle_lower_case_prefixes(debug, string)

            string = cls.sub_greek_char_abbreviations(debug, string)

            string = string.strip()
            if debug:
                print(string)
            return string

    @staticmethod
    def depluralize(debug, string):
        if len(string) > 3:
            string = depluralize(string)[0]
        if debug:
            print(string)
        return string

    @classmethod
    def sub_greek_char_abbreviations(cls, debug, string):
        for re_sub, replace in cls.re_subs_2.items():
            string = re.sub(re_sub, replace, string)
            if debug:
                print(string)
        return string

    @staticmethod
    def to_upper(debug, string):
        string = string.upper()
        if debug:
            print(string)
        return string

    @staticmethod
    def handle_lower_case_prefixes(debug, string):
        """
        preserve case only if first char of contiguous subsequence is lower case, and is alphanum, and upper
        case detected in rest of part
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
        if debug:
            print(string)
        return string

    @classmethod
    def replace_non_alphanum(cls, debug, string):
        string = "".join([x for x in string if (x.isalnum() or x in cls.allowed_additional_chars)])
        if debug:
            print(string)
        return string

    @classmethod
    def replace_greek(cls, debug, string):
        for substr, replace in cls.greek_subs_upper.items():
            if substr in string:
                string = string.replace(substr, replace)
                if debug:
                    print(string)
        return string

    @classmethod
    def split_on_numbers(cls, debug, string):
        splits = [x.strip() for x in re.split(cls.number_split_pattern, string)]
        string = " ".join(splits)
        if debug:
            print(string)
        return string

    @classmethod
    def replace_substrings(cls, debug, original_string):
        string = original_string
        # replace substrings
        for substr, replace in cls.other_subs.items():
            if substr in string:
                string = string.replace(substr, replace)
                if debug:
                    print(string)
        for re_sub, replace in cls.re_subs.items():
            string = re.sub(re_sub, replace, string)
            if debug:
                print(string)
        return string
