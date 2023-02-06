import re
from os import getenv

import regex
from functools import lru_cache
from typing import Optional, Dict, Protocol, Type, Tuple

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

        tokens = original_string.split(" ")
        token_count = len(tokens)

        if token_count == 1 and len(original_string) <= 3:
            return True

        for i, char in enumerate(original_string):
            if char.isalpha():
                if char.isupper():
                    upper_count += 1
                    if i > 0 and token_count == 1:
                        # if is single token, and any char apart from first is upper, assume symbol
                        return True

                else:
                    lower_count += 1

            elif char.isnumeric():
                if token_count == 1:
                    # if is single token and has a number in it, assume symbol
                    return True
                numeric_count += 1

        if upper_count >= lower_count:
            return True
        elif numeric_count >= (upper_count + lower_count):
            return True
        else:
            return False

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        return " ".join(original_string.upper().split())

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        string = DefaultStringNormalizer.replace_substrings(original_string)
        string = DefaultStringNormalizer.split_on_numbers(string)
        string = DefaultStringNormalizer.replace_greek(string)
        string = DefaultStringNormalizer.remove_non_alphanum(string)
        string = DefaultStringNormalizer.depluralize(string)
        string = DefaultStringNormalizer.sub_greek_char_abbreviations(string)
        return " ".join(string.upper().split())

    @staticmethod
    def depluralize(string):
        """
        apply some depluralisation rules

        :param string:
        :return:
        """
        if len(string) > 3:
            string = GildaUtils.depluralize(string)[0]
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


class DiseaseStringNormalizer(EntityClassNormalizer):
    known_disease_short_nouns = {"flu", "Flu", "HIV", "STI", "NSCLC"}

    @staticmethod
    def is_symbol_like(original_string: str) -> bool:
        if original_string in DiseaseStringNormalizer.known_disease_short_nouns:
            return False
        else:
            return DefaultStringNormalizer.is_symbol_like(original_string)

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_symbol

        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_symbol(original_string)

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_noun_phrase
        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_noun_phrase(original_string)


class AnatomyStringNormalizer(EntityClassNormalizer):
    @staticmethod
    def is_symbol_like(original_string: str) -> bool:
        # anatomy tends not to have symbolic representations
        return False

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_noun_phrase (note, since
        all anatomy is non-symbolic, this is theoretically superfluous, but we
        include it anyway)

        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_noun_phrase(original_string)

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_noun_phrase
        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_noun_phrase(original_string)


class GeneStringNormalizer(EntityClassNormalizer):
    gene_name_suffixes = {"in", "ase", "an", "gen", "gon"}

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
        string = GildaUtils.replace_dashes(original_string, " ")
        tokens = string.split(" ")
        if len(tokens) == 1 and not any(
            tokens[0].endswith(suffix) for suffix in GeneStringNormalizer.gene_name_suffixes
        ):
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
        if upper_count >= lower_count:
            return True
        elif numeric_count >= (upper_count + lower_count):
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
        return " ".join(string.upper().split())

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_noun_phrase for non symbolic genes

        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_noun_phrase(original_string)


class CompanyStringNormalizer(EntityClassNormalizer):
    @staticmethod
    def is_symbol_like(original_string: str) -> bool:
        alpha_chars = filter(lambda x: x.isalpha(), original_string)
        return all(x.isupper() for x in alpha_chars)

    @staticmethod
    def normalize_symbol(original_string: str) -> str:
        """
        just upper case.

        :param original_string:
        :return:
        """
        return original_string.upper()

    @staticmethod
    def normalize_noun_phrase(original_string: str) -> str:
        """
        revert to DefaultStringNormalizer.normalize_noun_phrase
        :param original_string:
        :return:
        """
        return DefaultStringNormalizer.normalize_noun_phrase(original_string)


class StringNormalizer:
    """
    call custom entity class normalizers, or a default normalizer if none is available
    """

    normalizers: Dict[Optional[str], Type[EntityClassNormalizer]] = {
        "gene": GeneStringNormalizer,
        "anatomy": AnatomyStringNormalizer,
        "disease": DiseaseStringNormalizer,
        "company": CompanyStringNormalizer,
    }

    @staticmethod
    @lru_cache(maxsize=int(getenv("KAZU_STRING_NORMALIZER_CACHE_SIZE", 5000)))
    def classify_symbolic(original_string: str, entity_class: Optional[str] = None) -> bool:
        return StringNormalizer.normalizers.get(
            entity_class, DefaultStringNormalizer
        ).is_symbol_like(original_string)

    @staticmethod
    @lru_cache(maxsize=int(getenv("KAZU_STRING_NORMALIZER_CACHE_SIZE", 5000)))
    def normalize(original_string: str, entity_class: Optional[str] = None) -> str:
        normaliser_for_entity_class = StringNormalizer.normalizers.get(
            entity_class, DefaultStringNormalizer
        )
        if normaliser_for_entity_class.is_symbol_like(original_string):
            return normaliser_for_entity_class.normalize_symbol(original_string)
        else:
            return normaliser_for_entity_class.normalize_noun_phrase(original_string)


class GildaUtils:
    """
    Original Credit:
    https://github.com/indralab/gilda
    https://github.com/indralab/gilda/blob/9e383213098144fe82103a3a5aa1bf4c14059e57/gilda/process.py

    @article{gyori2022gilda,
        author = {Gyori, Benjamin M and Hoyt, Charles Tapley and Steppi, Albert},
        title = "{{Gilda: biomedical entity text normalization with machine-learned disambiguation as a service}}",
        journal = {Bioinformatics Advances},
        year = {2022},
        month = {05},
        issn = {2635-0041},
        doi = {10.1093/bioadv/vbac034},
        url = {https://doi.org/10.1093/bioadv/vbac034},
        note = {vbac034}
    }


    BSD 2-Clause License

    Copyright (c) 2019, Benjamin M. Gyori, Harvard Medical School
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """

    dashes = [chr(0x2212), chr(0x002D)] + [chr(c) for c in range(0x2010, 0x2016)]

    @staticmethod
    def depluralize(word: str) -> Tuple[str, str]:
        """
        Return the depluralized version of the word, along with a status flag.

        :param word: The word which is to be depluralized.
        :return: The first element is the original word, if it is detected to be non-plural, or the
            depluralized version of the word.

            The second element is a status flag representing the detected pluralization status of
            the word, with non_plural (e.g., BRAF), plural_oes (e.g., mosquitoes),
            plural_ies (e.g., antibodies), plural_es (e.g., switches),
            plural_cap_s (e.g., MAPKs), and plural_s (e.g., receptors).
        """
        # If the word doesn't end in s, we assume it's not plural
        if not word.endswith("s"):
            return word, "non_plural"
        # Another case is words ending in -sis (e.g., apoptosis), these are almost
        # exclusively non plural so we return here too
        elif word.endswith("sis"):
            return word, "non_plural"
        # This is the case when the word ends with an o which is pluralized as oes
        # e.g., mosquitoes
        elif word.endswith("oes"):
            return word[:-2], "plural_oes"
        # This is the case when the word ends with a y which is pluralized as ies,
        # e.g., antibodies
        elif word.endswith("ies"):
            return word[:-3] + "y", "plural_ies"
        # These are the cases where words form plurals by adding -es so we
        # return by stripping it off
        elif word.endswith(("xes", "ses", "ches", "shes")):
            return word[:-2], "plural_es"
        # If the word is all caps and the last letter is an s, then it's a very
        # strong signal that it is pluralized so we have a custom return value
        # for that
        elif regex.match(r"^\p{Lu}+$", word[:-1]):
            return word[:-1], "plural_caps_s"
        # Otherwise, we just go with the assumption that the last s is the
        # plural marker
        else:
            return word[:-1], "plural_s"
        # Note: there don't seem to be any compelling examples of -f or -fe -> ves
        # so it is not implemented

    @classmethod
    def replace_dashes(cls, s: str, rep: str = "-") -> str:
        """Replace all types of dashes in a given string with a given replacement.

        :param s: The string in which all types of dashes should be replaced.
        :param rep: The string with which dashes should be replaced. By default, the plain
            ASCII dash (-) is used.
        :return: The string in which dashes have been replaced.
        """
        for d in cls.dashes:
            s = s.replace(d, rep)
        return s
