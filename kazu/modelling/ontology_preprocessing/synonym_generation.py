import copy
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Optional, Iterable, Set

from kazu.data.data import SynonymData
from kazu.utils.spacy_pipeline import SpacyPipeline

logger = logging.getLogger(__name__)


class SynonymGenerator(ABC):
    @abstractmethod
    def call(self, text: str, syn_data: Set[SynonymData]) -> Optional[Dict[str, Set[SynonymData]]]:
        pass

    def __call__(self, syn_data: Dict[str, Set[SynonymData]]) -> Dict[str, Set[SynonymData]]:

        result: Dict[str, Set[SynonymData]] = {}
        for synonym, metadata in syn_data.items():
            metadata_copy = copy.copy(metadata)
            generated_syn_dict: Optional[Dict[str, Set[SynonymData]]] = self.call(
                synonym, metadata_copy
            )
            if generated_syn_dict:
                for generated_syn in generated_syn_dict:
                    if generated_syn in syn_data:
                        logger.debug(
                            f"generated synonym '{generated_syn}' matches existing synonym {syn_data[generated_syn]} "
                        )
                    elif generated_syn in result:
                        logger.debug(
                            f"generated synonym '{generated_syn}' matches another generated synonym {result[generated_syn]} "
                        )
                    else:
                        result[generated_syn] = metadata_copy
        return result


class CombinatorialSynonymGenerator:
    def __init__(self, synonym_generators: List[SynonymGenerator]):
        self.synonym_generator_permutations = self.get_synonym_generator_permutations(
            synonym_generators
        )

    def get_synonym_generator_permutations(
        self, synonym_generators: List[SynonymGenerator]
    ) -> List[Iterable[SynonymGenerator]]:
        result: List[Iterable[SynonymGenerator]] = []
        for i in range(len(synonym_generators)):
            result.extend(itertools.permutations(synonym_generators, i + 1))

        return result

    def __call__(self, synonym_data: Dict[str, Set[SynonymData]]) -> Dict[str, Set[SynonymData]]:
        """
        for every permutation of modifiers, generate a list of syns, then aggregate at the end
        :param synonym_data:
        :return:
        """
        # make a copy of the original synonyms
        all_syns = copy.deepcopy(synonym_data)
        results = defaultdict(set)
        if self.synonym_generator_permutations:
            for i, permutation_list in enumerate(self.synonym_generator_permutations):
                logger.info(f"running permutation set {i}. Permutations: {permutation_list}")
                for generator in permutation_list:
                    # run the generator
                    new_syns = generator(all_syns)
                    for new_syn, syn_data_set in new_syns.items():
                        # don't add if it maps to a clean syn
                        if new_syn not in synonym_data:
                            results[new_syn].update(syn_data_set)

        final = {k: set(v) for k, v in results.items()}
        return final


class SeparatorExpansion(SynonymGenerator):
    def __init__(self, spacy_pipeline: SpacyPipeline):
        self.all_stopwords = spacy_pipeline.nlp.Defaults.stop_words
        self.end_expression_brackets = r"(.*)\((.*)\)$"
        self.mid_expression_brackets = r"(.*)\(.*\)(.*)"
        self.excluded_parenthesis = ["", "non-protein coding"]

    def call(self, text: str, syn_data: Set[SynonymData]) -> Optional[Dict[str, Set[SynonymData]]]:
        bracket_results = {}
        all_group_results = {}
        if "(" in text and ")" in text:
            # expand end expression brackets
            matches = re.match(self.end_expression_brackets, text)
            if matches is not None:
                all_groups_no_brackets = []
                for group in matches.groups():
                    if (
                        group not in self.excluded_parenthesis
                        and group.lower() not in self.all_stopwords
                    ):
                        bracket_results[group.strip()] = syn_data
                        all_groups_no_brackets.append(group)
                all_group_results["".join(all_groups_no_brackets)] = syn_data
            else:
                # remove mid expression  brackets
                matches = re.match(self.mid_expression_brackets, text)
                if matches is not None:
                    all_groups_no_brackets = []
                    for group in matches.groups():
                        all_groups_no_brackets.append(group.strip())
                    all_group_results[" ".join(all_groups_no_brackets)] = syn_data

        # expand slashes
        for x in list(bracket_results.keys()):
            if "/" in x:
                splits = x.split("/")
                for split in splits:
                    bracket_results[split.strip()] = syn_data
            if "," in x:
                splits = x.split(",")
                for split in splits:
                    bracket_results[split.strip()] = syn_data
        bracket_results.update(all_group_results)
        if len(bracket_results) > 0:
            return bracket_results
        else:
            return None


class CaseModifier(SynonymGenerator):
    def __init__(self, upper: bool = False, lower: bool = False, title: bool = False):
        self.title = title
        self.lower = lower
        self.upper = upper

    def call(self, text: str, syn_data: Set[SynonymData]) -> Optional[Dict[str, Set[SynonymData]]]:
        results = {}
        if self.upper and not text.isupper():
            results[text.upper()] = syn_data
        if self.lower and not text.islower():
            results[text.lower()] = syn_data
        if self.title and not text.istitle():
            results[text.title()] = syn_data
        if len(results) > 0:
            return results
        else:
            return None


class StopWordRemover(SynonymGenerator):
    """
    remove stopwords from a string
    """

    def __init__(self, spacy_pipeline: SpacyPipeline):
        self.all_stopwords = copy.deepcopy(spacy_pipeline.nlp.Defaults.stop_words)
        # treating "i" as a stop word means stripping trailing roman numeral 1
        # e.g. in 'grade I' which gives the incorrect synonym 'grade' when using this
        # for NER
        self.all_stopwords.remove("i")

    def call(self, text: str, syn_data: Set[SynonymData]) -> Optional[Dict[str, Set[SynonymData]]]:
        lst = []
        detected = False
        for token in text.split():
            if token.lower() in self.all_stopwords:
                detected = True
            else:
                lst.append(token)
        if detected:
            return {" ".join(lst): syn_data}
        else:
            return None


class GreekSymbolSubstitution:
    GREEK_SUBS = {
        "\u0391": "alpha",
        "\u0392": "beta",
        "\u0393": "gamma",
        "\u0394": "delta",
        "\u0395": "epsilon",
        "\u0396": "zeta",
        "\u0397": "eta",
        "\u0398": "theta",
        "\u0399": "iota",
        "\u039A": "kappa",
        "\u039B": "lambda",
        "\u039C": "mu",
        "\u039D": "nu",
        "\u039E": "xi",
        "\u039F": "omicron",
        "\u03A0": "pi",
        "\u03A1": "rho",
        "\u03A3": "sigma",
        "\u03A4": "tau",
        "\u03A5": "upsilon",
        "\u03A6": "phi",
        "\u03A7": "chi",
        "\u03A8": "psi",
        "\u03A9": "omega",
        "\u03F4": "theta",
        "\u03B1": "alpha",
        "\u03D0": "beta",
        "\u03B2": "beta",
        "\u03B3": "gamma",
        "\u03B4": "delta",
        "\u03B5": "epsilon",
        "\u03B6": "zeta",
        "\u03B7": "eta",
        "\u03B8": "theta",
        "\u03B9": "iota",
        "\u03BA": "kappa",
        "\u03BB": "lambda",
        "\u03BC": "mu",
        "\u03BD": "nu",
        "\u03BE": "xi",
        "\u03BF": "omicron",
        "\u03C0": "pi",
        "\u03C1": "rho",
        "\u03C2": "final sigma",
        "\u03C3": "sigma",
        "\u03C4": "tau",
        "\u03C5": "upsilon",
        "\u03C6": "phi",
        "\u03D5": "phi",
        "\u03C7": "chi",
        "\u03C8": "psi",
        "\u03C9": "omega",
    }

    ALL_SUBS: Dict[str, Set[str]] = defaultdict(set)
    for greek_letter, spelling in GREEK_SUBS.items():

        ALL_SUBS[greek_letter].add(spelling)
        # single letter abbreviation
        ALL_SUBS[greek_letter].add(spelling[0])
        # reversed
        ALL_SUBS[spelling].add(greek_letter)

        # replace between lower and upper case
        # to generate synonyms
        if greek_letter.islower():
            upper_greek_letter = greek_letter.upper()
            ALL_SUBS[greek_letter].add(upper_greek_letter)
        elif greek_letter.isupper():
            lower_greek_letter = greek_letter.lower()
            ALL_SUBS[greek_letter].add(lower_greek_letter)

    # we don't want this to have the semantics of a defaultdict for missing lookups
    ALL_SUBS = dict(ALL_SUBS)


class StringReplacement(SynonymGenerator):
    def __init__(
        self,
        replacement_dict: Optional[Dict[str, List[str]]] = None,
        digit_aware_replacement_dict: Optional[Dict[str, List[str]]] = None,
        include_greek: bool = True,
    ):
        self.include_greek = include_greek
        self.replacement_dict = replacement_dict
        self.digit_aware_replacement_dict = digit_aware_replacement_dict

    def call(self, text: str, syn_data: Set[SynonymData]) -> Optional[Dict[str, Set[SynonymData]]]:
        results = {}
        if self.replacement_dict:
            for to_replace, replacement_list in self.replacement_dict.items():
                if to_replace in text:
                    for replace_with in replacement_list:
                        results[text.replace(to_replace, replace_with).strip()] = syn_data
        if self.digit_aware_replacement_dict:
            for to_replace, replacement_list in self.digit_aware_replacement_dict.items():
                matches = set(re.findall(to_replace + r"[0-9]+", text))
                for match in matches:
                    number = match.split(to_replace)[1]
                    for sub_in in replacement_list:
                        new_str = text.replace(match, f"{sub_in}{number}").strip()
                        results[new_str] = syn_data

        if self.include_greek:
            # only strip text once initially - the greek character replacement
            # will not introduce leading or trailing whitespace unlike the other
            # replacements above
            stripped_text = text.strip()
            strings_to_substitute = {stripped_text}
            for to_replace, replacement_set in GreekSymbolSubstitution.ALL_SUBS.items():
                # if it's in the original text it should be in all previous substitutions, no
                # need to check all of them
                if to_replace in text:
                    # necessary so we don't modify strings_to_substitute while looping over it,
                    # which throws an error
                    outputs_this_step = set()
                    for string_to_subsitute in strings_to_substitute:
                        for replacement in replacement_set:
                            single_unique_letter_substituted = string_to_subsitute.replace(
                                to_replace, replacement
                            )
                            outputs_this_step.add(single_unique_letter_substituted)
                            results[single_unique_letter_substituted] = syn_data
                    strings_to_substitute.update(outputs_this_step)

        if len(results) > 0:
            return results
        else:
            return None


#
# class GreekSymbolSubstitution():
#     GREEK_SUBS = {
#         "\u0391": "alpha",
#         "\u0392": "beta",
#         "\u0393": "gamma",
#         "\u0394": "delta",
#         "\u0395": "epsilon",
#         "\u0396": "zeta",
#         "\u0397": "eta",
#         "\u0398": "theta",
#         "\u0399": "iota",
#         "\u039A": "kappa",
#         "\u039B": "lamda",
#         "\u039C": "mu",
#         "\u039D": "nu",
#         "\u039E": "xi",
#         "\u039F": "omicron",
#         "\u03A0": "pi",
#         "\u03A1": "rho",
#         "\u03A3": "sigma",
#         "\u03A4": "tau",
#         "\u03A5": "upsilon",
#         "\u03A6": "phi",
#         "\u03A7": "chi",
#         "\u03A8": "psi",
#         "\u03A9": "omega",
#         "\u03F4": "theta",
#         "\u03B1": "alpha",
#         "\u03B2": "beta",
#         "\u03B3": "gamma",
#         "\u03B4": "delta",
#         "\u03B5": "epsilon",
#         "\u03B6": "zeta",
#         "\u03B7": "eta",
#         "\u03B8": "theta",
#         "\u03B9": "iota",
#         "\u03BA": "kappa",
#         "\u03BC": "mu",
#         "\u03BD": "nu",
#         "\u03BE": "xi",
#         "\u03BF": "omicron",
#         "\u03C0": "pi",
#         "\u03C1": "rho",
#         "\u03C2": "final sigma",
#         "\u03C3": "sigma",
#         "\u03C4": "tau",
#         "\u03C5": "upsilon",
#         "\u03C6": "phi",
#         "\u03C7": "chi",
#         "\u03C8": "psi",
#         "\u03C9": "omega",
#     }
#
#     GREEK_SUBS_ABBRV = {k: v[0] for k, v in GREEK_SUBS.items()}
#     GREEK_SUBS_REVERSED = {v: k for k, v in GREEK_SUBS.items()}
#     #
#     # def call(
#     #     self, text: str, syn_data: List[SynonymData]
#     # ) -> Optional[Dict[str, List[SynonymData]]]:
#     #     results = {}
#     #     results[self.substitute(text, self.GREEK_SUBS)] = syn_data
#     #     results[self.substitute(text, self.GREEK_SUBS_REVERSED)] = syn_data
#     #     results[self.substitute(text, self.GREEK_SUBS_ABBRV)] = syn_data
#     #     if len(results) > 0:
#     #         return results
#     #     else:
#     #         return None
#     #
#     # def substitute(self, text: str, replace_dict: Dict[str, str]) -> str:
#     #     chars_found = filter(lambda x: x in text, replace_dict.keys())
#     #     for greek_unicode in chars_found:
#     #         text = text.replace(greek_unicode, replace_dict[greek_unicode])
#     #         text = self.substitute(text, replace_dict)
#     #     return text
