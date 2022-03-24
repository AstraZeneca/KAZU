import copy
import itertools
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable

from kazu.utils.spacy_pipeline import SpacyPipeline

logger = logging.getLogger(__name__)


@dataclass
class SynonymData:
    """
    data class required by DictionaryIndex add method. See docs on :py:class:`kazu.utils.link_index.DictionaryIndex`
    for usage
    """

    idx: str
    mapping_type: List[str]
    generated_from: List[str] = field(default_factory=list, hash=False)


class SynonymGenerator(ABC):
    @abstractmethod
    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
        pass

    def __call__(self, syn_data: Dict[str, List[SynonymData]]) -> Dict[str, List[SynonymData]]:

        result: Dict[str, List[SynonymData]] = {}
        for synonym, metadata in syn_data.items():
            metadata_copy = copy.copy(metadata)
            generated_syn_dict: Optional[Dict[str, List[SynonymData]]] = self.call(
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
                for syn_info in metadata_copy:
                    syn_info.generated_from.append(self.__class__.__name__)
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

    def __call__(self, synonym_data: Dict[str, List[SynonymData]]) -> Dict[str, List[SynonymData]]:
        """
        for every perumation of modifiers, generate a list of syns, then aggregate at the end
        :param synonym_data:
        :return:
        """
        final = {}
        if self.synonym_generator_permutations:
            for i, permutation_list in enumerate(self.synonym_generator_permutations):
                logger.info(f"running permutation set {i}. Permutations: {permutation_list}")
                generated_synonym_data = copy.deepcopy(synonym_data)
                for generator in permutation_list:
                    generated_synonym_data = generator(generated_synonym_data)
                final.update(generated_synonym_data)
        return final


class SeparatorExpansion(SynonymGenerator):
    def __init__(self, spacy_pipeline: SpacyPipeline):
        self.all_stopwords = spacy_pipeline.nlp.Defaults.stop_words
        self.paren_re = r"(.*)\((.*)\)(.*)"
        self.excluded_parenthesis = ["", "non-protein coding"]

    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
        bracket_results = {}
        all_group_results = {}
        if "(" in text and ")" in text:
            # expand brackets
            matches = re.match(self.paren_re, text)
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

        # expand slashes
        for x in list(bracket_results.keys()):
            if "/" in x:
                splits = x.split("/")
                for split in splits:
                    bracket_results[split] = syn_data
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

    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
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
        self.all_stopwords = spacy_pipeline.nlp.Defaults.stop_words

    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
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


class StringReplacement(SynonymGenerator):
    def __init__(self, replacement_dict: Dict[str, List[str]]):
        self.replacement_dict = replacement_dict

    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
        results = {}
        for to_replace, replacement_list in self.replacement_dict.items():
            if to_replace in text:
                for replace_with in replacement_list:
                    results[text.replace(to_replace, replace_with).strip()] = syn_data
        if len(results) > 0:
            return results
        else:
            return None


class GreekSymbolSubstitution(SynonymGenerator):
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
        "\u039B": "lamda",
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
        "\u03B2": "beta",
        "\u03B3": "gamma",
        "\u03B4": "delta",
        "\u03B5": "epsilon",
        "\u03B6": "zeta",
        "\u03B7": "eta",
        "\u03B8": "theta",
        "\u03B9": "iota",
        "\u03BA": "kappa",
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
        "\u03C7": "chi",
        "\u03C8": "psi",
        "\u03C9": "omega",
    }

    GREEK_SUBS_ABBRV = {k: v[0] for k, v in GREEK_SUBS.items()}
    GREEK_SUBS_REVERSED = {v: k for k, v in GREEK_SUBS.items()}

    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
        results = {}
        results[self.substitute(text, self.GREEK_SUBS)] = syn_data
        results[self.substitute(text, self.GREEK_SUBS_REVERSED)] = syn_data
        results[self.substitute(text, self.GREEK_SUBS_ABBRV)] = syn_data
        if len(results) > 0:
            return results
        else:
            return None

    def substitute(self, text: str, replace_dict: Dict[str, str]) -> str:
        chars_found = filter(lambda x: x in text, replace_dict.keys())
        for greek_unicode in chars_found:
            text = text.replace(greek_unicode, replace_dict[greek_unicode])
            text = self.substitute(text, replace_dict)
        return text
