import copy
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
    is_symbol: bool = False
    ner_blacklist: bool = False
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
            metadata = copy.copy(metadata)
            generated_syn_dict: Optional[Dict[str, List[SynonymData]]] = self.call(
                synonym, metadata
            )
            if generated_syn_dict:
                for generated_syn, generated_metadata in generated_syn_dict.items():
                    if generated_syn in syn_data:
                        logger.debug(
                            f"generated synonym '{generated_syn}' matches existing synonym {syn_data[generated_syn]} "
                        )
                    elif generated_syn in result:
                        logger.debug(
                            f"generated synonym '{generated_syn}' matches another generated synonym {result[generated_syn]} "
                        )
                    else:
                        for y in generated_metadata:
                            y.generated_from.append(self.__class__.__name__)
                        result[generated_syn] = generated_metadata
        return result


class SeperatorExpansion(SynonymGenerator):
    def __init__(self):
        self.paren_re = r"(.*)\((.*)\)(.*)"
        self.excluded_parenthesis = ["", "non-protein coding"]

    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
        results = {}
        if "(" in text and ")" in text:
            # expand brackets
            matches = re.match(self.paren_re, text)
            if matches is not None:
                all_groups_no_brackets = []
                for group in matches.groups():
                    if group not in self.excluded_parenthesis:
                        results[group] = syn_data
                        all_groups_no_brackets.append(group)
                results["".join(all_groups_no_brackets)] = syn_data

        # expand slashes
        for x in list(results.keys()):
            if "/" in x:
                splits = x.split("/")
                for split in splits:
                    results[split] = syn_data
        return results


class CaseModifier(SynonymGenerator):
    def call(
        self, text: str, syn_data: List[SynonymData]
    ) -> Optional[Dict[str, List[SynonymData]]]:
        results = {}
        results[text.lower()] = syn_data
        results[text.upper()] = syn_data
        results[text.title()] = syn_data
        return results


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
        return results

    def substitute(self, text: str, replace_dict: Dict[str, str]) -> str:
        chars_found = filter(lambda x: x in text, replace_dict.keys())
        for greek_unicode in chars_found:
            text = text.replace(greek_unicode, replace_dict[greek_unicode])
            text = self.substitute(text, replace_dict)
        return text
