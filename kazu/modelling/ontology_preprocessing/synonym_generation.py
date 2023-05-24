import dataclasses
import itertools
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Optional, Iterable, Set

from kazu.data.data import SynonymTerm, EquivalentIdAggregationStrategy
from kazu.modelling.language.language_phenomena import GREEK_SUBS, DASHES
from kazu.utils.utils import PathLike

import spacy

logger = logging.getLogger(__name__)


class SynonymGenerator(ABC):
    @abstractmethod
    def call(self, synonym_str: str) -> Optional[Set[str]]:
        pass

    def __call__(self, synonym_terms: Set[SynonymTerm]) -> Set[SynonymTerm]:

        existing_terms = set(term for synonym in synonym_terms for term in synonym.terms)

        result: Set[SynonymTerm] = set()
        for synonym_term in synonym_terms:
            for synonym_str in synonym_term.terms:
                generated_synonym_strings = self.call(synonym_str)
                if generated_synonym_strings:
                    new_terms: Set[str] = set()
                    for generated_syn in generated_synonym_strings:
                        if generated_syn in existing_terms:
                            logger.debug(
                                f"generated synonym '{generated_syn}' matches existing synonym {synonym_term} "
                            )
                        elif generated_syn in new_terms:
                            logger.debug(
                                f"generated synonym '{generated_syn}' matches another generated synonym {synonym_term} "
                            )
                        else:
                            new_terms.add(generated_syn)
                    if new_terms:
                        if synonym_term.original_term is not None:
                            orig_term = synonym_term.original_term
                        else:
                            orig_term = synonym_str
                        result.add(
                            dataclasses.replace(
                                synonym_term,
                                original_term=orig_term,
                                terms=frozenset(new_terms),
                                aggregated_by=EquivalentIdAggregationStrategy.CUSTOM,
                            )
                        )

        return result


class CombinatorialSynonymGenerator:
    def __init__(self, synonym_generators: Iterable[SynonymGenerator]):
        self.synonym_generators: Set[SynonymGenerator] = set(synonym_generators)

    def __call__(self, synonyms: Set[SynonymTerm]) -> Set[SynonymTerm]:
        """
        For every permutation of modifiers, generate a list of syns, then aggregate at the end

        :param synonyms: Set[SynonymTerm] to generate from
        :return: Set[SynonymTerm] of generated synonyms. Note, the field values of the generated synonyms will
            be the same as the seed synonym, apart from SynonymTerm.terms, which contains the generated synonyms
        """
        synonym_gen_permutations = itertools.permutations(self.synonym_generators)
        results = set()
        for i, permutation_list in enumerate(synonym_gen_permutations):
            # make a copy of the original synonyms
            all_syns = deepcopy(synonyms)
            logger.info(f"running permutation set {i}. Permutations: {permutation_list}")
            for generator in permutation_list:
                # run the generator
                new_syns: Set[SynonymTerm] = generator(all_syns)
                for new_syn_term in new_syns:
                    results.add(new_syn_term)
                    all_syns.add(new_syn_term)

        return results


# TODO: this isn't used currently - do we want to try and refine it
# or just kill it off altogether?
class SeparatorExpansion(SynonymGenerator):
    def __init__(self, spacy_pipeline: spacy.Language):
        self.all_stopwords = spacy_pipeline.Defaults.stop_words
        self.end_expression_brackets = r"(.*)\((.*)\)$"
        self.mid_expression_brackets = r"(.*)\(.*\)(.*)"
        self.excluded_parenthesis = ["", "non-protein coding"]

    def call(self, synonym_str: str) -> Optional[Set[str]]:
        bracket_results = set()
        all_group_results = set()
        if "(" in synonym_str and ")" in synonym_str:
            # expand end expression brackets
            matches = re.match(self.end_expression_brackets, synonym_str)
            if matches is not None:
                all_groups_no_brackets = []
                for group in matches.groups():
                    if (
                        group not in self.excluded_parenthesis
                        and group.lower() not in self.all_stopwords
                    ):
                        bracket_results.add(group.strip())
                        all_groups_no_brackets.append(group)
                all_group_results.add("".join(all_groups_no_brackets))
            else:
                # remove mid expression  brackets
                matches = re.match(self.mid_expression_brackets, synonym_str)
                if matches is not None:
                    all_groups_no_brackets = []
                    for group in matches.groups():
                        all_groups_no_brackets.append(group.strip())
                    all_group_results.add(" ".join(all_groups_no_brackets))

        # expand slashes
        for x in list(bracket_results):
            if "/" in x:
                splits = x.split("/")
                for split in splits:
                    bracket_results.add(split.strip())
            if "," in x:
                splits = x.split(",")
                for split in splits:
                    bracket_results.add(split.strip())
        bracket_results.update(all_group_results)
        if len(bracket_results) > 0:
            return bracket_results
        else:
            return None


class StopWordRemover(SynonymGenerator):
    """
    remove stopwords from a string
    """

    all_stopwords = {"of", "and", "in", "to", "with", "caused", "involved", "by", "the"}

    @classmethod
    def call(cls, synonym_str: str) -> Optional[Set[str]]:
        new_terms = set()
        lst = []
        detected = False
        for token in synonym_str.split():
            if token.lower() in cls.all_stopwords:
                detected = True
            else:
                lst.append(token)
        if detected:
            new_terms.add(" ".join(lst))
        if new_terms:
            return new_terms
        else:
            return None


class GreekSymbolSubstitution:

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

    def call(self, synonym_str: str) -> Optional[Set[str]]:
        results = set()
        if self.replacement_dict:
            for to_replace, replacement_list in self.replacement_dict.items():
                if to_replace in synonym_str:
                    for replace_with in replacement_list:
                        results.add(synonym_str.replace(to_replace, replace_with).strip())
        if self.digit_aware_replacement_dict:
            for to_replace, replacement_list in self.digit_aware_replacement_dict.items():
                matches = set(re.findall(to_replace + r"[0-9]+", synonym_str))
                for match in matches:
                    number = match.split(to_replace)[1]
                    for sub_in in replacement_list:
                        new_str = synonym_str.replace(match, f"{sub_in}{number}").strip()
                        results.add(new_str)

        if self.include_greek:
            # only strip text once initially - the greek character replacement
            # will not introduce leading or trailing whitespace unlike the other
            # replacements above
            stripped_text = synonym_str.strip()
            strings_to_substitute = {stripped_text}
            for to_replace, replacement_set in GreekSymbolSubstitution.ALL_SUBS.items():
                # if it's in the original text it should be in all previous substitutions, no
                # need to check all of them
                if to_replace in synonym_str:
                    # necessary so we don't modify strings_to_substitute while looping over it,
                    # which throws an error
                    outputs_this_step = set()
                    for string_to_subsitute in strings_to_substitute:
                        for replacement in replacement_set:
                            single_unique_letter_substituted = string_to_subsitute.replace(
                                to_replace, replacement
                            )
                            outputs_this_step.add(single_unique_letter_substituted)
                            results.add(single_unique_letter_substituted)
                    strings_to_substitute.update(outputs_this_step)

        if len(results) > 0:
            return results
        else:
            return None


class SuffixReplacement(SynonymGenerator):
    """Interchange all suffixes within a provided set to produce new synonyms.

    Note, this is expected to be noisy, and for most of the generated synonyms not to be valid
    words. This class is present as a generation step for high recall, with curation of synonyms
    expected later (see :ref:`curating_for_explosion`).

    In particular, note that this also doesn't check for the longest matching suffix - e.g. for a
    synonym 'anaemia' and the suffixes 'ia', 'a' and 'ic', the new synonyms 'anaemic' and
    'amaemiic' will both be generated.
    """

    def __init__(self, suffixes: Iterable[str]):
        self.suffixes = set(suffixes)

    def call(self, synonym_str: str) -> Optional[Set[str]]:
        new_terms: Set[str] = set()
        for suffix in self.suffixes:
            # Note that this will trigger twice for 'ia' since 'a' is also present.
            # We expect this to be noisy, and then curate from this.
            if synonym_str.endswith(suffix):
                term_without_suffix = synonym_str.removesuffix(suffix)
                new_terms.update(
                    term_without_suffix + new_suffix
                    for new_suffix in self.suffixes
                    if new_suffix is not suffix
                )

        if new_terms:
            return new_terms
        else:
            return None


class SpellingVariationReplacement(SynonymGenerator):
    """Generate additional synonyms using a mapping of (known) synonyms to a list of variations."""

    def __init__(self, input_path: PathLike):
        with open(input_path, mode="r", encoding="utf-8") as inf:
            raw_variation_mapping: Dict[str, List[str]] = json.load(inf)

        # lowercase for case-insensitive comparison
        self.variation_mapping = {k.lower(): val for k, val in raw_variation_mapping.items()}

    def call(self, synonym_str: str) -> Optional[Set[str]]:
        new_terms = set()
        variations = self.variation_mapping.get(synonym_str.lower())
        if variations is not None:
            new_terms.update(variations)
        if new_terms:
            return new_terms
        else:
            return None


class NgramHyphenation(SynonymGenerator):
    """Generate hyphenated variants of ngrams."""

    def __init__(self, ngram: int = 2):
        self.ngram = ngram

    def call(self, synonym_str: str) -> Optional[Set[str]]:
        parts = synonym_str.split()
        if len(parts) != self.ngram:
            return None
        else:
            new_terms = set()
            for hyphen in DASHES:
                new_terms.add(hyphen.join(parts))
        return new_terms
