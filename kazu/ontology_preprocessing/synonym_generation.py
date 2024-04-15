import dataclasses
import functools
import itertools
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from kazu.data import OntologyStringResource, Synonym
from kazu.language.language_phenomena import GREEK_SUBS, DASHES
from kazu.utils.spacy_pipeline import SpacyPipelines, BASIC_PIPELINE_NAME, basic_spacy_pipeline
from kazu.utils.utils import PathLike
from spacy.matcher import Matcher
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SynonymGenerator(ABC):
    @abstractmethod
    def call(self, string_to_mutate: str) -> set[str]:
        """Implementations should override this method to generate new strings from an
        input string."""
        pass

    @functools.cache
    def __call__(self, string_to_mutate: str) -> set[str]:
        """Takes a string, and returns a set containing the original and generated
        strings.

        Caching prevents re-computation of redundant strings.

        :param string_to_mutate:
        :return:
        """

        return self.call(string_to_mutate)


class CombinatorialSynonymGenerator:
    """For every permutation of modifiers, generate a list of syns, then aggregate at
    the end."""

    def __init__(self, synonym_generators: Iterable[SynonymGenerator]):
        self.synonym_generators: set[SynonymGenerator] = set(synonym_generators)

    def __call__(
        self, ontology_resources: set[OntologyStringResource]
    ) -> set[OntologyStringResource]:
        """Takes a set of :class:`~.OntologyStringResource`\\s, and returns a new set of
        ``OntologyStringResource``\\s with generated synonyms added as
        `alternative_synonyms`.

        :param ontology_resources:
        :return:
        """
        synonym_gen_permutations = list(itertools.permutations(self.synonym_generators))
        logger.info(
            "Running synonym generation permutations. This may be slow at first, but will speed up as caching takes effect."
        )
        final_results: defaultdict[OntologyStringResource, set[Synonym]] = defaultdict(set)
        original_strings = {
            syn.text for resource in ontology_resources for syn in resource.active_ner_synonyms()
        }
        for i, permutation_list in enumerate(synonym_gen_permutations):
            logger.info(
                "running permutation set %s of %s. Permutations: %s",
                i + 1,
                len(synonym_gen_permutations),
                permutation_list,
            )
            generated_results: defaultdict[OntologyStringResource, set[Synonym]] = defaultdict(set)
            for generator in permutation_list:
                # run the generator. We call list here as we modify the original list
                for resource in tqdm(
                    ontology_resources,
                    desc=f"generating synonyms for {generator.__class__.__name__}",
                ):

                    for syn in list(
                        generated_results.get(resource, resource.active_ner_synonyms())
                    ):
                        new_strings = generator(syn.text)
                        for new_syn_text in new_strings:
                            if new_syn_text in original_strings:
                                logger.debug("ignoring pre-existing string: %s", new_syn_text)
                                continue
                            alternative_syn = Synonym(
                                text=new_syn_text,
                                case_sensitive=syn.case_sensitive,
                                mention_confidence=syn.mention_confidence,
                            )
                            generated_results[resource].add(alternative_syn)
            for resource in ontology_resources:
                final_results[resource].update(generated_results[resource])

        for generator in self.synonym_generators:
            generator.__call__.cache_clear()

        new_resources = {
            dataclasses.replace(resource, alternative_synonyms=frozenset(alternative_synonyms))
            for resource, alternative_synonyms in final_results.items()
        }

        return new_resources


# TODO: this isn't used currently - do we want to try and refine it
# or just kill it off altogether?
class SeparatorExpansion(SynonymGenerator):
    def __init__(self):
        self.all_stopwords = basic_spacy_pipeline().Defaults.stop_words
        self.end_expression_brackets = r"(.*)\((.*)\)$"
        self.mid_expression_brackets = r"(.*)\(.*\)(.*)"
        self.excluded_parenthesis = ["", "non-protein coding"]

    def call(self, synonym_str: str) -> set[str]:
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
        return bracket_results


class StopWordRemover(SynonymGenerator):
    """Remove stopwords from a string."""

    all_stopwords = {"of", "and", "in", "to", "with", "caused", "involved", "by", "the"}

    @classmethod
    def call(cls, synonym_str: str) -> set[str]:
        new_synoyms = set()
        lst = []
        detected = False
        for token in synonym_str.split():
            if token.lower() in cls.all_stopwords:
                detected = True
            else:
                lst.append(token)
        if detected:
            new_synoyms.add(" ".join(lst))
        return new_synoyms


class GreekSymbolSubstitution:

    ALL_SUBS: dict[str, set[str]] = defaultdict(set)
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
    GREEK_VARIANT_PREFIX_SUFFIX = DASHES.union(set(" "))

    def __init__(
        self,
        replacement_dict: Optional[dict[str, list[str]]] = None,
        digit_aware_replacement_dict: Optional[dict[str, list[str]]] = None,
        include_greek: bool = True,
    ):
        self.include_greek = include_greek
        self.replacement_dict = replacement_dict
        self.digit_aware_replacement_dict = digit_aware_replacement_dict

    def call(self, synonym_str: str) -> set[str]:
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
            self._generate_greek_subs(results, synonym_str)

        return results

    def _generate_greek_subs(self, results, synonym_str):
        # only strip text once initially - the greek character replacement
        # will not introduce leading or trailing whitespace unlike the other
        # replacements above
        stripped_text = synonym_str.strip()
        strings_to_substitute = {stripped_text}
        for candidate, replacement_set in GreekSymbolSubstitution.ALL_SUBS.items():
            # if it's in the original text it should be in all previous substitutions, no
            # need to check all of them
            for fix in self.GREEK_VARIANT_PREFIX_SUFFIX:
                prefix = False
                suffix = False
                # necessary so we don't modify strings_to_substitute while looping over it,
                # which throws an error
                outputs_this_step = set()
                if f"{fix}{candidate}" in synonym_str:
                    suffix = True
                if f"{candidate}{fix}" in synonym_str:
                    prefix = True

                for string_to_substitute in strings_to_substitute:
                    for replacement in replacement_set:
                        if prefix:
                            single_unique_letter_substituted = string_to_substitute.replace(
                                f"{candidate}{fix}", f"{replacement}{fix}"
                            )
                            outputs_this_step.add(single_unique_letter_substituted)
                            results.add(single_unique_letter_substituted)
                        if suffix:
                            single_unique_letter_substituted = string_to_substitute.replace(
                                f"{fix}{candidate}", f"{fix}{replacement}"
                            )
                            outputs_this_step.add(single_unique_letter_substituted)
                            results.add(single_unique_letter_substituted)
                strings_to_substitute.update(outputs_this_step)


class SuffixReplacement(SynonymGenerator):
    """Interchange all suffixes within a provided set to produce new synonyms.

    Note, this is expected to be noisy, and for most of the generated synonyms not to be valid
    words. This class is present as a generation step for high recall, with curation of synonyms
    expected later.

    In particular, note that this also doesn't check for the longest matching suffix - e.g. for a
    synonym 'anaemia' and the suffixes 'ia', 'a' and 'ic', the new synonyms 'anaemic' and
    'amaemiic' will both be generated.
    """

    def __init__(self, suffixes: Iterable[str]):
        self.suffixes = set(suffixes)

    def call(self, synonym_str: str) -> set[str]:
        new_synonyms: set[str] = set()
        for suffix in self.suffixes:
            # Note that this will trigger twice for 'ia' since 'a' is also present.
            # We expect this to be noisy, and then curate from this.
            if synonym_str.endswith(suffix):
                syn_without_suffix = synonym_str.removesuffix(suffix)
                new_synonyms.update(
                    syn_without_suffix + new_suffix
                    for new_suffix in self.suffixes
                    if new_suffix is not suffix
                )

        return new_synonyms


class SpellingVariationReplacement(SynonymGenerator):
    """Generate additional synonyms using a mapping of (known) synonyms to a list of
    variations."""

    def __init__(self, input_path: PathLike):
        with open(input_path, mode="r", encoding="utf-8") as inf:
            raw_variation_mapping: dict[str, list[str]] = json.load(inf)

        # lowercase for case-insensitive comparison
        self.variation_mapping = {k.lower(): val for k, val in raw_variation_mapping.items()}

    def call(self, synonym_str: str) -> set[str]:
        new_synonyms = set()
        variations = self.variation_mapping.get(synonym_str.lower())
        if variations is not None:
            new_synonyms.update(variations)
        return new_synonyms


class NgramHyphenation(SynonymGenerator):
    """Generate hyphenated variants of ngrams."""

    def __init__(self, ngram: int = 2):
        self.ngram = ngram

    def call(self, synonym_str: str) -> set[str]:
        new_synonyms: set[str] = set()
        parts = synonym_str.split()
        if len(parts) != self.ngram:
            return new_synonyms
        else:
            for hyphen in DASHES:
                new_synonyms.add(hyphen.join(parts))
        return new_synonyms


class TokenListReplacementGenerator(SynonymGenerator):
    """Given lists of tokens, generate an alternative string based upon a query token.

    .. define a sphinx 'replacement' so we can reuse this later as well

    .. |meddra_specific_syn_generation_comment| replace::
       It's mainly designed for ontologies like Meddra which stretch the definition of an
       entity somewhat, by incorporating verbs (e.g. "increase in AST").

    Note, this implementation is pretty basic, and only replaces one token at a time.
    |meddra_specific_syn_generation_comment|
    """

    def __init__(
        self,
        token_lists_to_consider: list[list[str]],
    ):
        """

        :param token_lists_to_consider: if any token from the sublist matches a query string, generate
            new strings based upon all tokens in this sublist.
        """

        SpacyPipelines().add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
        SpacyPipelines().add_reload_callback_func(BASIC_PIPELINE_NAME, self._init_token_matcher)
        self.token_lists_to_consider = token_lists_to_consider
        self._init_token_matcher()

    def _init_token_matcher(self) -> None:
        matcher = Matcher(SpacyPipelines.get_model(BASIC_PIPELINE_NAME).vocab)
        for i, token_list in enumerate(self.token_lists_to_consider):
            matcher.add(key=i, patterns=[[{"LOWER": {"IN": token_list}}]])
        self.token_matcher = matcher

    def call(self, synonym_str: str) -> set[str]:
        new_synonyms = set()
        doc = SpacyPipelines().process_single(text=synonym_str, model_name=BASIC_PIPELINE_NAME)
        matches = self.token_matcher(doc)
        if matches is not None:
            for match_id, match_start, match_end in matches:
                found_tokens = doc[match_start:match_end].text
                variant_list = self.token_lists_to_consider[match_id]
                for variant in variant_list:
                    new_synonyms.add(synonym_str.replace(found_tokens, variant))
        return new_synonyms


class VerbPhraseVariantGenerator(SynonymGenerator):
    """Generate alternative verb phrases based on a list of tense templates, and lemmas
    matched in a query.

    |meddra_specific_syn_generation_comment|
    """

    NOUN_PLACEHOLDER = "NOUN"
    VERB_PLACEHOLDER = "TARGET"

    def __init__(
        self,
        tense_templates: list[str],
        lemmas_to_consider: dict[str, list[str]],
        spacy_model_path: str,
    ):
        """

        :param tense_templates: template expressons to generate, for example:

            .. code-block:: python

                ["{NOUN} {TARGET}", "{TARGET} in {NOUN}"]

        :param lemmas_to_consider: a dict of verb lemmas to surface forms to generate, for example:

            .. code-block:: python

                {"increase": ["increasing", "increased"], "decrease": ["decreased", "decreasing"]}

        :param spacy_model_path: path to a serialised spaCy model - must have a lemmatizer component.
        """
        SpacyPipelines().add_from_path(spacy_model_path, spacy_model_path)
        SpacyPipelines().add_reload_callback_func(spacy_model_path, self._init_lemma_matcher)
        self.spacy_model_path = spacy_model_path
        self.tense_templates = tense_templates
        self.lemmas_to_consider = lemmas_to_consider
        self._init_lemma_matcher()

    def _init_lemma_matcher(self) -> None:
        matcher = Matcher(SpacyPipelines.get_model(self.spacy_model_path).vocab)
        matcher.add(key=0, patterns=[[{"LEMMA": {"IN": list(self.lemmas_to_consider.keys())}}]])
        self.lemma_matcher = matcher

    def _populate_lemma_template(
        self, template: str, lemma: str, surface_forms: list[str], noun: str
    ) -> Iterable[str]:
        for form in [lemma] + surface_forms:
            yield template.format(**{self.NOUN_PLACEHOLDER: noun, self.VERB_PLACEHOLDER: form})

    def call(self, synonym_str: str) -> set[str]:
        new_synonyms: set[str] = set()
        doc = SpacyPipelines().process_single(text=synonym_str, model_name=self.spacy_model_path)
        noun_matches = self.lemma_matcher(doc)
        if noun_matches is not None:
            for _match_id, match_start, _ in noun_matches:
                verb_lemma = None
                noun = []
                for i, tok in enumerate(doc):
                    if i == match_start:
                        verb_lemma = tok.lemma_
                    else:
                        noun.append(tok.text)
                if len(noun) > 0 and verb_lemma is not None:
                    noun_str = " ".join(noun)
                    surface_forms = self.lemmas_to_consider[verb_lemma]
                    for template in self.tense_templates:
                        new_synonyms.update(
                            self._populate_lemma_template(
                                template=template,
                                surface_forms=surface_forms,
                                noun=noun_str,
                                lemma=verb_lemma,
                            )
                        )
        return new_synonyms
