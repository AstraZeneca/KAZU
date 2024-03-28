import logging
from collections import defaultdict
from collections.abc import Iterable

import ahocorasick
from kazu.data import Document, Entity, MentionConfidence, LinkingMetrics, LinkingCandidate
from kazu.database.in_memory_db import SynonymDatabase, ParserName, NormalisedSynonymStr
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.steps import document_iterating_step, ParserDependentStep
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.grouping import sort_then_group
from kazu.utils.spacy_pipeline import SpacyPipelines, basic_spacy_pipeline, BASIC_PIPELINE_NAME

logger = logging.getLogger(__name__)

EntityClass = str
CaseSensitive = bool
EntityInfoToOntologyInfoMapping = defaultdict[
    tuple[EntityClass, MentionConfidence, CaseSensitive, NormalisedSynonymStr, str], set[ParserName]
]


class MemoryEfficientStringMatchingStep(ParserDependentStep):
    """A wrapper for the ahocorasick algorithm.

    In testing, this implementation is comparable in speed to a spaCy
    `PhraseMatcher <https://spacy.io/api/phrasematcher>`_\\,
    and uses a fraction of the memory. Since this implementation is unaware
    of NLP concepts such as tokenization, we backfill this capability by
    checking for word boundaries with a custom spaCy tokenizer.
    """

    def __init__(self, parsers: Iterable[OntologyParser]):
        super().__init__(parsers)
        self.parsers = parsers
        self.automaton = self._create_automaton()
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
        self.synonym_db = SynonymDatabase()

    @kazu_disk_cache.memoize(ignore={0})
    def _create_automaton(self) -> ahocorasick.Automaton:
        """Create `ahocorasick.Automaton
        <https://pyahocorasick.readthedocs.io/en/latest/#automaton-class>`_\\ s
        for parsers."""
        key_to_ontology_info: defaultdict[str, EntityInfoToOntologyInfoMapping] = defaultdict(
            lambda: defaultdict(set)
        )

        logger.info("Creating ahocorasick Automaton")
        for parser in self.parsers:
            parser_resources = parser.populate_databases(return_resources=True)
            if parser_resources is None:
                logger.warning(
                    "tried to create ahocorasick data for parser %s, but no ontology resources are available",
                    parser.name,
                )
                continue
            if len(parser_resources) == 0:
                logger.warning(
                    "tried to create ahocorasick data for parser %s, but no ontology resources were produced",
                    parser.name,
                )
                continue

            for resource in parser_resources:
                # a resource can have different syn_norms for different parsers,
                # since the string normalizer's output depends on the entity class.
                # Also, a synonym may exist in multiple LinkingCandidate.raw_synonyms
                syn_norm = resource.syn_norm_for_linking(parser.entity_class)
                for syn in resource.active_ner_synonyms():
                    entity_key = (
                        parser.entity_class,
                        syn.mention_confidence,
                        syn.case_sensitive,
                        syn_norm,
                        syn.text,
                    )
                    key_to_ontology_info[syn.text.lower()][entity_key].add(parser.name)

        if len(key_to_ontology_info) > 0:
            case_insensitive_automaton = ahocorasick.Automaton()
            for key, entity_to_ontology_info in key_to_ontology_info.items():
                case_insensitive_automaton.add_word(key, entity_to_ontology_info)
            case_insensitive_automaton.make_automaton()
        else:
            raise RuntimeError(
                f"No valid resources were available for {self.__class__.__name__}. Either remove this step "
                "from your pipeline or ensure at least one valid resource is generated from the chosen "
                "parsers."
            )

        return case_insensitive_automaton

    def _word_is_valid(
        self, start_char: int, end_char: int, starts: set[int], ends: set[int]
    ) -> bool:
        return start_char in starts and end_char in ends

    def _case_matches(self, actual_match: str, original_casing: str, case_sensitive: bool) -> bool:
        # no need to check the strings actually match if it's not case sensitive
        # as this is already done as part of the string matching, so we know that
        # actual_match.lower() == original_casing.lower()
        return (not case_sensitive) or actual_match == original_casing

    def _process_automaton(
        self,
        automaton: ahocorasick.Automaton,
        matchable_text: str,
        original_text: str,
        starts: set[int],
        ends: set[int],
    ) -> list[Entity]:
        entities = []
        for end_index, ontology_dict in automaton.iter(matchable_text):
            # all keys should have the same value for synonym text
            match_key = next(iter(ontology_dict.keys()))[-1]
            start_index = end_index - len(match_key) + 1
            matched_text = original_text[start_index : end_index + 1]
            if self._word_is_valid(start_index, end_index, starts, ends):

                for entity_class, entity_info_groups in sort_then_group(
                    ontology_dict.keys(), key_func=lambda x: x[0]
                ):
                    candidates: set[LinkingCandidate] = set()
                    confidences: defaultdict[str, set[MentionConfidence]] = defaultdict(set)
                    for entity_info in entity_info_groups:
                        (
                            _,
                            confidence,
                            case_sensitive,
                            syn_norm,
                            original_case,
                        ) = entity_info

                        # filter out cases where we've specified we only want exact matches, but we only have a lowercase match
                        if not self._case_matches(
                            actual_match=matched_text,
                            original_casing=original_case,
                            case_sensitive=case_sensitive,
                        ):
                            continue

                        parser_name_set = ontology_dict[entity_info]

                        for parser_name in parser_name_set:
                            confidences[parser_name].add(confidence)
                            candidates.add(self.synonym_db.get(parser_name, syn_norm))

                    if len(candidates) > 0:

                        confidence_values = {max(conf_set) for conf_set in confidences.values()}
                        chosen_conf = max(confidence_values)
                        if len(confidence_values) > 1:
                            logger.warning(
                                f"confidences conflict between parsers for {matched_text}: {confidences}. The maximum will be selected ({chosen_conf})"
                            )

                        e = Entity.load_contiguous_entity(
                            start=start_index,
                            end=end_index + 1,
                            match=matched_text,
                            entity_class=entity_class,
                            namespace=self.namespace(),
                            mention_confidence=chosen_conf,
                        )
                        for candidate in candidates:
                            e.add_or_update_linking_candidate(
                                candidate, LinkingMetrics(exact_match=True)
                            )
                        entities.append(e)
        return entities

    @document_iterating_step
    def __call__(self, doc: Document) -> None:

        for section in doc.sections:
            spacy_doc = self.spacy_pipelines.process_single(section.text, BASIC_PIPELINE_NAME)
            starts, ends = set(), set()
            for tok in spacy_doc:
                starts.add(tok.idx)
                ends.add(tok.idx + len(tok) - 1)

            section.entities.extend(
                self._process_automaton(
                    self.automaton, section.text.lower(), section.text, starts, ends
                )
            )
