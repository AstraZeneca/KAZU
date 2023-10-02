import logging
from collections import defaultdict
from collections.abc import Iterable

import ahocorasick
from spacy.tokens import Span

from kazu.data.data import Document, Entity, SynonymTermWithMetrics, MentionConfidence
from kazu.database.in_memory_db import SynonymDatabase, ParserName, NormalisedSynonymStr
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.steps import document_iterating_step
from kazu.steps.step import ParserDependentStep
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.curated_term_tools import filter_curations_for_ner
from kazu.utils.spacy_pipeline import SpacyPipelines, basic_spacy_pipeline, BASIC_PIPELINE_NAME

logger = logging.getLogger(__name__)

EntityClass = str
EntityInfoToOntologyInfoMapping = defaultdict[
    tuple[EntityClass, MentionConfidence], set[tuple[ParserName, NormalisedSynonymStr]]
]
EntityStartIndex = int
EntityEndIndex = int
MatchedString = str


class MemoryEfficientStringMatchingStep(ParserDependentStep):
    """A wrapper for the ahocorasick algorithm.

    In testing, this implementation is comparable in speed to a Spacy
    `PhraseMatcher <https://spacy.io/api/phrasematcher>`_\\,
    and uses a fraction of the memory. Since this implementation is unaware
    of NLP concepts such as tokenization, we backfill this capability by
    checking for word boundaries with a custom spacy tokenizer.
    """

    def __init__(self, parsers: Iterable[OntologyParser]):
        super().__init__(parsers)
        self.parsers = parsers
        self.automaton_strict, self.automaton_lc = self._create_automatons()
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)
        self.synonym_db = SynonymDatabase()

    @kazu_disk_cache.memoize(ignore={0})
    def _create_automatons(self):
        """Create :class:`ahocorasick.Automaton`\\'s for parsers."""
        key_to_ontology_info_lc: defaultdict[str, EntityInfoToOntologyInfoMapping] = defaultdict(
            lambda: defaultdict(set)
        )
        key_to_ontology_info_cs: defaultdict[str, EntityInfoToOntologyInfoMapping] = defaultdict(
            lambda: defaultdict(set)
        )

        logger.info("Creating ahocorasick Automatons")
        for parser in self.parsers:
            parser_curations = parser.populate_databases(return_curations=True)
            if parser_curations is None:
                logger.warning(
                    "tried to create ahocorasick data for parser %s, but no curations are available",
                    parser.name,
                )
                continue
            if len(parser_curations) == 0:
                logger.warning(
                    "tried to create ahocorasick data for parser %s, but no curations were produced",
                    parser.name,
                )
                continue

            curations_for_ner = set(filter_curations_for_ner(parser_curations, parser))

            for curation in curations_for_ner:
                # a curation can have different term_norms for different parsers,
                # since the string normalizer's output depends on the entity class.
                # Also, a curation may exist in multiple SynonymTerm.terms
                term_norm = curation.term_norm_for_linking(parser.entity_class)
                entity_key = (parser.entity_class, curation.mention_confidence)
                ontology_value = (parser.name, term_norm)
                if curation.case_sensitive:
                    match_string = curation.curated_synonym
                    target_dict = key_to_ontology_info_cs
                else:
                    match_string = curation.curated_synonym.lower()
                    target_dict = key_to_ontology_info_lc

                target_dict[match_string][entity_key].add(ontology_value)

        if len(key_to_ontology_info_cs) > 0:
            automaton_strict = ahocorasick.Automaton()
            for key, entity_to_ontology_info in key_to_ontology_info_cs.items():
                automaton_strict.add_word(key, (key, entity_to_ontology_info))
            automaton_strict.make_automaton()
        else:
            automaton_strict = None
        if len(key_to_ontology_info_lc) > 0:
            automaton_lc = ahocorasick.Automaton()
            for key, entity_to_ontology_info in key_to_ontology_info_lc.items():
                automaton_lc.add_word(key, (key, entity_to_ontology_info))
            automaton_lc.make_automaton()
        else:
            automaton_lc = None

        return automaton_strict, automaton_lc

    def _word_is_valid(
        self, start_char: int, end_char: int, starts: set[int], ends: set[int]
    ) -> bool:
        return start_char in starts and end_char in ends

    def _process_automaton(
        self,
        automaton: ahocorasick.Automaton,
        matchable_text: str,
        original_text: str,
        starts: set[int],
        ends: set[int],
    ) -> list[Entity]:
        entities = []
        for end_index, (match_key, ontology_dict) in automaton.iter(matchable_text):
            start_index = end_index - len(match_key) + 1
            original_match = original_text[start_index : end_index + 1]
            if self._word_is_valid(start_index, end_index, starts, ends):
                for (entity_class, confidence), parser_info_set in ontology_dict.items():
                    terms = list()
                    for parser_name, term_norm in parser_info_set:
                        term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                            self.synonym_db.get(parser_name, term_norm), exact_match=True
                        )
                        terms.append(term_with_metrics)
                    e = Entity.load_contiguous_entity(
                        start=start_index,
                        end=end_index,
                        match=original_match,
                        entity_class=entity_class,
                        namespace=self.namespace(),
                        mention_confidence=confidence,
                    )
                    e.update_terms(terms)
                    entities.append(e)
        return entities

    @document_iterating_step
    def __call__(self, doc: Document) -> None:

        for section in doc.sections:
            spacy_doc = self.spacy_pipelines.process_single(section.text, BASIC_PIPELINE_NAME)
            starts, ends = set(), set()
            for tok in spacy_doc:
                span = Span(spacy_doc, tok.i, tok.i + 1)
                starts.add(span.start_char)
                ends.add(span.end_char - 1)

            if self.automaton_lc is not None:
                section.entities.extend(
                    self._process_automaton(
                        self.automaton_lc, section.text.lower(), section.text, starts, ends
                    )
                )
            if self.automaton_strict is not None:
                section.entities.extend(
                    self._process_automaton(
                        self.automaton_strict, section.text, section.text, starts, ends
                    )
                )
