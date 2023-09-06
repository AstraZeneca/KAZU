import logging
from collections import defaultdict
from collections.abc import Iterable

import ahocorasick
import spacy
from kazu.data.data import Document, Entity, SynonymTermWithMetrics, MentionConfidence, CharSpan
from kazu.database.in_memory_db import SynonymDatabase, ParserName, NormalisedSynonymStr
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.steps import document_iterating_step
from kazu.ontology_matching.assemble_pipeline import (  # noqa: F401 #we need this import to register the spacy component
    KazuCustomEnglish,
)
from spacy.tokens import Span
from kazu.steps.step import ParserDependentStep
from kazu.utils.caching import kazu_disk_cache
from kazu.utils.curated_term_tools import filter_curations_for_ner

logger = logging.getLogger(__name__)

EntityClass = str
EntityInfoToOntologyInfoMapping = defaultdict[
    tuple[EntityClass, MentionConfidence], set[tuple[ParserName, NormalisedSynonymStr]]
]
EntityStartIndex = int
EntityEndIndex = int
MatchedString = str


class FastStringMatchingStep(ParserDependentStep):
    """A wrapper for the ahocorasick algorithm.

    In testing, this implementation is comparable in speed to a Spacy
    `PhraseMatcher <https://spacy.io/api/phrasematcher>`_\\,
    and uses a fraction of the memory. Since this implementation is unaware
    of NLP concepts such as tokenisation, we backfill this capability by
    checking for word boundaries with the spacy tokeniser.
    """

    def __init__(
        self,
        parsers: Iterable[OntologyParser],
        include_sentence_indices: bool = True,
        reload_spacy_at: int = 2000,
    ):
        """

        :param parsers:
        :param include_sentence_indices: Should spacy sentence indices be added to the section?
        :param reload_spacy_at: Since spacy suffers from
            `memory leaks <https://github.com/explosion/spaCy/discussions/9362>`_, reload the
            spacy pipeline to clear the vocab build up after this many calls.
        """
        super().__init__(parsers)
        self.include_sentence_offsets = include_sentence_indices
        self.reload_spacy_at = reload_spacy_at
        self.parsers = parsers
        self.automaton_strict, self.automaton_lc = self.create_automatons()
        self.nlp = self.reload_spacy_pipeline()
        self.call_count = 0
        self.synonym_db = SynonymDatabase()

    def reload_spacy_pipeline(self):
        nlp = spacy.blank("kazu_custom_en")
        nlp.add_pipe("sentencizer")
        return nlp

    @kazu_disk_cache.memoize(ignore={0})
    def create_automatons(self):
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

    def word_is_valid(
        self, start_char: int, end_char: int, starts: set[int], ends: set[int]
    ) -> bool:
        return start_char in starts and end_char in ends

    def _process_automation(
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
            if self.word_is_valid(start_index, end_index, starts, ends):
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
        self.call_count += 1
        if self.call_count % self.reload_spacy_at == 0:
            self.nlp = self.reload_spacy_pipeline()

        for section in doc.sections:
            spacy_doc = self.nlp(section.text)
            starts, ends = set(), set()
            for tok in spacy_doc:
                span = Span(spacy_doc, tok.i, tok.i + 1)
                starts.add(span.start_char)
                ends.add(span.end_char - 1)

            if self.automaton_lc is not None:
                section.entities.extend(
                    self._process_automation(
                        self.automaton_lc, section.text.lower(), section.text, starts, ends
                    )
                )
            if self.automaton_strict is not None:
                section.entities.extend(
                    self._process_automation(
                        self.automaton_strict, section.text, section.text, starts, ends
                    )
                )
            if self.include_sentence_offsets:
                section.sentence_spans = (
                    CharSpan(sent.start_char, sent.end_char) for sent in spacy_doc.sents
                )
