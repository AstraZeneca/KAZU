import logging
from typing import Iterator, List, Tuple, Iterable

import spacy
from kazu.data.data import (
    CharSpan,
    Document,
    Section,
    Entity,
    SynonymTermWithMetrics,
)
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.steps import Step, batch_step
from kazu.utils.utils import PathLike
from spacy.tokens import Span

logger = logging.getLogger(__name__)


class ExplosionStringMatchingStep(Step):
    """
    A wrapper for the explosion ontology-based entity matcher and linker.
    """

    def __init__(
        self,
        path: PathLike,
        include_sentence_offsets: bool = True,
    ):
        """
        :param path: path to spacy pipeline including Ontology Matcher.
        :param include_sentence_offsets: whether to add sentence offsets to the metadata.

        """
        self.include_sentence_offsets = include_sentence_offsets
        self.path = path

        # TODO: config override for when how we map parser names to entity types has changed since the last pipeline build
        # think about how this affects the OntologyMatcher's lookup of parser names in case they
        # are not there in the new config.
        self.spacy_pipeline = spacy.load(path)
        matcher: OntologyMatcher = self.spacy_pipeline.get_pipe("ontology_matcher")
        self.span_key = matcher.span_key

        self.synonym_db = SynonymDatabase()

    def extract_entity_data_from_spans(
        self, spans: Iterable[Span]
    ) -> Iterator[Tuple[int, int, str, str, str, str]]:
        for span in spans:
            for entity_class, ontology_data in span._.ontology_dict_.items():
                for parser_name, term_norm in ontology_data:
                    yield span.start_char, span.end_char, span.text, entity_class, parser_name, term_norm

    @batch_step
    def __call__(self, docs: List[Document]) -> None:
        texts_and_sections = (
            (section.get_text(), section) for doc in docs for section in doc.sections
        )

        # TODO: multiprocessing within the pipe command?
        spacy_result: Iterator[Tuple[spacy.tokens.Doc, Section]] = self.spacy_pipeline.pipe(
            texts_and_sections, as_tuples=True
        )

        for processed_text, section in spacy_result:
            entities = []

            spans = processed_text.spans[self.span_key]
            for (
                start_char,
                end_char,
                text,
                entity_class,
                parser_name,
                term_norm,
            ) in self.extract_entity_data_from_spans(spans):
                e = Entity.load_contiguous_entity(
                    start=start_char,
                    end=end_char,
                    match=text,
                    entity_class=entity_class,
                    namespace=self.namespace(),
                )
                entities.append(e)
                terms = []
                term = self.synonym_db.get(parser_name, term_norm)
                terms.append(term)
                terms_with_metrics = (
                    SynonymTermWithMetrics.from_synonym_term(term, exact_match=True)
                    for term in terms
                )
                e.update_terms(terms_with_metrics)

            # add sentence offsets
            if self.include_sentence_offsets:
                sent_metadata = []
                for sent in processed_text.sents:
                    sent_metadata.append(CharSpan(sent.start_char, sent.end_char))
                section.sentence_spans = sent_metadata

            # if one section of a doc fails after others have succeeded, this will leave failed docs
            # in a partially processed state. It's actually unclear to me whether this is desireable or not.
            section.entities.extend(entities)
