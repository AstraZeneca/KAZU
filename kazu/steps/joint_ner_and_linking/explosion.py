import logging
from typing import Iterator, List, Tuple, Iterable, Dict, Set, cast

import spacy
from spacy.tokens import Span

from kazu.data.data import (
    CharSpan,
    Document,
    Section,
    Entity,
    SynonymTermWithMetrics,
)
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_matching import assemble_pipeline
from kazu.modelling.ontology_matching.ontology_matcher import OntologyMatcher
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.steps import document_batch_step
from kazu.steps.step import ParserDependentStep
from kazu.utils.utils import PathLike, as_path

logger = logging.getLogger(__name__)


class ExplosionStringMatchingStep(ParserDependentStep):
    """A wrapper for the explosion ontology-based entity matcher and linker."""

    def __init__(
        self,
        parsers: Iterable[OntologyParser],
        path: PathLike,
        include_sentence_offsets: bool = True,
        ignore_cache: bool = False,
    ):
        """


        :param path: path to spacy pipeline including Ontology Matcher.
        :param include_sentence_offsets: whether to add sentence offsets to the metadata.
        :param ignore_cache: ignore cached version of spacy pipeline (if available) and rebuild

        """
        super().__init__(parsers)
        self.include_sentence_offsets = include_sentence_offsets
        self.path = as_path(path)

        if self.path.exists() and not ignore_cache:
            logger.info("loading spacy pipeline from %s", str(path))
            self.spacy_pipeline = spacy.load(path)
        else:
            logger.info(
                "cached spacy pipeline not detected or ignore_cache=True. Creating it at %s. This may take some time",
                str(self.path),
            )
            self.spacy_pipeline = assemble_pipeline.main(
                output_dir=self.path, parsers=list(parsers)
            )
        matcher = cast(OntologyMatcher, self.spacy_pipeline.get_pipe("ontology_matcher"))
        self.span_key = matcher.span_key

        self.synonym_db = SynonymDatabase()

    def extract_entity_data_from_spans(
        self, spans: Iterable[Span]
    ) -> Iterator[Tuple[int, int, str, Dict[str, Set[Tuple[str, str]]]]]:
        for span in spans:
            yield span.start_char, span.end_char, span.text, span._.ontology_dict_

    @document_batch_step
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
                ontology_data,
            ) in self.extract_entity_data_from_spans(spans):
                for entity_class, per_parser_term_norm_set in ontology_data.items():

                    e = Entity.load_contiguous_entity(
                        start=start_char,
                        end=end_char,
                        match=text,
                        entity_class=entity_class,
                        namespace=self.namespace(),
                    )
                    entities.append(e)
                    terms = []
                    for parser_name, term_norm in per_parser_term_norm_set:
                        term_with_metrics = SynonymTermWithMetrics.from_synonym_term(
                            self.synonym_db.get(parser_name, term_norm), exact_match=True
                        )
                        terms.append(term_with_metrics)
                    e.update_terms(terms)

            # add sentence offsets
            if self.include_sentence_offsets:
                section.sentence_spans = (
                    CharSpan(sent.start_char, sent.end_char) for sent in processed_text.sents
                )

            # if one section of a doc fails after others have succeeded, this will leave failed docs
            # in a partially processed state. It's actually unclear to me whether this is desireable or not.
            section.entities.extend(entities)
