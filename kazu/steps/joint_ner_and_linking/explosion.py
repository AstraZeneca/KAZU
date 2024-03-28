import logging
from collections.abc import Iterator, Iterable
from typing import cast

from kazu.data import (
    CharSpan,
    Document,
    Section,
    Entity,
    MentionConfidence,
    LinkingMetrics,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.ontology_matching import assemble_pipeline
from kazu.ontology_matching.ontology_matcher import OntologyMatcher, _MatcherOntologyData
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.steps import document_batch_step
from kazu.steps import ParserDependentStep
from kazu.utils.spacy_pipeline import SpacyPipelines
from kazu.utils.utils import PathLike, as_path
from spacy.tokens import Span, Doc

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

        :param parsers: the parsers used for the matching.
        :param path: path to spaCy pipeline including Ontology Matcher.
        :param include_sentence_offsets: whether to add sentence offsets to the metadata.
        :param ignore_cache: ignore cached version of spaCy pipeline (if available) and rebuild

        """
        # if we pass this straight to super().__init__ , this could exhaust the iterable
        # and we iterate over it again later, so this needs to be a list
        parser_list = list(parsers)
        super().__init__(parser_list)
        self.include_sentence_offsets = include_sentence_offsets
        self.path = as_path(path)

        if self.path.exists() and not ignore_cache:
            logger.info("loading spacy pipeline from %s", str(path))
            SpacyPipelines.add_from_path(name=self.namespace(), path=str(self.path.absolute()))
        else:
            logger.info(
                "cached spacy pipeline not detected or ignore_cache=True. Creating it at %s. This may take some time",
                str(self.path),
            )
            assemble_pipeline.main(output_dir=self.path, parsers=parser_list)
            SpacyPipelines.add_from_path(name=self.namespace(), path=str(self.path.absolute()))

        matcher = cast(
            OntologyMatcher, SpacyPipelines.get_model(self.namespace()).get_pipe("ontology_matcher")
        )
        self.span_key = matcher.span_key

        self.synonym_db = SynonymDatabase()
        self.spacy_pipelines = SpacyPipelines()

    def extract_entity_data_from_spans(
        self, spans: Iterable[Span]
    ) -> Iterator[tuple[int, int, str, _MatcherOntologyData]]:
        for span in spans:
            yield span.start_char, span.end_char, span.text, span._.ontology_dict_

    @document_batch_step
    def __call__(self, docs: list[Document]) -> None:
        texts_and_sections = ((section.text, section) for doc in docs for section in doc.sections)

        # TODO: multiprocessing within the pipe command?
        spacy_result = cast(
            Iterable[tuple[Doc, Section]],
            self.spacy_pipelines.process_batch(
                texts=texts_and_sections, model_name=self.namespace(), as_tuples=True
            ),
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

                for entity_class, per_parser_syn_norm_set in ontology_data.items():
                    confidences = set()
                    e = Entity.load_contiguous_entity(
                        start=start_char,
                        end=end_char,
                        match=text,
                        entity_class=entity_class,
                        namespace=self.namespace(),
                    )
                    for parser_name, syn_norm, confidence in per_parser_syn_norm_set:
                        mention_confidence = MentionConfidence(int(confidence))
                        confidences.add(mention_confidence)
                        e.add_or_update_linking_candidate(
                            self.synonym_db.get(parser_name, syn_norm),
                            LinkingMetrics(exact_match=True),
                        )

                    e.mention_confidence = max(confidences)
                    entities.append(e)

            # add sentence offsets
            if self.include_sentence_offsets:
                section.sentence_spans = (
                    CharSpan(sent.start_char, sent.end_char) for sent in processed_text.sents
                )

            # if one section of a doc fails after others have succeeded, this will leave failed docs
            # in a partially processed state. It's actually unclear to me whether this is desireable or not.
            section.entities.extend(entities)
