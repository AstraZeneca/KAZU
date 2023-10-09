from collections.abc import Iterable

from spacy.tokens import Doc, Span, Token

from kazu.data.data import Section, Entity
from kazu.ontology_preprocessing.base import OntologyParser
from kazu.utils.spacy_pipeline import SpacyPipelines, BASIC_PIPELINE_NAME, basic_spacy_pipeline


class SpacyToKazuObjectMapper:
    """Maps entities and text from a :class:`.Section` to a Spacy `Doc
    <https://spacy.io/api/doc>`_ using :func:`.basic_spacy_pipeline`\\."""

    def __init__(self, parsers: Iterable[OntologyParser]):
        """

        :param parsers: parsers are required so that all entity classes can be set as spacy extensions
        """
        for parser in parsers:
            Token.set_extension(parser.entity_class, default=False, force=True)
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)

    def __call__(self, section: Section) -> dict[Entity, Span]:
        """Convert a :class:`.Section` into a dictionary of :class:`.Entity` to
        Spacy `Span <https://spacy.io/api/span>`_ mappings."""

        spacy_doc: Doc = self.spacy_pipelines.process_single(section.text, BASIC_PIPELINE_NAME)
        ent_to_span = {}
        for entity in section.entities:
            entity_class = entity.entity_class
            span = spacy_doc.char_span(
                start_idx=entity.start,
                end_idx=entity.end,
                label=entity_class,
                alignment_mode="expand",
            )
            if span is not None:
                ent_to_span[entity] = span
                for token in span:
                    token._.set(entity_class, True)
        return ent_to_span
