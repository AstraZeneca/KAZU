import logging
import traceback
from typing import Iterator, List, Optional, Tuple

import spacy

from kazu.data.data import CharSpan, Document, Section, Entity, PROCESSING_EXCEPTION
from kazu.steps import BaseStep
from kazu.utils.grouping import sort_then_group
from kazu.utils.utils import as_path, PathLike

logger = logging.getLogger(__name__)


class ExplosionNERStep(BaseStep):
    """
    A wrapper for the explosion ontology-based entity matcher and linker
    """

    def __init__(
        self,
        depends_on: Optional[List[str]],
        path: PathLike,
        include_sentence_offsets: bool = True,
    ):
        """
        :param depends_on:
        :param path: path to spacy pipeline including Ontology Matcher.
        :param include_sentence_offsets: whether to add sentence offsets to the metadata.

        """

        super().__init__(depends_on=depends_on)
        self.include_sentence_offsets = include_sentence_offsets
        self.path = as_path(path)

        # TODO: config override for when how we map parser names to entity types has changed since the last pipeline buid
        # think about how this affects OntologyMatcher._set_span_attributes lookup of parser names in case they
        # are not there in the new config.
        self.spacy_pipeline = spacy.load(self.path)
        self.span_key = self.spacy_pipeline.get_pipe("ontology_matcher").span_key

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []

        try:
            texts_and_sections = (
                (section.get_text(), (section, doc)) for doc in docs for section in doc.sections
            )

            # TODO: multiprocessing within the pipe command?
            spacy_result: Iterator[
                Tuple[spacy.tokens.Doc, Tuple[Section, Document]]
            ] = self.spacy_pipeline.pipe(texts_and_sections, as_tuples=True)

            for processed_text, (section, doc) in spacy_result:
                entities = []

                spans = processed_text.spans[self.span_key]
                grouped_spans = sort_then_group(spans, self._mapping_invariant_span)
                for mapping_invariant_span, span_group in grouped_spans:
                    span_start, span_end, span_text, entity_type = mapping_invariant_span
                    e = Entity.load_contiguous_entity(
                        start=span_start,
                        end=span_end,
                        match=span_text,
                        entity_class=entity_type,
                        namespace=self.namespace(),
                    )
                    entities.append(e)

                # add sentence offsets
                if self.include_sentence_offsets:
                    sent_metadata = []
                    for sent in processed_text.sents:
                        sent_metadata.append(CharSpan(sent.start_char, sent.end_char))
                    section.sentence_spans = sent_metadata

                # if one section of a doc fails after others have succeeded, this will leave failed docs
                # in a partially processed state. It's actually unclear to me whether this is desireable or not.
                section.entities.extend(entities)

        # this will give up on all docs as soon as one fails - we could have an additional
        # try-except inside the loop. We'd probably need to handle the case when the iterator raises an
        # error when we try iterating further though, or we might get stuck in a loop.
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            message = f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
            for doc in docs:
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
        return docs, failed_docs

    @staticmethod
    def _mapping_invariant_span(span: spacy.tokens.Span) -> Tuple[int, int, str, str]:
        """Return key information about a span excluding mapping information.

        This includes the entity_class (stored in span.label_) since this is
        stored on Kazu's :class:`kazu.data.data.Entity` concept rather than
        :class:`kazu.data.data.Mapping`
        """
        return (span.start_char, span.end_char, span.text, span.label_)
