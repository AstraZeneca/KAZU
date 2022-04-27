import logging
import traceback
from collections import defaultdict
from itertools import groupby
from typing import Dict, Iterator, List, Optional, Set, Tuple

import spacy
from kazu.data.data import CharSpan, Document, Section, Entity, PROCESSING_EXCEPTION
from kazu.modelling.ontology_matching.assemble_pipeline import main as assemble_pipeline
from kazu.modelling.ontology_matching.ontology_matcher import SPAN_KEY
from kazu.modelling.ontology_preprocessing.base import OntologyParser
from kazu.steps import BaseStep
from kazu.utils.utils import as_path, PathLike
from spacy.tokens import Span as spacy_span

logger = logging.getLogger(__name__)


class ExplosionNERStep(BaseStep):
    """
    A wrapper for the explosion ontology-based entity matcher and linker
    """

    def __init__(
        self,
        depends_on: List[str],
        path: PathLike,
        parsers: List[OntologyParser],
        rebuild_pipeline: bool = False,
        include_sentence_offsets: bool = True,
        span_key: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        :param path: path to spacy pipeline including Ontology Matcher.
        :param depends_on:
        """

        super().__init__(depends_on=depends_on)
        self.include_sentence_offsets = include_sentence_offsets
        self.parsers = parsers
        self.path = as_path(path)
        self.labels = labels

        if rebuild_pipeline or not self.path.exists():
            if not self.parsers or self.labels is None:
                # invalid - no way to get a spacy pipeline.
                # gather the relevant information for the error message
                if rebuild_pipeline:
                    reason_for_trying_to_build_pipeline = (
                        "rebuild_pipeline parameter was set to True"
                    )
                else:
                    reason_for_trying_to_build_pipeline = (
                        f"the path parameter {self.path} does not exist"
                    )

                none_param = "parsers" if not parsers else "labels"
                raise ValueError(
                    f"Cannot instantiate {self.__class__.__name__} as the {none_param} param was None and {reason_for_trying_to_build_pipeline}"
                )

            logger.info("forcing a rebuild of spacy 'arizona' NER and EL pipeline")
            self.span_key = span_key if span_key is not None else SPAN_KEY
            self.spacy_pipeline = self.build_pipeline(
                self.path, self.parsers, self.span_key, self.labels
            )
        else:
            self.spacy_pipeline = spacy.load(self.path)

            self.span_key = self.spacy_pipeline.get_pipe("ontology_matcher").span_key
            if span_key is not None and span_key != self.span_key:
                logger.warning(
                    f"span key {span_key} used to instantiate {self} does not match the actual span key used in the loaded spacy pipeline: {self.span_key}"
                )

    @staticmethod
    def build_pipeline(
        path: PathLike, parsers: List[OntologyParser], span_key: str, labels: List[str]
    ) -> spacy.language.Language:
        return assemble_pipeline(
            parsers=parsers,
            labels=labels,
            span_key=span_key,
            output_dir=path,
        )

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        failed_docs = []

        try:
            texts_and_sections = (
                (section.get_text(), (section, doc)) for doc in docs for section in doc.sections
            )

            doc_to_processed_sections: Dict[Document, Set[Section]] = defaultdict(set)
            # TODO: multiprocessing within the pipe command?
            spacy_result: Iterator[
                Tuple[spacy.tokens.Doc, Tuple[Section, Document]]
            ] = self.spacy_pipeline.pipe(texts_and_sections, as_tuples=True)

            for processed_text, (section, doc) in spacy_result:
                entities = []

                spans = processed_text.spans[self.span_key]
                sorted_spans = sorted(spans, key=self._mapping_invariant_span)
                grouped_spans = groupby(sorted_spans, key=self._mapping_invariant_span)
                for mapping_invariant_span, span_group in grouped_spans:
                    span_start, span_end, span_text, span_label_ = mapping_invariant_span
                    e = Entity(
                        match=span_text,
                        entity_class=span_label_,
                        spans=frozenset((CharSpan(start=span_start, end=span_end),)),
                        namespace=self.namespace(),
                    )
                    entities.append(e)

                # add sentence offsets
                if self.include_sentence_offsets:
                    sent_metadata = []
                    for sent in processed_text.sents:
                        sent_metadata.append([sent.start_char, sent.end_char])
                    section.metadata["sentence_offsets"] = sent_metadata

                # if one section of a doc fails after others have succeeded, this will leave failed docs
                # in a partially processed state. It's actually unclear to me whether this is desireable or not.
                section.entities.extend(entities)
                doc_to_processed_sections[doc].add(section)
        # this will give up on all docs as soon as one fails - we could have an additional
        # try-except inside the loop. We'd probably need to handle the case when the iterator raises an
        # error when we try iterating further though, or we might get stuck in a loop.
        except Exception:
            affected_doc_ids = [doc.idx for doc in docs]
            for doc in docs:
                message = (
                    f"batch failed: affected ids: {affected_doc_ids}\n" + traceback.format_exc()
                )
                doc.metadata[PROCESSING_EXCEPTION] = message
                failed_docs.append(doc)
        return docs, failed_docs

    @staticmethod
    def _mapping_invariant_span(span: spacy_span) -> Tuple[int, int, str, str]:
        """Return key information about a span excluding mapping information.

        This still includes the class of the recognised entity (span.label_) since
        the entity_class is stored on Kazu's Entity concept rather than Mapping."""
        return (span.start_char, span.end_char, span.text, span.label_)
