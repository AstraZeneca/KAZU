from typing import TypedDict, NamedTuple
from collections.abc import Iterable
from collections import defaultdict

import numpy as np

from kazu.data import Document, Entity, Section, CharSpan
from kazu.steps import Step, document_iterating_step
from sklearn.feature_extraction.text import TfidfVectorizer


class ScoredContext(NamedTuple):
    entity_class: str
    score: float
    thresh: float


class TfIdfDisambiguationEntry(NamedTuple):
    entity_class: str
    tfidf_document: np.ndarray
    tfidf_vectorizer: TfidfVectorizer
    thresh: float


class DisambiguationEntry(TypedDict):
    entity_class: str
    relevant_text: list[str]
    thresh: float


EntSectionPair = tuple[Entity, Section]


class EntityClassTfIdfScorer:
    def __init__(self, spans_to_tfidf_disambiguator: dict[str, list[TfIdfDisambiguationEntry]]):
        self.spans_to_tfidf_disambiguator = spans_to_tfidf_disambiguator

    @staticmethod
    def from_spans_to_sentence_disambiguator(
        spans_text_disambiguator: dict[str, list[DisambiguationEntry]]
    ) -> "EntityClassTfIdfScorer":
        return EntityClassTfIdfScorer(
            EntityClassTfIdfScorer.build_tfidf_documents(spans_text_disambiguator)
        )

    @staticmethod
    def build_tfidf_documents(
        spans_text_disambiguator: dict[str, list[DisambiguationEntry]]
    ) -> dict[str, list[TfIdfDisambiguationEntry]]:
        span_to_tfidf_disambiguator = {}
        for span, disambiguation_entries in spans_text_disambiguator.items():
            span_to_tfidf_disambiguator[span] = [
                EntityClassTfIdfScorer.disambiguation_entry_to_tfidf_entry(dis_ent)
                for dis_ent in disambiguation_entries
            ]
        return span_to_tfidf_disambiguator

    @staticmethod
    def disambiguation_entry_to_tfidf_entry(
        disamb_entry: DisambiguationEntry,
    ) -> TfIdfDisambiguationEntry:
        tfidf_vectorizer = TfidfVectorizer()
        document_mat = tfidf_vectorizer.fit_transform(disamb_entry["relevant_text"])
        return TfIdfDisambiguationEntry(
            entity_class=disamb_entry["entity_class"],
            tfidf_document=document_mat.data,
            tfidf_vectorizer=tfidf_vectorizer,
            thresh=disamb_entry["thresh"],
        )

    @staticmethod
    def tfidf_score(
        ent_context: str, tfidf_disambig_entry: TfIdfDisambiguationEntry
    ) -> ScoredContext:
        vectorizer = tfidf_disambig_entry.tfidf_vectorizer
        doc_mat = tfidf_disambig_entry.tfidf_document
        vector_context = vectorizer.transform([ent_context])
        score = np.squeeze(np.asarray(vector_context.dot(doc_mat.T))).item()
        return ScoredContext(
            entity_class=tfidf_disambig_entry.entity_class,
            score=score,
            thresh=tfidf_disambig_entry.thresh,
        )

    def score_entity_context(self, ent_span: str, ent_context: str) -> Iterable[ScoredContext]:
        """Score the entity context against the TfIdf documents specified for the
        entity's span.

        :param ent_span:
        :param ent_context:
        :return:
        """
        if ent_span not in self.spans_to_tfidf_disambiguator:
            yield from ()
        else:
            tfidf_disambiguation_entries = self.spans_to_tfidf_disambiguator[ent_span]
            for tfidf_disambiguation_entry in tfidf_disambiguation_entries:
                scored_context = self.tfidf_score(ent_context, tfidf_disambiguation_entry)
                yield scored_context


class EntityClassDisambiguationStep(Step):
    """
    .. warning::
           This step is deprecated and may be removed in a future release.
    """

    def __init__(self, context: dict[str, list[DisambiguationEntry]]):
        """Optionally disambiguates the entity class (anatomy, drug, etc.) of entities
        that exactly share a span in a document.

        For example, "UCB" could refer to "umbilical cord blood" an anatomical entity, or the pharmaceutical company
        UCB, a corporate entity. An expected context might be "umbilical pregnancy blood baby placenta..." in the former
        case, or "company business..." in the latter. Multiple expected contexts (disambiguation entries) should be
        provided to allow this step to choose the best matching entity class for an entity span. A tf-idf model is built
        to correlate an entity's actual textual context with the provided expected context, and provided thresholds are
        used to allow the tf-idf model to choose the most suitable entity class.

        :param context: Specifies synonyms to disambiguate along with an expected textual context around those synonyms.
        """
        self.spans_to_disambiguate = set(context.keys())
        self.entity_class_scorer = EntityClassTfIdfScorer.from_spans_to_sentence_disambiguator(
            context
        )

    @staticmethod
    def sentence_context_for_entity(entity: Entity, section: Section, window: int = 3) -> str:
        sent_spans = list(section.sentence_spans)
        if len(sent_spans) == 0:
            return section.text
        else:
            idx = next(
                idx
                for idx, sent_span in enumerate(sent_spans)
                if any(
                    entity_span.is_completely_overlapped(sent_span) for entity_span in entity.spans
                )
            )
            context_start_idx = max(0, idx - int(window / 2))
            context_end_idx = min(len(sent_spans), idx + int(window / 2)) + 1
            sentence_context_spans = sent_spans[context_start_idx:context_end_idx]
            return section.text[sentence_context_spans[0].start : sentence_context_spans[-1].end]

    def spangrouped_ent_section_pairs(self, doc: Document) -> Iterable[list[EntSectionPair]]:
        spans_to_ents_and_sections: defaultdict[
            tuple[int, frozenset[CharSpan]], list[EntSectionPair]
        ] = defaultdict(list)
        for section_idx, section in enumerate(doc.sections):
            for ent in section.entities:
                if ent.match in self.spans_to_disambiguate:
                    spans_to_ents_and_sections[(section_idx, ent.spans)].append((ent, section))
        return spans_to_ents_and_sections.values()

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        drop_set: dict[Section, set[Entity]] = defaultdict(set)
        for spansharing_ent_section_pairs in self.spangrouped_ent_section_pairs(doc):
            if len(spansharing_ent_section_pairs) == 1:
                # only a single entity; span(s) are unambiguous for entity class
                continue

            representative_ent, representative_section = spansharing_ent_section_pairs[0]
            ents: list[Entity] = [ent for ent, section in spansharing_ent_section_pairs]
            class_to_ents = defaultdict(set)
            for ent in ents:
                class_to_ents[ent.entity_class].add(ent)

            ent_match = representative_ent.match
            ent_sentence_context_str = self.sentence_context_for_entity(
                representative_ent, representative_section
            )
            scored_contexts = self.entity_class_scorer.score_entity_context(
                ent_span=ent_match, ent_context=ent_sentence_context_str
            )

            acceptable_scores = (
                scored_context
                for scored_context in scored_contexts
                if scored_context.score >= scored_context.thresh
            )
            best_score = max(
                acceptable_scores, default=None, key=lambda scored_context: scored_context.score
            )
            if best_score is not None:
                best_match_ents = class_to_ents[best_score.entity_class]
                drop_set[representative_section].update(
                    (ent for ent in ents if ent not in best_match_ents)
                )
            else:
                drop_set[representative_section].update(ents)

        for section, drop_entities in drop_set.items():
            section.entities = [ent for ent in section.entities if ent not in drop_entities]
