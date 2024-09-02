import dataclasses
import logging
import random
from abc import abstractmethod
from collections import defaultdict, Counter
from typing import Iterable, Optional

import ahocorasick


try:
    from gliner import GLiNER
except ImportError as e:
    raise ImportError(
        "To use GLiNERStep, you need to install gliner.\n"
        "Install with 'pip install kazu[all-steps]'.\n"
    ) from e

from kazu.utils.spacy_pipeline import BASIC_PIPELINE_NAME, SpacyPipelines, basic_spacy_pipeline
from kazu.data import Document, Entity, CharSpan, Section
from kazu.steps import Step, document_batch_step
from kazu.utils.utils import sort_then_group, word_is_valid

logger = logging.getLogger(__name__)

GLINER_SCORE_METADATA_KEY = "gliner_score"


@dataclasses.dataclass
class GliNERBatchItem:
    doc: Document
    section: Section
    start_span: CharSpan
    end_span: CharSpan
    sentence: str


class ConflictScorer:
    def __init__(self) -> None:
        self.doc_entities_map: defaultdict[Document, list[Entity]] = defaultdict(list)
        self.section_entities_map: defaultdict[Section, list[Entity]] = defaultdict(list)
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)

    def update(self, doc: Document, section: Section, entity: Entity) -> None:
        self.doc_entities_map[doc].append(entity)
        self.section_entities_map[section].append(entity)
        self._update(entity)

    @abstractmethod
    def _update(self, ent: Entity) -> None:
        pass

    @abstractmethod
    def _choose_best_match(self, ent_match: str) -> Entity:
        pass

    def finalise(self, namespace: str) -> None:
        for doc, entities in self.doc_entities_map.items():
            best_ent_per_match = {}
            for ent_match, ents_this_match in sort_then_group(entities, lambda x: x.match):
                best_ent = self._choose_best_match(ent_match)
                best_ent_per_match[best_ent.match] = best_ent

            automaton = ahocorasick.Automaton()
            for ent_match, ent in best_ent_per_match.items():
                automaton.add_word(ent_match, ent)
            automaton.make_automaton()

            for section in doc.sections:
                self._automaton_matching(automaton, namespace, section)

    def _automaton_matching(
        self, automaton: ahocorasick.Automaton, namespace: str, section: Section
    ) -> None:
        spacy_doc = self.spacy_pipelines.process_single(section.text, BASIC_PIPELINE_NAME)
        starts, ends = set(), set()
        for tok in spacy_doc:
            starts.add(tok.idx)
            ends.add(tok.idx + len(tok) - 1)
        for end_index, matched_ent in automaton.iter(section.text):
            start_index = end_index - len(matched_ent.match) + 1
            if word_is_valid(start_index, end_index, starts, ends):
                e = Entity.load_contiguous_entity(
                    start=start_index,
                    end=end_index + 1,
                    match=matched_ent.match,
                    entity_class=matched_ent.entity_class,
                    namespace=namespace,
                    mention_confidence=matched_ent.mention_confidence,
                )
                section.entities.append(e)


class MajorityVoteScorer(ConflictScorer):
    def __init__(self) -> None:
        super().__init__()
        self.entity_class_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.example_ent_this_match: dict[str, Entity] = {}

    def _update(self, ent: Entity) -> None:
        self.entity_class_counter[ent.match][ent.entity_class] += 1
        self.example_ent_this_match[ent.entity_class] = ent

    def _choose_best_match(self, ent_match: str) -> Entity:
        best_class = self.entity_class_counter[ent_match].most_common(1)[0][0]
        return self.example_ent_this_match[best_class]


class MaxScoreScorer(ConflictScorer):
    def __init__(self) -> None:
        super().__init__()
        self.highest_score_this_match: dict[str, float] = {}
        self.best_ent_this_match: dict[str, Entity] = {}

    def _update(self, ent: Entity) -> None:
        score = ent.metadata[GLINER_SCORE_METADATA_KEY]
        if score > self.highest_score_this_match.get(ent.match, 0):
            self.highest_score_this_match[ent.match] = score
            self.best_ent_this_match[ent.match] = ent

    def _choose_best_match(self, ent_match: str) -> Entity:
        return self.best_ent_this_match[ent_match]


class GLiNERStep(Step):
    """Wrapper for GLiNER models and library. Requires :class:`kazu.data.Section` to
    have sentence spans set on it, as sentences are processed in batches by GLiNER. This
    is to avoid the 'windowing' problem, whereby a multi-token entity could be split
    across two windows, leading to ambiguity over the entity class and spans. Since
    entities cannot theoretically cross sentences, batching sentences eliminates this
    problem.

    If multiple classes are detected for the same string, it will be resolved via
    the supplied :class:`~ConflictScorer` class

    .. attention::

       To use this step, you will need `gliner <https://github.com/urchade/GLiNER>`_
       installed, which is not installed as part of the default kazu install
       because this step isn't used as part of the default pipeline.
       You can either do:

       .. code-block:: console

          $ pip install gliner

       Or you can install required dependencies for all steps included in kazu
       with:

       .. code-block:: console

          $ pip install kazu[all-steps]

    Paper:

    | GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.
    | Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois
    | https://arxiv.org/abs/2311.08526

    .. raw:: html

       <details>
       <summary>Bibtex Citation Details</summary>

    .. code:: bibtex

       @misc{zaratiana2023gliner,
             title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer},
             author={Urchade Zaratiana and Nadi Tomeh and Pierre Holat and Thierry Charnois},
             year={2023},
             eprint={2311.08526},
             archivePrefix={arXiv},
             primaryClass={cs.CL}
       }

    .. raw:: html

       </details>
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        gliner_class_prompt_to_entity_class: dict[str, str],
        threshold: float = 0.3,
        batch_size: int = 2,
        device: Optional[str] = None,
        local_files_only: bool = True,
        conflict_scorer: type[ConflictScorer] = MajorityVoteScorer,
        max_context_size: Optional[int] = None,
        iterations: int = 5,
    ) -> None:
        """

        :param pretrained_model_name_or_path: Passed to ``GLiNER.from_pretrained``. Note that this could attempt to
            download a model from the HuggingFace Hub (see docs for `ModelHubMixin <https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin>`_ ).
        :param gliner_class_prompt_to_entity_class: Since GLiNER needs entity class prompts, these might not map exactly
            to our global NER classes. Therefore, this dictionary provides this mapping.
        :param threshold: passed to ``GLiNER.predict_entities``.
        :param batch_size: The number of sentences to process in a single batch. This is to avoid memory issues.
        :param device: passed to ``GLiNER.to``.
        :param local_files_only: passed to ``GLiNER.from_pretrained``.
        :param conflict_scorer: The method to use to resolve conflicts between entity classes. Defaults to :class:`~MajorityVoteScorer`\\.
        :param max_context_size: The maximum number of tokens to process in a single batch. This is to avoid memory issues. If None, the default is the model's max_len - 10 (for special tokens).
        :param iterations: The number of times to shuffle the entity class prompts. This is to avoid any bias in the model.
        """
        self.model = GLiNER.from_pretrained(
            pretrained_model_name_or_path, local_files_only=local_files_only
        )
        if device:
            self.model = self.model.to(device)
        self.gliner_class_prompt_to_entity_class = gliner_class_prompt_to_entity_class
        self.threshold = threshold
        self.splitter = self.model.data_processor.words_splitter
        self.max_context_size = (
            (self.model.config.max_len - 10) if max_context_size is None else max_context_size
        )  # -10 to keep a few tokens back for 'special tokens'
        self.batch_size = batch_size
        self.conflict_scorer: type[ConflictScorer]
        self.conflict_scorer = conflict_scorer
        self.label_sets: set[tuple[str, ...]] = set()
        random.seed(42)
        while len(self.label_sets) != iterations:
            self.label_sets.add(
                tuple(
                    random.sample(
                        self.gliner_class_prompt_to_entity_class.keys(),
                        k=len(self.gliner_class_prompt_to_entity_class.keys()),
                    )
                )
            )

    def _create_multidoc_batches(self, docs: Iterable[Document]) -> Iterable[list[GliNERBatchItem]]:
        accumulator = []
        for doc in docs:
            for section in doc.sections:
                for start_span, end_span in self._create_batches(section, doc.idx):
                    accumulator.append(
                        GliNERBatchItem(
                            doc,
                            section,
                            start_span,
                            end_span,
                            section.text[start_span.start : end_span.end],
                        )
                    )
                    if len(accumulator) >= self.batch_size:
                        yield accumulator
                        accumulator = []
        if accumulator:
            yield accumulator

    def _create_batches(
        self, section: Section, doc_idx: str
    ) -> Iterable[tuple[CharSpan, CharSpan]]:
        tokens_this_batch = 0
        span_list = list(section.sentence_spans)
        start_span_this_batch: CharSpan = span_list[0]
        end_span_this_batch: Optional[CharSpan] = None
        for sent_span in section.sentence_spans:

            sentence = section.text[sent_span.start : sent_span.end]
            token_count = len(list(self.splitter(sentence)))
            if token_count > self.model.config.max_len:
                logger.warning(
                    "long sentence detected in docid %s. Only the first %s tokens will"
                    "be processed",
                    doc_idx,
                    self.model.config.max_len,
                )
            if (
                tokens_this_batch + token_count >= self.max_context_size
                and end_span_this_batch is not None
            ):
                yield start_span_this_batch, end_span_this_batch
                tokens_this_batch = 0
                start_span_this_batch = sent_span
            elif sent_span is span_list[-1]:
                yield start_span_this_batch, sent_span

            end_span_this_batch = sent_span
            tokens_this_batch += token_count

    @document_batch_step
    def __call__(self, docs: Iterable[Document]) -> None:
        # needed for majority voting at the end of the document

        conflict_scorer = self.conflict_scorer()
        for batch_items in self._create_multidoc_batches(docs):
            for label_shuffle in self.label_sets:
                sentences = [x.sentence for x in batch_items]
                predictions = self.model.batch_predict_entities(
                    sentences,
                    labels=list(label_shuffle),
                    threshold=self.threshold,
                )

                for doc_idx, batch_and_prediction in sort_then_group(
                    zip(batch_items, predictions), key_func=lambda x: x[0].doc.idx
                ):
                    for batch_item, prediction in batch_and_prediction:
                        for ent_pred_dict in prediction:
                            start = ent_pred_dict["start"] + batch_item.start_span.start
                            end = ent_pred_dict["end"] + batch_item.start_span.start
                            entity_class = self.gliner_class_prompt_to_entity_class[
                                ent_pred_dict["label"]
                            ]
                            match = ent_pred_dict["text"]
                            entity = Entity.load_contiguous_entity(
                                start=start,
                                end=end,
                                match=match,
                                entity_class=entity_class,
                                namespace=self.namespace(),
                                metadata={GLINER_SCORE_METADATA_KEY: ent_pred_dict["score"]},
                            )
                            conflict_scorer.update(batch_item.doc, batch_item.section, entity)
        conflict_scorer.finalise(self.namespace())
