import dataclasses
import logging
from collections import defaultdict, Counter
from typing import Iterable, Optional

try:
    from gliner import GLiNER
    from gliner.modules.token_splitter import WhitespaceTokenSplitter
except ImportError as e:
    raise ImportError(
        "To use GLiNERStep, you need to install gliner.\n"
        "Install with 'pip install kazu[all-steps]'.\n"
    ) from e
from kazu.data import Document, Entity, CharSpan, Section
from kazu.steps import Step, document_iterating_step
from kazu.utils.utils import sort_then_group

logger = logging.getLogger(__name__)


class GLiNERStep(Step):
    """Wrapper for GLiNER models and library. Requires :class:`kazu.data.Section` to
    have sentence spans set on it, as sentences are processed in batches by GLiNER. This
    is to avoid the 'windowing' problem, whereby a multi-token entity could be split
    across two windows, leading to ambiguity over the entity class and spans. Since
    entities cannot theoretically cross sentences, batching sentences eliminates this
    problem.

    If multiple classes are detected for the same string, the most frequently occuring
    one will be selected for all strings (a.k.a majority vote). In the case of a tie,
    the first sequentially detected class will be used.

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
    ):
        """

        :param pretrained_model_name_or_path: Passed to ``GLiNER.from_pretrained``. Note that this could attempt to
            download a model from the HuggingFace Hub (see docs for `ModelHubMixin <https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin>`_ ).
        :param gliner_class_prompt_to_entity_class: Since GLiNER needs entity class prompts, these might not map exactly
            to our global NER classes. Therefore, this dictionary provides this mapping.
        :param threshold: passed to ``GLiNER.predict_entities``.

        """

        self.model = GLiNER.from_pretrained(pretrained_model_name_or_path)
        self.gliner_class_prompt_to_entity_class = gliner_class_prompt_to_entity_class
        self.threshold = threshold
        self.splitter = WhitespaceTokenSplitter()
        self.max_batch_size = (
            self.model.config.max_len - 10
        )  # -10 to keep a few tokens back for 'special tokens'

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
                tokens_this_batch + token_count >= self.max_batch_size
                and end_span_this_batch is not None
            ):
                yield start_span_this_batch, end_span_this_batch
                tokens_this_batch = 0
                start_span_this_batch = sent_span
            elif sent_span is span_list[-1]:
                yield start_span_this_batch, sent_span

            end_span_this_batch = sent_span
            tokens_this_batch += token_count

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        # needed for majority voting at the end of the document
        section_entities_map = defaultdict(list)
        entity_class_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for section in doc.sections:
            if section.text and not section.sentence_spans:
                logger.warning(
                    "Skipping section of docid %s as %s requires a sentence splitter to have run",
                    doc.idx,
                    self.namespace(),
                )
                continue
            for batch_start_span, batch_end_span in self._create_batches(section, doc.idx):
                predictions = self.model.predict_entities(
                    section.text[batch_start_span.start : batch_end_span.end],
                    labels=self.gliner_class_prompt_to_entity_class.keys(),
                    threshold=self.threshold,
                )
                for ent_pred_dict in predictions:
                    start = ent_pred_dict["start"] + batch_start_span.start
                    end = ent_pred_dict["end"] + batch_start_span.start
                    entity_class = self.gliner_class_prompt_to_entity_class[ent_pred_dict["label"]]
                    match = ent_pred_dict["text"]
                    entity_class_counter[match][entity_class] += 1

                    section_entities_map[section].append(
                        Entity.load_contiguous_entity(
                            start=start,
                            end=end,
                            match=match,
                            entity_class=entity_class,
                            namespace=self.namespace(),
                            metadata={f"{self.namespace()}_score": ent_pred_dict["score"]},
                        )
                    )
        # calculate majority vote and update sections.
        for section, entities in section_entities_map.items():
            for ent_match, ents_this_match in sort_then_group(entities, lambda x: x.match):
                most_common_class_this_match = entity_class_counter[ent_match].most_common(1)[0][0]
                section.entities.extend(
                    dataclasses.replace(ent, entity_class=most_common_class_this_match)
                    for ent in ents_this_match
                )
