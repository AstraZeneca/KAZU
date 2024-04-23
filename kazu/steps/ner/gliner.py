import dataclasses
from collections import defaultdict, Counter

from gliner import GLiNER
from gliner.modules.token_splitter import WhitespaceTokenSplitter
from kazu.data import Document, Entity
from kazu.steps import Step, document_iterating_step
from kazu.utils.utils import token_sliding_window, sort_then_group


class GLiNERStep(Step):
    """Wrapper for GLiNER models and library.

    Original Paper:
        GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.
        Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois
        https://arxiv.org/abs/2311.08526

    Implements a sliding window to enable the library to process long contexts. If multiple classes
    are detected for the same string, the most frequently occuring one will be selected for all strings
    (a.k.a majority vote). In the case of a tie, the first sequentially detected class will be used.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        gliner_class_prompt_to_entity_class: dict[str, str],
        threshold: float = 0.3,
        stride: int = 30,
        window_size: int = 300,
    ):
        """

        :param pretrained_model_name_or_path: Passed to :meth:`GLiNER.from_pretrained`\\.
        :param gliner_class_prompt_to_entity_class: Since GLiNER needs entity class prompts, these might not map exactly
            to our global NER classes. Therefore, this dictionary provides this mapping.
        :param threshold: passed to :meth:`GLiNER.predict_entities`\\.
        :param stride: number of tokens used for sliding window overlap.
        :param window_size: total window size. This should be the same as the maximum number of tokens the GLiNER model
            supports.
        """

        self.model = GLiNER.from_pretrained(pretrained_model_name_or_path)
        self.token_splitter = WhitespaceTokenSplitter()
        self.gliner_class_prompt_to_entity_class = gliner_class_prompt_to_entity_class
        self.threshold = threshold
        self.stride = stride
        self.window_size = window_size

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        # needed for majority voting at the end of the document
        section_entities_map = defaultdict(list)
        entity_class_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)

        for section in doc.sections:
            token_and_offsets = [token for token in self.token_splitter(section.text)]
            for sub_text, entities_start_at_index, entities_end_at_index in token_sliding_window(
                token_and_offsets,
                window_size=self.window_size,
                stride=self.stride,
                text=section.text,
            ):

                predictions = self.model.predict_entities(
                    sub_text,
                    labels=self.gliner_class_prompt_to_entity_class.keys(),
                    threshold=self.threshold,
                )
                for ent_pred_dict in predictions:
                    start = ent_pred_dict["start"] + entities_start_at_index
                    end = ent_pred_dict["end"] + entities_start_at_index
                    # skip the window overlap
                    if start >= entities_start_at_index and end < entities_end_at_index:
                        ent_class = entity_class = self.gliner_class_prompt_to_entity_class[
                            ent_pred_dict["label"]
                        ]
                        match = ent_pred_dict["text"]
                        entity_class_counter[match][ent_class] += 1

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
        # calculate majority vote and update sections
        for section, entities in section_entities_map.items():
            for ent_match, ents_this_match in sort_then_group(entities, lambda x: x.match):
                most_common_class_this_match = entity_class_counter[ent_match].most_common(1)[0][0]
                section.entities.extend(
                    dataclasses.replace(ent, entity_class=most_common_class_this_match)
                    for ent in ents_this_match
                )
