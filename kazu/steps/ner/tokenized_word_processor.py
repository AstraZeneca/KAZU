import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable, Optional

import pydash
import torch
from torch import Tensor

from kazu.data.data import (
    Entity,
    ENTITY_OUTSIDE_SYMBOL,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenizedWord:
    """
    A convenient container for a word, which may be split into multiple tokens by e.g. WordPiece tokenisation
    """

    token_ids: List[int]
    tokens: List[str]
    token_confidences: List[Tensor]
    token_offsets: List[Tuple[int, int]]
    word_offset_start: int
    word_offset_end: int
    word_id: int


class SpanFinder:
    """
    finds spans across a sequence of TokenizedWord, according to some rules (see span_end_condition and
    span_continue_condition). After being called, the spans can be accessed via self.spans. This is a list of
    dictionaries. Each dictionary has a key of the entity class, and a list of TokenizedWord representing the entity
    """

    def __init__(self, threshold: float, text: str, id2label: Dict[int, str]):
        self.text = text
        self.threshold = threshold
        self.class_word_map: Dict[str, List[TokenizedWord]] = defaultdict(list)
        self.words: List[TokenizedWord] = []
        self.span_breaking_chars = set("() ")
        self.non_breaking_span_chars = set("-")
        self.spans: List[Dict[str, List[TokenizedWord]]] = []
        self.id2label = id2label

    def resolve_word(self, word: TokenizedWord) -> Tuple[Set[str], List[float]]:
        """
        get a set of classes and a list of token confidences associated with a word
        :param word:
        :return:
        """
        classes = []
        confidences = []

        for i in range(len(word.token_confidences)):
            for bio_label, class_label, confidence_val in self.get_labels_under_threshold(
                word=word, label_index=i
            ):
                classes.append(class_label)
                confidences.append(confidence_val)
        return set([x for x in classes if x is not None]), confidences

    def get_labels_under_threshold(
        self, word: TokenizedWord, label_index: int
    ) -> Iterable[Tuple[str, Optional[str], float]]:
        """
        for a given word, yield a BIO label, class label and confidence val for all labels above the configured
        threshold
        :param word:
        :param label_index:
        :return:
        """
        confidences_indices_sorted = torch.argsort(
            word.token_confidences[label_index], dim=-1, descending=True
        )
        for confidence_index in confidences_indices_sorted:
            confidence_val: float = word.token_confidences[label_index][confidence_index].item()
            if confidence_val > self.threshold:
                bio_label = self.id2label[confidence_index.item()]
                if bio_label == ENTITY_OUTSIDE_SYMBOL:
                    yield bio_label, None, confidence_val
                else:
                    bio_label, class_label = bio_label.split("-")
                    yield bio_label, class_label, confidence_val
            else:
                break

    def span_continue_condition(self, classes: Set[str]):
        """
        A potential entity span must continue if any of the following conditions are met:
        1. There are any class assignments in any of the tokens in the next word.
        2. The previous character in the span is in the set of self.non_breaking_span_chars
        3. The previous character in the span is not in the set of self.span_breaking_chars
        :param word:
        :return:
        """
        span_broken_char = self.text[self.words[-1].word_offset_end] not in self.span_breaking_chars
        span_continue_char = (
            self.text[self.words[-1].word_offset_end] in self.non_breaking_span_chars
        )
        class_found = len(classes) > 0
        return any([span_broken_char, span_continue_char, class_found])

    def _update_class_word_map(self, classes: Set[str], word: TokenizedWord):
        for clazz in classes:
            self.class_word_map[clazz].append(word)

    def process_next_word(self, word: TokenizedWord):
        """
        process the next word in the sequence, updating span information accordingly
        :param word:
        :return:
        """

        classes, confidences = self.resolve_word(word)
        if self.words:
            if self.span_continue_condition(classes):
                self._update_class_word_map(classes, word)
            else:
                if len(self.class_word_map) > 0:
                    self.spans.append(self.class_word_map)
                self.reset()
                self._update_class_word_map(classes, word)
        else:
            self._update_class_word_map(classes, word)
        self.words.append(word)

    def reset(self):
        self.class_word_map = defaultdict(list)

    def __call__(self, words: List[TokenizedWord]):
        for word in words:
            self.process_next_word(word)


class TokenizedWordProcessor:
    """
    Because of the inherent obscurity of the inner workings of transformers, sometimes they produce BIO tags that
    don't correctly align to whole words. So what do we do? Fix it with rules :~Z

    This class is designed to work when an entire sequence of NER labels is known and therefore we can apply some
    post-processing logic - i.e. a hack until we have time to debug the model

    """

    def __init__(self, confidence_threshold: float, id2label: Dict[int, str]):
        self.id2label = id2label
        self.confidence_threshold = confidence_threshold

    def __call__(self, words: List[TokenizedWord], text: str, namespace: str) -> List[Entity]:
        span_finder = SpanFinder(
            threshold=self.confidence_threshold, text=text, id2label=self.id2label
        )
        span_finder(words)
        ents = pydash.flatten(
            [self.spans_to_entities(x, text, namespace) for x in span_finder.spans]
        )
        return ents

    def spans_to_entities(
        self, span_dict: Dict[str, List[TokenizedWord]], text: str, namespace: str
    ) -> List[Entity]:
        """
        convert spans to instances of Entity, adding in namespace info as appropriate
        :param span_dict:
        :param text:
        :param namespace:
        :return:
        """
        entities = []
        for class_label, offsets in span_dict.items():
            start, end = self.calculate_span_offsets(offsets)
            entity = Entity.load_contiguous_entity(
                start=start,
                end=end,
                match=text[start:end],
                namespace=namespace,
                entity_class=class_label,
            )
            entities.append(entity)
        return entities

    def calculate_span_offsets(self, words: List[TokenizedWord]) -> Tuple[int, int]:
        starts, ends = [], []
        for word in words:
            starts.append(word.word_offset_start)
            ends.append(word.word_offset_end)
        return min(starts), max(ends)
