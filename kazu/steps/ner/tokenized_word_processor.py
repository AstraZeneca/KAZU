import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional

import pydash
import torch
from torch import Tensor

from kazu.data.data import (
    Entity,
    ENTITY_OUTSIDE_SYMBOL,
    ENTITY_START_SYMBOL,
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
    word_char_start: int
    word_char_end: int
    word_id: int


class SpanFinder:
    """
    finds spans across a sequence of TokenizedWord, according to some rules. After being called, the spans can be
    accessed via self.spans. This is a list of dictionaries. Each dictionary has a key of the entity class, and a
    list of TokenizedWord representing the entity
    """

    def __init__(self, threshold: float, text: str, id2label: Dict[int, str]):
        self.text = text
        self.threshold = threshold
        self.active_spans: List[Dict[str, List[TokenizedWord]]] = []
        self.words: List[TokenizedWord] = []
        self.span_breaking_chars = set("() ")
        self.non_breaking_span_chars = set("-")
        self.closed_spans: List[Dict[str, List[TokenizedWord]]] = []
        self.id2label = id2label

    def resolve_word(
        self, word: TokenizedWord
    ) -> Tuple[List[str], List[Optional[str]], List[float]]:
        """
        get a set of classes and a list of token confidences associated with a token within a word
        :param word:
        :return:
        """
        bio_labels = []
        class_labels = []
        confidences = []
        for i in range(len(word.token_confidences)):
            for bio_label, class_label, confidence_val in self.get_labels_under_threshold(
                word=word, label_index=i
            ):
                bio_labels.append(bio_label)
                class_labels.append(class_label)
                confidences.append(confidence_val)
        return bio_labels, class_labels, confidences

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

    def span_continue_condition(self, word: TokenizedWord, classes: List[Optional[str]]):
        """
        A potential entity span must continue if any of the following conditions are met:
        1. There are any class assignments in any of the tokens in the next word.
        2. The previous character to the word is in the set of self.non_breaking_span_chars
        3. The previous character to the word is not in the set of self.span_breaking_chars
        :param word:
        :return:
        """
        classes_set = set(classes)
        classes_set.discard(None)
        if (
            self.text[word.word_char_start - 1] not in self.span_breaking_chars
            or self.text[word.word_char_start - 1] in self.non_breaking_span_chars
            or len(classes_set) > 0
        ):
            return True
        else:
            return False

    def _update_active_spans(
        self, bio_labels: List[str], classes: List[Optional[str]], word: TokenizedWord
    ):
        """
        updates any active spans. If a B label is detected in an active span, make a copy and add to closed spans,
        as it's likely the start of another entity of the same class (but we still want to keep the original span open)
        :param bio_labels:
        :param classes:
        :param word:
        :return:
        """
        for spandict in self.active_spans:
            for bio, c in zip(bio_labels, classes):
                if bio == ENTITY_START_SYMBOL and c in spandict:
                    self.closed_spans.append({c: copy.deepcopy(spandict[c])})
                if c is not None:
                    spandict[c].append(word)

    def start_span(self, bio_labels: List[str], classes: List[Optional[str]], word: TokenizedWord):
        """
        start a new span if a B label is detected
        :param bio_labels:
        :param classes:
        :param word:
        :return:
        """
        di = defaultdict(list)
        for bio, clazz in zip(bio_labels, classes):
            if bio == ENTITY_START_SYMBOL and clazz is not None:
                di[clazz].append(word)
        if len(di) > 0:
            self.active_spans.append(di)

    def close_spans(self):
        """
        close any active spans
        :return:
        """
        for active_span in self.active_spans:
            if len(active_span) > 0:
                self.closed_spans.append(active_span)
        self.active_spans = []

    def process_next_word(self, word: TokenizedWord):
        """
        process the next word in the sequence, updating span information accordingly
        :param word:
        :return:
        """

        bio_labels, classes, confidences = self.resolve_word(word)
        if self.words:
            if self.span_continue_condition(word, classes):
                self._update_active_spans(bio_labels, classes, word)
                self.start_span(bio_labels=bio_labels, classes=classes, word=word)
            else:
                self.close_spans()
                self.start_span(bio_labels=bio_labels, classes=classes, word=word)
        else:
            self.start_span(bio_labels=bio_labels, classes=classes, word=word)
        self.words.append(word)

    def __call__(self, words: List[TokenizedWord]):
        for word in words:
            self.process_next_word(word)
        self.close_spans()


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
            [self.spans_to_entities(x, text, namespace) for x in span_finder.closed_spans]
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
            starts.append(word.word_char_start)
            ends.append(word.word_char_end)
        return min(starts), max(ends) + 1
