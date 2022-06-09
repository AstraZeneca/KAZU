import copy
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable, Optional, Set

import torch
from kazu.data.data import (
    Entity,
    ENTITY_OUTSIDE_SYMBOL,
    ENTITY_START_SYMBOL,
    IS_SUBSPAN,
)
from torch import Tensor

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


@dataclass
class TokWordSpan:
    clazz: str
    subspan: bool
    tok_words: List[TokenizedWord] = field(default_factory=list)


class SpanFinder:
    """
    finds spans across a sequence of TokenizedWord, according to some rules. After being called, the spans can be
    accessed via self.spans. This is a list of dictionaries. Each dictionary has a key of the entity class, and a
    list of TokenizedWord representing the entity
    """

    def __init__(self, text: str, id2label: Dict[int, str]):
        self.text = text
        self.active_spans: List[TokWordSpan] = []
        self.words: List[TokenizedWord] = []
        self.span_breaking_chars = set("() ;")
        self.non_breaking_span_chars = set("-")
        self.closed_spans: List[TokWordSpan] = []
        self.id2label = id2label

    def resolve_word(self, word: TokenizedWord) -> Set[Tuple[str, Optional[str]]]:
        """
        get a set of classes and a list of token confidences associated with a token within a word
        :param word:
        :return:
        """
        bio_and_class_labels = set()
        for i in range(len(word.token_confidences)):
            for bio_label, class_label, confidence_val in self.get_labels(word=word, label_index=i):
                bio_and_class_labels.add(
                    (
                        bio_label,
                        class_label,
                    )
                )
        return bio_and_class_labels

    def get_labels(
        self, word: TokenizedWord, label_index: int
    ) -> Iterable[Tuple[str, Optional[str], float]]:
        """
        for a given word, yield a BIO label, class label and confidence val based upon some logic
        :param word:
        :param label_index:
        :return:
        """

        raise NotImplementedError()

    def span_continue_condition(
        self, word: TokenizedWord, bio_and_class_labels: Set[Tuple[str, Optional[str]]]
    ):
        """
        based upon some logic, determine whether a span should continue or not
        :param word:
        :param bio_and_class_labels:
        :return:
        """
        raise NotImplementedError()

    def _update_active_spans(
        self, bio_and_class_labels: Set[Tuple[str, Optional[str]]], word: TokenizedWord
    ):
        """
        updates any active spans. If a B label is detected in an active span, make a copy and add to closed spans,
        as it's likely the start of another entity of the same class (but we still want to keep the original span open)
        :param bio_labels:
        :param classes:
        :param word:
        :return:
        """
        for span in self.active_spans:
            for bio, c in bio_and_class_labels:
                if bio == ENTITY_START_SYMBOL and c == span.clazz:
                    # a new word of the same class is beginning, so add a copy of the currently open span to the
                    # closed spans
                    self.closed_spans.append(copy.deepcopy(span))
                if c == span.clazz:
                    # the word is the same class as the active span, so extend it
                    span.tok_words.append(word)

    def start_span(
        self,
        bio_and_class_labels: Set[Tuple[str, Optional[str]]],
        word: TokenizedWord,
        subspan: bool,
    ):
        """
        start a new span if a B label is detected
        :param bio_labels:
        :param classes:
        :param word:
        :return:
        """
        lst = []
        for bio, clazz in bio_and_class_labels:
            if bio == ENTITY_START_SYMBOL and clazz is not None:
                span = TokWordSpan(clazz=clazz, subspan=subspan, tok_words=[word])
                lst.append(span)
        if len(lst) > 0:
            self.active_spans.extend(lst)

    def close_spans(self):
        """
        close any active spans
        :return:
        """
        for active_span in self.active_spans:
            if len(active_span.tok_words) > 0:
                self.closed_spans.append(active_span)
        self.active_spans = []

    def process_next_word(self, word: TokenizedWord):
        """
        process the next word in the sequence, according to some logic
        :param word:
        :return:
        """

        raise NotImplementedError()

    def __call__(self, words: List[TokenizedWord]):
        for word in words:
            self.process_next_word(word)
        self.close_spans()


class SimpleSpanFinder(SpanFinder):
    """
    finds spans across a sequence of TokenizedWord, according to some rules. After being called, the spans can be
    accessed via self.spans. This is a list of dictionaries. Each dictionary has a key of the entity class, and a
    list of TokenizedWord representing the entity
    """

    def __init__(self, text: str, id2label: Dict[int, str]):

        super().__init__(text, id2label)

    def get_labels(
        self, word: TokenizedWord, label_index: int
    ) -> Iterable[Tuple[str, Optional[str], float]]:
        """
        for a given word, yield a BIO label, class label and confidence val the most confident result
        :param word:
        :param label_index:
        :return:
        """
        confidences_indices_sorted = torch.argsort(
            word.token_confidences[label_index], dim=-1, descending=True
        )
        for confidence_index in confidences_indices_sorted:
            confidence_val: float = word.token_confidences[label_index][confidence_index].item()
            bio_label = self.id2label[confidence_index.item()]
            if bio_label == ENTITY_OUTSIDE_SYMBOL:
                yield bio_label, None, confidence_val
            else:
                bio_label, class_label = bio_label.split("-")
                yield bio_label, class_label, confidence_val
            break

    def span_continue_condition(
        self, word: TokenizedWord, bio_and_class_labels: Set[Tuple[str, Optional[str]]]
    ):
        """
        A potential entity span must continue if any of the following conditions are met:
        1. There are any class assignments in any of the tokens in the next word.
        2. The previous character to the word is in the set of self.non_breaking_span_chars
        3. The previous character to the word is not in the set of self.span_breaking_chars
        4. the most confident label in the word under consideration is not None
        :param word:
        :return:
        """
        classes_set = set([x[1] for x in bio_and_class_labels])
        if None in classes_set:
            return False
        else:
            if (
                self.text[word.word_char_start - 1] not in self.span_breaking_chars
                or self.text[word.word_char_start - 1] in self.non_breaking_span_chars
                or len(classes_set) > 0
            ):
                return True
            else:
                return False

    def process_next_word(self, word: TokenizedWord):
        """
        process the next word in the sequence, updating span information accordingly
        :param word:
        :return:
        """

        bio_and_class_labels = self.resolve_word(word)
        if self.words:
            if len(self.active_spans) == 0:
                self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
            elif self.span_continue_condition(word, bio_and_class_labels):
                self._update_active_spans(bio_and_class_labels, word)
            else:
                self.close_spans()
                self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
        else:
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
        self.words.append(word)


class SmartSpanFinder(SpanFinder):
    """
    finds spans across a sequence of TokenizedWord, according to some rules. After being called, the spans can be
    accessed via self.spans. This is a list of dictionaries. Each dictionary has a key of the entity class, and a
    list of TokenizedWord representing the entity
    """

    def __init__(self, threshold: Optional[float], text: str, id2label: Dict[int, str]):
        super().__init__(text, id2label)
        self.text = text
        self.threshold = threshold
        self.active_spans: List[TokWordSpan] = []
        self.words: List[TokenizedWord] = []
        self.span_breaking_chars = set("() ;")
        self.non_breaking_span_chars = set("-")
        self.closed_spans: List[TokWordSpan] = []
        self.id2label = id2label

    def get_labels(
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
            if self.threshold is not None:
                if confidence_val > self.threshold:
                    bio_label = self.id2label[confidence_index.item()]
                    if bio_label == ENTITY_OUTSIDE_SYMBOL:
                        yield bio_label, None, confidence_val
                    else:
                        bio_label, class_label = bio_label.split("-")
                        yield bio_label, class_label, confidence_val
                else:
                    break
            else:
                bio_label = self.id2label[confidence_index.item()]
                if bio_label == ENTITY_OUTSIDE_SYMBOL:
                    yield bio_label, None, confidence_val
                else:
                    bio_label, class_label = bio_label.split("-")
                    yield bio_label, class_label, confidence_val
                break

    def span_continue_condition(
        self, word: TokenizedWord, bio_and_class_labels: Set[Tuple[str, Optional[str]]]
    ):
        """
        A potential entity span must continue if any of the following conditions are met:
        1. There are any class assignments in any of the tokens in the next word.
        2. The previous character to the word is in the set of self.non_breaking_span_chars
        3. The previous character to the word is not in the set of self.span_breaking_chars
        :param word:
        :return:
        """
        classes_set = set([x[1] for x in bio_and_class_labels])
        classes_set.discard(None)
        if (
            self.text[word.word_char_start - 1] not in self.span_breaking_chars
            or self.text[word.word_char_start - 1] in self.non_breaking_span_chars
            or len(classes_set) > 0
        ):
            return True
        else:
            return False

    def process_next_word(self, word: TokenizedWord):
        """
        process the next word in the sequence, updating span information accordingly
        :param word:
        :return:
        """

        bio_and_class_labels = self.resolve_word(word)
        if self.words:
            if len(self.active_spans) == 0:
                self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
            elif self.span_continue_condition(word, bio_and_class_labels):
                self._update_active_spans(bio_and_class_labels, word)
                self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=True)
            else:
                self.close_spans()
                self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
        else:
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
        self.words.append(word)

    def __call__(self, words: List[TokenizedWord]):
        for word in words:
            self.process_next_word(word)
        self.close_spans()


class TokenizedWordProcessor:
    """
    Because of the inherent obscurity of the inner workings of transformers, sometimes they produce BIO tags that
    don't correctly align to whole words, or maybe the classic BIO format gets confused by nested entities.

    This class is designed to work when an entire sequence of NER labels is known and therefore we can apply some
    post-processing logic. Namely, we use the SpanFinder class to assign soft labels based upon some threshold, rather
    hard labels as per softmax. Additionally, we define some conditions in which an entity must always break, or
    must always continue

    """

    def __init__(
        self,
        confidence_threshold: Optional[float],
        id2label: Dict[int, str],
        detect_subspans: bool = False,
    ):
        self.detect_subspans = detect_subspans
        self.id2label = id2label
        self.confidence_threshold = confidence_threshold

    def __call__(self, words: List[TokenizedWord], text: str, namespace: str) -> List[Entity]:
        span_finder: SpanFinder
        if self.detect_subspans:
            span_finder = SmartSpanFinder(
                threshold=self.confidence_threshold,
                text=text,
                id2label=self.id2label,
            )
        else:
            span_finder = SimpleSpanFinder(text=text, id2label=self.id2label)
        span_finder(words)
        ents = self.spans_to_entities(span_finder.closed_spans, text, namespace)
        return ents

    def spans_to_entities(
        self, spans: List[TokWordSpan], text: str, namespace: str
    ) -> List[Entity]:
        """
        convert spans to instances of Entity, adding in namespace info as appropriate
        :param span_dict:
        :param text:
        :param namespace:
        :return:
        """
        entities = []
        for span in spans:
            start, end = self.calculate_span_offsets(span.tok_words)
            # sometimes the tokenizer seems to mess up the offsets
            match_str = text[start:end]
            if (
                len(match_str) > 1
                and match_str[-1] == " "
                and any(char.isalpha() for char in match_str)
            ):
                end = end - 1
            match_str = text[start:end]
            entity = Entity.load_contiguous_entity(
                start=start,
                end=end,
                match=match_str,
                namespace=namespace,
                entity_class=span.clazz,
                metadata={IS_SUBSPAN: span.subspan},
            )
            entities.append(entity)

        return entities

    def calculate_span_offsets(self, words: List[TokenizedWord]) -> Tuple[int, int]:
        starts, ends = [], []
        for word in words:
            starts.append(word.word_char_start)
            ends.append(word.word_char_end)
        return min(starts), max(ends) + 1
