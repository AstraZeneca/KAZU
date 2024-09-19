import abc
import logging
import re
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Iterable

import torch
from kazu.data import Entity, ENTITY_OUTSIDE_SYMBOL, ENTITY_START_SYMBOL
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class TokenizedWord:
    """A convenient container for a word, which may be split into multiple tokens by
    e.g. WordPiece tokenisation."""

    token_ids: list[int]
    #: token string representations
    tokens: list[str]
    #: tensor of token logit softmax
    token_confidences: Tensor
    #: character indices of tokens
    token_offsets: list[tuple[int, int]]
    #: char start index of word
    word_char_start: int
    #: char end index of word
    word_char_end: int
    word_id: int


@dataclass
class TokWordSpan:
    """Dataclass for a span (i.e. a list[TokenizedWord] representing an NE)"""

    #: entity_class
    clazz: str
    #: words associated with this span
    tok_words: list[TokenizedWord] = field(default_factory=list)


class SpanFinder(ABC):
    def __init__(self, text: str, id2label: dict[int, str]) -> None:
        """

        :param text: the raw text to be processed
        :param id2label: BIO to class label mappings
        """
        self.text = text
        self.active_spans: list[TokWordSpan] = []
        self.words: list[TokenizedWord] = []
        self.span_breaking_chars = {"(", ")", ";"}
        self.closed_spans: list[TokWordSpan] = []
        self.id2label = id2label

    @abc.abstractmethod
    def __call__(self, words: list[TokenizedWord]) -> list[TokWordSpan]:
        """Find spans of entities.

        :param words:
        :return: The spans found
        """
        pass


class SimpleSpanFinder(SpanFinder):
    """Since Bert like models use wordpiece tokenisers to handle the OOV problem, we
    need to reconstitute this info back into document character indices to represent
    actual NEs.

    Since transformer NER is an imprecise art, we may want to use
    different logic in how this works, so we can subclass this class to
    determine how this should be done

    The __call__ method of this class operates on a list of
    TokenizedWord, processing each sequentially according to a logic
    determined by the implementing class. It returns the spans found
    - a list of :class:`~TokWordSpan`\\ .

    After being called, the spans can be later accessed via
    self.closed_spans.
    """

    def __init__(self, text: str, id2label: dict[int, str]):
        """

        :param text: the raw text to be processed
        :param id2label: id to BIO-class label mappings
        """
        super().__init__(text, id2label)

    def __call__(self, words: list[TokenizedWord]) -> list[TokWordSpan]:
        """Find spans of entities.

        :param words:
        :return: The spans found
        """
        for word in words:
            self.process_next_word(word)
        self.close_spans()
        return self.closed_spans

    def _update_active_spans(
        self, bio_and_class_labels: set[tuple[str, Optional[str]]], word: TokenizedWord
    ) -> None:
        """Updates any active spans. If a B label is detected in an active span, make a
        copy and add to closed spans, as it's likely the start of another entity of the
        same class (but we still want to keep the original span open)

        :param bio_and_class_labels: BIO and optional class label set
        :param word:
        :return:
        """
        for span in self.active_spans:
            for bio, c in bio_and_class_labels:
                if bio == ENTITY_START_SYMBOL and c == span.clazz:
                    # a new word of the same class is beginning, so add a copy of the currently open TokWordSpan to the
                    # closed spans
                    self.closed_spans.append(deepcopy(span))
                if c == span.clazz:
                    # the word is the same class as the active TokWordSpan, so extend it
                    span.tok_words.append(word)

    def start_span(
        self,
        bio_and_class_labels: set[tuple[str, Optional[str]]],
        word: TokenizedWord,
    ) -> None:
        """Start a new TokWordSpan if a B label is detected.

        :param bio_and_class_labels:
        :param word:
        :return:
        """
        for bio, clazz in bio_and_class_labels:
            if bio == ENTITY_START_SYMBOL and clazz is not None:
                span = TokWordSpan(clazz=clazz, tok_words=[word])
                self.active_spans.append(span)

    def close_spans(self) -> None:
        """Close any active spans."""
        for active_span in self.active_spans:
            if len(active_span.tok_words) > 0:
                self.closed_spans.append(active_span)
        self.active_spans = []

    def span_continue_condition(
        self, word: TokenizedWord, bio_and_class_labels: set[tuple[str, Optional[str]]]
    ) -> bool:
        """A potential entity span will end if any of the following conditions
        are met:

        1. any of the BIO classes for word are O
        2. The previous character to the word is in the set of self.span_breaking_chars

        :param word:
        :param bio_and_class_labels:
        :return:
        """
        classes_set = set(x[1] for x in bio_and_class_labels)
        if None in classes_set or self.text[word.word_char_start - 1] in self.span_breaking_chars:
            return False
        return True

    def get_bio_and_class_labels(self, word: TokenizedWord) -> set[tuple[str, Optional[str]]]:
        """Return a set of tuple[<BIO label>,Optional[<class label>]] for a
        TokenizedWord. Optional[<class label>] is None if the BIO label is "O".

        :param word:
        :return:
        """
        bio_and_class_labels: set[tuple[str, Optional[str]]] = set()
        most_conf_index_per_token = torch.argmax(word.token_confidences, dim=1)
        for confidence_index in most_conf_index_per_token:
            bio_label = self.id2label[confidence_index.item()]
            if bio_label == ENTITY_OUTSIDE_SYMBOL:
                bio_and_class_labels.add(
                    (
                        bio_label,
                        None,
                    )
                )
            else:
                bio_label, class_label = bio_label.split("-")
                bio_and_class_labels.add(
                    (
                        bio_label,
                        class_label,
                    )
                )

        return bio_and_class_labels

    def process_next_word(self, word: TokenizedWord) -> None:
        """Process the next word in the sequence, updating span information accordingly.

        :param word:
        :return:
        """

        bio_and_class_labels = self.get_bio_and_class_labels(word)
        if not self.words or len(self.active_spans) == 0:
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word)
        elif self.span_continue_condition(word, bio_and_class_labels):
            self._update_active_spans(bio_and_class_labels, word)
        else:
            self.close_spans()
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word)
        self.words.append(word)


class MultilabelSpanFinder(SpanFinder):
    """A span finder that can handle multiple labels per token, as opposed to the
    standard 'BIO' format."""

    def __init__(self, text: str, id2label: dict[int, str]):
        """

        :param text: the raw text to be processed
        :param id2label: id to class label mappings
        """
        super().__init__(text, id2label)
        self.active_labels: set[str] = set()

    def _update_active_spans(self, class_label: str, word: TokenizedWord) -> None:
        """Updates any active spans.

        :param class_label: class label to update in active spans
        :param word: TokenizedWord to append to active spans
        """
        for span in self.active_spans:
            if class_label == span.clazz:
                # the word is the same class as the active TokWordSpan, so extend it
                span.tok_words.append(word)

    def start_span(self, class_label: str, word: TokenizedWord) -> None:
        """Start a new TokWordSpan for the given class label.

        :param class_label: the label to use
        :param word: the word to start the span with
        :return:
        """
        span = TokWordSpan(clazz=class_label, tok_words=[word])
        self.active_spans.append(span)
        self.active_labels.add(class_label)

    def close_spans(self, class_label: str) -> None:
        """Close any active spans."""
        for active_span in list(self.active_spans):
            if class_label == active_span.clazz:
                self.closed_spans.append(active_span)
                self.active_spans.remove(active_span)
                self.active_labels.remove(class_label)

    def __call__(self, words: list[TokenizedWord]) -> list[TokWordSpan]:
        """Find spans of entities.

        :param words:
        :return: The spans found
        """
        for word in words:
            self.process_next_word(word)
        for clazz in set(self.active_labels):
            self.close_spans(clazz)
        return self.closed_spans

    def get_class_labels(self, word: TokenizedWord) -> set[str]:
        class_labels: set[str] = set()
        indices_of_confs_above_threshold = torch.argwhere(word.token_confidences > 0)
        # the 0th dimension is the order of the tokens in the word, which we don't need.
        # The first dimension represents which label the confidence score relates to.
        label_indices_above_threshold = indices_of_confs_above_threshold[:, 1]
        for label_index in label_indices_above_threshold:
            class_label = self.id2label[label_index.item()]
            if class_label != ENTITY_OUTSIDE_SYMBOL:
                class_labels.add(class_label)

        return class_labels

    def span_continue_condition(self, word: TokenizedWord, class_labels: set[str]) -> bool:
        """A potential entity span will end if any of the following conditions
        are met:

        1. any of the BIO classes for word are O
        2. The previous character to the word is in the set of self.span_breaking_chars

        :param word:
        :param bio_and_class_labels:
        :return:
        """
        if not class_labels or self.text[word.word_char_start - 1] in self.span_breaking_chars:
            return False
        return True

    def process_next_word(self, word: TokenizedWord) -> None:
        """Process the next word in the sequence, updating span information accordingly.

        :param word:
        :return:
        """

        class_labels = self.get_class_labels(word)
        new_labels = class_labels.difference(self.active_labels)
        for new_label in new_labels:
            self.start_span(new_label, word)
        for finished_label in self.active_labels.difference(class_labels):
            self.close_spans(finished_label)

        current_labels = class_labels.intersection(self.active_labels).difference(new_labels)
        if self.span_continue_condition(word, current_labels):
            for current_label in current_labels:
                self._update_active_spans(current_label, word)

        self.words.append(word)


class TokenizedWordProcessor:
    """Because of the inherent obscurity of the inner workings of transformers,
    sometimes they produce BIO tags that don't correctly align to whole words, or maybe
    the classic BIO format gets confused by nested entities.

    This class is designed to work when an entire sequence of NER labels is known and
    therefore we can apply some post-processing logic. Namely, we use the SpanFinder
    class to find entity spans according to their internal logic
    """

    def __init__(
        self,
        labels: Iterable[str],
        use_multilabel: bool = False,
        strip_re: Optional[dict[str, str]] = None,
    ):
        """

        :param labels: mapping of label int id to str label
        :param use_multilabel: whether to use multilabel classification (needs to be supported by the model)
        :param strip_re: an optional dict of {<entity_class>:<python regex to remove>} to process NER results that the
            model frequently misclassifies.
        """

        self.use_multilabel = use_multilabel
        self.id2label = self.id2labels_from_label_list(labels)
        self.strip_re = (
            {k: re.compile(v) for k, v in strip_re.items()} if strip_re is not None else None
        )

    @staticmethod
    def id2labels_from_label_list(labels: Iterable[str]) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(labels)}

    def _make_simple_span_finder(self, text: str) -> SimpleSpanFinder:
        return SimpleSpanFinder(text=text, id2label=self.id2label)

    def _make_multilabel_span_finder(self, text: str) -> MultilabelSpanFinder:
        return MultilabelSpanFinder(text=text, id2label=self.id2label)

    def make_span_finder(self, text: str) -> SpanFinder:
        if self.use_multilabel:
            return self._make_multilabel_span_finder(text)
        return self._make_simple_span_finder(text)

    def __call__(self, words: list[TokenizedWord], text: str, namespace: str) -> list[Entity]:
        span_finder: SpanFinder = self.make_span_finder(text)
        spans = span_finder(words)
        ents = self.spans_to_entities(spans, text, namespace)
        return ents

    def spans_to_entities(
        self, spans: list[TokWordSpan], text: str, namespace: str
    ) -> list[Entity]:
        """Convert spans to instances of Entity, adding in namespace info as
        appropriate.

        :param spans: list of TokWordSpan to consider
        :param text: original text
        :param namespace: namespace to add to Entity
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
            match_str, end = self.attempt_strip_suffixes(start, end, match_str, span.clazz)

            entity = Entity.load_contiguous_entity(
                start=start,
                end=end,
                match=match_str,
                namespace=namespace,
                entity_class=span.clazz,
            )
            entities.append(entity)

        return entities

    def calculate_span_offsets(self, words: list[TokenizedWord]) -> tuple[int, int]:
        starts, ends = [], []
        for word in words:
            starts.append(word.word_char_start)
            ends.append(word.word_char_end)
        return min(starts), max(ends) + 1

    def attempt_strip_suffixes(
        self, start: int, end: int, match_str: str, clazz: str
    ) -> tuple[str, int]:
        """Transformers sometimes get confused about precise offsets, depending on the
        training data (e.g. "COX2" is often classified as "COX2 gene"). This method
        offers light post-processing to correct these, for better entity linking
        results.

        :param start: original start
        :param end: original end
        :param match_str: original string
        :param clazz: entity class
        :return: new string, new end
        """
        if self.strip_re is not None:
            suffixes_re = self.strip_re.get(clazz)
            if suffixes_re is not None:
                match_str = suffixes_re.sub("", match_str)
                end = len(match_str) + start
        return match_str, end
