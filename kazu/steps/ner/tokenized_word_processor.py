import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from kazu.data import Entity, ENTITY_OUTSIDE_SYMBOL, ENTITY_START_SYMBOL, IS_SUBSPAN
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
    #: is this a subspan? only relevant if using SmartSpanFinder
    subspan: Optional[bool] = None
    #: words associated with this span
    tok_words: list[TokenizedWord] = field(default_factory=list)


class SpanFinder(ABC):
    """Since Bert like models use wordpiece tokenisers to handle the OOV problem, we
    need to reconstitute this info back into document character indices to represent
    actual NEs.

    Since transformer NER is an inprecise art, we may want to use
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
        :param id2label: BIO to class label mappings
        """
        self.text = text
        self.active_spans: list[TokWordSpan] = []
        self.words: list[TokenizedWord] = []
        self.span_breaking_chars = {"(", ")", ";"}
        self.closed_spans: list[TokWordSpan] = []
        self.id2label = id2label

    @abstractmethod
    def get_bio_and_class_labels(self, word: TokenizedWord) -> set[tuple[str, Optional[str]]]:
        """Return a set of tuple[<BIO label>,Optional[<class label>]] for a
        TokenizedWord. Optional[<class label>] is None if the BIO label is "O".

        :param word:
        :return:
        """
        pass

    @abstractmethod
    def span_continue_condition(
        self, word: TokenizedWord, bio_and_class_labels: set[tuple[str, Optional[str]]]
    ) -> bool:
        """Based upon some logic, determine whether a span should continue or not.

        :param word:
        :param bio_and_class_labels:
        :return:
        """
        pass

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
        subspan: Optional[bool],
    ) -> None:
        """Start a new TokWordSpan if a B label is detected.

        :param bio_and_class_labels:
        :param word:
        :param subspan:
        :return:
        """
        for bio, clazz in bio_and_class_labels:
            if bio == ENTITY_START_SYMBOL and clazz is not None:
                span = TokWordSpan(clazz=clazz, subspan=subspan, tok_words=[word])
                self.active_spans.append(span)

    def close_spans(self):
        """Close any active spans."""
        for active_span in self.active_spans:
            if len(active_span.tok_words) > 0:
                self.closed_spans.append(active_span)
        self.active_spans = []

    @abstractmethod
    def process_next_word(self, word: TokenizedWord) -> None:
        """Process the next word in the sequence, according to some logic.

        :param word:
        :return:
        """
        pass

    def __call__(self, words: list[TokenizedWord]) -> list[TokWordSpan]:
        """Find spans of entities.

        :param words:
        :return: The spans found
        """
        for word in words:
            self.process_next_word(word)
        self.close_spans()
        return self.closed_spans


class SimpleSpanFinder(SpanFinder):
    """A basic implementation of SpanFinder.

    Uses highest confidence labels only
    """

    def __init__(self, text: str, id2label: dict[int, str]):

        super().__init__(text, id2label)

    def get_bio_and_class_labels(self, word: TokenizedWord) -> set[tuple[str, Optional[str]]]:
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
        else:
            return True

    def process_next_word(self, word: TokenizedWord) -> None:
        """Process the next word in the sequence, updating span information accordingly.

        :param word:
        :return:
        """

        bio_and_class_labels = self.get_bio_and_class_labels(word)
        if not self.words or len(self.active_spans) == 0:
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=None)
        elif self.span_continue_condition(word, bio_and_class_labels):
            self._update_active_spans(bio_and_class_labels, word)
        else:
            self.close_spans()
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=None)
        self.words.append(word)


class SmartSpanFinder(SpanFinder):
    """
    A more complicated implementation of SpanFinder using 'soft labels' - i.e. nested and overlapping entities of
    different classes may be produced, if the associated label is above some threshold
    """

    def __init__(self, threshold: float, text: str, id2label: dict[int, str]):
        super().__init__(text, id2label)
        self.text = text
        self.threshold = threshold
        # we include a whitespace here so that new spans (potential entities) are started for each word
        self.span_breaking_chars = {"(", ")", ";", " "}

    def get_bio_and_class_labels(self, word: TokenizedWord) -> set[tuple[str, Optional[str]]]:
        """Returns bio and class labels if their confidence is above the configured
        threshold.

        :param word:
        :return:
        """
        bio_and_class_labels: set[tuple[str, Optional[str]]] = set()
        indices_of_confs_above_threshold = torch.argwhere(word.token_confidences > self.threshold)
        # the 0th dimension is the order of the tokens in the word, which we don't need.
        # The first dimension represents which label the confidence score relates to.
        label_indices_above_threshold = indices_of_confs_above_threshold[:, 1]
        for label_index in label_indices_above_threshold:
            bio_label = self.id2label[label_index.item()]
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

    def span_continue_condition(
        self, word: TokenizedWord, bio_and_class_labels: set[tuple[str, Optional[str]]]
    ) -> bool:
        """A potential entity span must continue if any of the following
        conditions are met:

        1. The previous character to the word is not in the set of self.span_breaking_chars
        2. There are any entity class assignments in any of the tokens in the TokenizedWord under consideration.

        :param word:
        :param bio_and_class_labels:
        :return:
        """
        classes_set = set(x[1] for x in bio_and_class_labels)
        classes_set.discard(None)
        if (
            self.text[word.word_char_start - 1] not in self.span_breaking_chars
            or len(classes_set) > 0
        ):
            return True
        else:
            return False

    def process_next_word(self, word: TokenizedWord) -> None:
        """Process the next word in the sequence, updating span information accordingly.

        :param word:
        :return:
        """

        bio_and_class_labels = self.get_bio_and_class_labels(word)
        if not self.words or len(self.active_spans) == 0:
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)
        elif self.span_continue_condition(word, bio_and_class_labels):
            self._update_active_spans(bio_and_class_labels, word)
            # we start a new span here for each new word if B is detected as a soft label
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=True)
        else:
            self.close_spans()
            self.start_span(bio_and_class_labels=bio_and_class_labels, word=word, subspan=False)

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
        confidence_threshold: Optional[float],
        id2label: dict[int, str],
        detect_subspans: bool = False,
        strip_re: Optional[dict[str, str]] = None,
    ):
        """

        :param confidence_threshold: optional threshold if using :class:`SmartSpanFinder`\\ . Ignored if detect_subspans is false
        :param id2label: mapping of label int id to str label
        :param detect_subspans: use :class:`SmartSpanFinder` if True. A confidence_threshold must be provided
        :param strip_re: an optional dict of {<entity_class>:<python regex to remove>} to process NER results that the
            model frequently misclassifies.
        """

        self.id2label = id2label
        self.detect_subspans = detect_subspans
        self.confidence_threshold = confidence_threshold
        self.strip_re = (
            {k: re.compile(v) for k, v in strip_re.items()} if strip_re is not None else None
        )

    def _make_smart_span_finder(self, text: str) -> SmartSpanFinder:
        logger.debug(
            "detect_subspans is %s. Using %s, confidence threshold: %s",
            self.detect_subspans,
            SmartSpanFinder.__name__,
            self.confidence_threshold,
        )
        if isinstance(self.confidence_threshold, float):
            return SmartSpanFinder(
                threshold=self.confidence_threshold,
                text=text,
                id2label=self.id2label,
            )
        else:
            raise ValueError(
                f"tried to instantiate {SmartSpanFinder.__name__} but confidence threshold is not a float"
            )

    def _make_simple_span_finder(self, text: str) -> SimpleSpanFinder:
        logger.debug(
            "detect_subspans is %s. Using %s",
            self.detect_subspans,
            SimpleSpanFinder.__name__,
        )
        return SimpleSpanFinder(text=text, id2label=self.id2label)

    def make_span_finder(self, text: str) -> SpanFinder:
        if self.detect_subspans:
            return self._make_smart_span_finder(text)
        else:
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
            if span.subspan is not None:
                metadata = {IS_SUBSPAN: span.subspan}
            else:
                metadata = {}

            match_str, end = self.attempt_strip_suffixes(start, end, match_str, span.clazz)

            entity = Entity.load_contiguous_entity(
                start=start,
                end=end,
                match=match_str,
                namespace=namespace,
                entity_class=span.clazz,
                metadata=metadata,
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
