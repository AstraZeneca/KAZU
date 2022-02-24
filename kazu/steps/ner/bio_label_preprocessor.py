import logging
import traceback
from collections import defaultdict
from typing import List, Tuple, Dict

import pydash

from kazu.data.data import (
    TokenizedWord,
    Entity,
)

logger = logging.getLogger(__name__)


class SpanFinder:
    def __init__(self, threshold: float, text: str):
        self.text = text
        self.threshold = threshold
        self.class_word_map: Dict[str, List[TokenizedWord]] = defaultdict(list)
        self.words: List[TokenizedWord] = []
        self.span_breaking_chars = set("() ")
        self.span_contiguation_chars = set("-")
        self.spans: List[Dict[str, List[TokenizedWord]]] = []

    def span_end_condition(self, word: TokenizedWord):
        return all(clazz is None for clazz in word.classes)

    def span_continue_condition(self, word: TokenizedWord):
        class_count = len([x for x in word.classes if x is not None])
        prev_char = self.text[self.words[-1].end] if self.words[-1].end is not None else ""
        return (
            class_count > 0
            or prev_char not in self.span_breaking_chars
            or prev_char in self.span_contiguation_chars
        )

    def _update_span(self, word: TokenizedWord):
        for clazz in word.classes:
            if clazz is not None:
                self.class_word_map[clazz].append(word)
        self.words.append(word)

    def update(self, word: TokenizedWord):
        word.resolve(self.threshold)
        if self.words:
            if self.span_continue_condition(word):
                self._update_span(word)
            elif self.span_end_condition(word):
                self.spans.append(self.class_word_map)
                self.words.append(word)
                self.reset()
            else:
                print(word.render())
        else:
            self._update_span(word)

    def reset(self):
        self.class_word_map = defaultdict(list)


class TokenizedWordProcessor:
    """
    Because of the inherent obscurity of the inner workings of transformers, sometimes they produce BIO tags that
    don't correctly align to whole words. So what do we do? Fix it with rules :~Z

    This class is designed to work when an entire sequence of NER labels is known and therefore we can apply some
    post-processing logic - i.e. a hack until we have time to debug the model

    """

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    def __call__(self, words: List[TokenizedWord], text: str, namespace: str) -> List[Entity]:
        ents = []
        span_finder = SpanFinder(threshold=self.confidence_threshold, text=text)
        try:
            for word in words:
                span_finder.update(word)
            ents = pydash.flatten(
                [self.spans_to_entities(x, text, namespace) for x in span_finder.spans]
            )
        except Exception:
            print(traceback.format_exc())

        return ents

    def spans_to_entities(self, span_dict: Dict, text: str, namespace: str) -> List[Entity]:
        entities = []
        for class_label, offsets in span_dict.items():
            start, end = self.merge_offsets(offsets)
            entity = Entity.load_contiguous_entity(
                start=start,
                end=end,
                match=text[start:end],
                namespace=namespace,
                entity_class=class_label,
            )
            entities.append(entity)
        return entities

    def merge_offsets(self, words: List[TokenizedWord]) -> Tuple[int, int]:
        starts, ends = [], []
        for word in words:
            starts.append(word.start)
            ends.append(word.end)
        return min(starts), max(ends)
