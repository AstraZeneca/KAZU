import copy
import logging
from typing import List, Callable, Tuple, Dict, Optional

from spacy.matcher import Matcher
from spacy.tokens import Token, Span

from kazu.data.data import Entity
from kazu.utils.spacy_pipeline import SpacyPipeline

logger = logging.getLogger(__name__)


def _copy_ent_with_new_spans(
    entity: Entity,
    spans: List[Tuple[int, int]],
    text: str,
    join_str: str,
    rule_name: Optional[str] = None,
):
    attrb_dict = copy.deepcopy(entity).__dict__
    attrb_dict.pop("spans")
    attrb_dict.pop("match")
    attrb_dict.pop("start")
    attrb_dict.pop("end")
    ent = Entity.from_spans(spans=spans, text=text, join_str=join_str, **attrb_dict)
    if rule_name:
        ent.metadata["split_rule"] = rule_name
    return ent


class SplitOnConjunctionPattern:
    def __init__(self, spacy_pipeline: SpacyPipeline):
        """
        analyse
        :param pattern:
        """
        self.nlp = spacy_pipeline.nlp
        self.matcher = Matcher(self.nlp.vocab)
        patterns = [
            [
                {"POS": "ADJ", "OP": "*"},
                {"POS": "NOUN", "OP": "*"},
                {"POS": "PUNCT", "OP": "*"},
                {"POS": "ADJ", "OP": "*"},
                {"POS": "NOUN"},
                {"POS": "CCONJ"},
                {"POS": "ADJ", "OP": "*"},
                {"POS": "NOUN"},
                {"POS": "NOUN"},
            ],
            [
                {"POS": "NOUN", "OP": "*"},
                {"POS": "PUNCT", "OP": "*"},
                {"POS": "NOUN"},
            ],
        ]
        for i, pattern in enumerate(patterns):
            self.matcher.add(f"{self.__class__.__name__}_{i}", [pattern], greedy="LONGEST")

    def __call__(self, entity: Entity, text: str) -> List[Entity]:
        # if any((x in entity.match for x in [" and ", " or ", " nor ", ", ", "/"])):
        doc = self.nlp(entity.match)
        ents = []
        if any((x in entity.match for x in [" and ", " or ", " nor "])):
            ents = self.run_conjunction_rules(doc, entity, text)
        return ents

    def run_conjunction_rules(self, doc, entity, text):
        # high precision rules for splitting conjunctions
        ents = []
        matches = self.matcher(doc)
        if len(matches) == 1:
            _, start, end = matches[0]
            span = doc[start:end]
            toks = list(span)
            anchor = toks[-1]
            noun_chunks = list(span.noun_chunks)
            if len(noun_chunks) > 1:
                # noun chunks detected, use them
                ents = self.process_noun_chunks(anchor, entity, noun_chunks, text)
            if len(ents) < 2:
                ents = self.process_pos_tags(anchor, entity, text, toks)
        return ents

    def process_pos_tags(self, anchor: Token, entity: Entity, text: str, toks: List[Token]):
        ents = []
        nouns = list(filter(lambda x: x.pos_ == "NOUN", toks[:-1]))
        for noun in nouns:
            ents.append(
                _copy_ent_with_new_spans(
                    spans=[
                        (entity.start + noun.idx, entity.start + len(noun) + noun.idx),
                        (entity.start + anchor.idx, entity.start + len(anchor) + anchor.idx),
                    ],
                    text=text,
                    entity=entity,
                    join_str=" ",
                    rule_name="spacy pos tags",
                )
            )
        return ents

    def process_noun_chunks(
        self, anchor: Token, entity: Entity, noun_chunks: List[Span], text: str
    ):
        ents = []
        for chunk in noun_chunks[:-1]:
            ents.append(
                _copy_ent_with_new_spans(
                    spans=[
                        (entity.start + chunk.start_char, entity.start + chunk.end_char),
                        (entity.start + anchor.idx, len(anchor) + entity.start + anchor.idx),
                    ],
                    text=text,
                    entity=entity,
                    join_str=" ",
                    rule_name="spacy noun chunker",
                )
            )
        ents.append(
            _copy_ent_with_new_spans(
                spans=[(noun_chunks[-1].start_char, noun_chunks[-1].end_char)],
                text=text,
                entity=entity,
                join_str=" ",
                rule_name=self.__class__.__name__,
            )
        )
        return ents


class SplitOnNumericalListPatternWithPrefix:
    """
    split a string of numerically incrementing parts, e.g.
    BRACA1/2 ->
    [
        Entity(match="BRACA1"),
        Entity(match="BRACA2")
    ]
    """

    def __init__(self, pattern: str = "/"):
        """
        pattern to split the string on (typically "/")
        :param pattern:
        """
        self.pattern = pattern

    def __call__(self, entity: Entity, text: str) -> List[Entity]:
        input_ent_start = entity.start
        parts = entity.match.split(self.pattern)
        new_ents = []
        if len(parts) > 1:
            i = None
            for i, char in enumerate(reversed(list(parts[0]))):
                if not char.isdigit():
                    break

            new_ents.append(
                _copy_ent_with_new_spans(
                    entity=entity,
                    spans=[(input_ent_start, input_ent_start + len(parts[0]))],
                    text=text,
                    join_str="",
                    rule_name=self.__class__.__name__,
                )
            )
            if i:
                prefix = parts[0][: len(parts[0]) - i]
                span_offset = len(parts[0]) + len(self.pattern)
                for part in parts[1:]:
                    if not part.isdigit():
                        continue
                    else:
                        new_spans = [
                            (input_ent_start, input_ent_start + len(prefix)),
                            (span_offset, span_offset + len(part)),
                        ]
                        span_offset += len(self.pattern) + len(part)
                        new_ents.append(
                            _copy_ent_with_new_spans(
                                entity=entity,
                                spans=new_spans,
                                text=text,
                                join_str="",
                                rule_name=self.__class__.__name__,
                            )
                        )
        return new_ents


class NonContiguousEntitySplitter:
    """
    Some simple rules to split non-contiguous entities into component entities
    """

    def __init__(self, entity_conditions: Dict[str, List[Callable[[Entity, str], List[Entity]]]]):

        self.entity_conditions = entity_conditions

    def __call__(self, entity: Entity, text: str) -> List[Entity]:
        existing_offsets = {
            (
                entity.start,
                entity.end,
            )
        }

        new_ents = []
        for rule in self.entity_conditions.get(entity.entity_class, []):
            found_ents = rule(entity, text)
            for found_ent in found_ents:
                # only add new ent if offsets have changed

                if (
                    found_ent.start,
                    found_ent.end,
                ) not in existing_offsets:
                    new_ents.append(found_ent)
                    existing_offsets.add(
                        (
                            found_ent.start,
                            found_ent.end,
                        )
                    )
        return new_ents
