from collections.abc import Callable
from copy import deepcopy
from typing import Optional

from kazu.data import Entity, CharSpan
from kazu.utils.spacy_pipeline import SpacyPipelines
from spacy.tokens import Doc


def _copy_ent_with_new_spans(
    entity: Entity,
    spans: list[tuple[int, int]],
    text: str,
    join_str: str,
    rule_name: Optional[str] = None,
) -> Entity:
    attrb_dict = deepcopy(entity).__dict__
    attrb_dict.pop("spans")
    attrb_dict.pop("match_norm")
    attrb_dict.pop("match")
    attrb_dict.pop("start")
    attrb_dict.pop("end")
    ent = Entity.from_spans(spans=spans, text=text, join_str=join_str, **attrb_dict)
    if rule_name:
        ent.metadata["split_rule"] = rule_name
    return ent


class SplitOnConjunctionPattern:
    def __init__(self, path: str):
        """Analyse.

        :param pattern:
        """
        self.path = path
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_path(path, path)

    def __call__(self, entity: Entity, text: str) -> list[Entity]:
        if any((x in entity.match for x in [" and ", " or ", " nor "])):
            doc = self.spacy_pipelines.process_single(entity.match, model_name=self.path)
            return self.run_conjunction_rules(doc, entity, text)
        else:
            return []

    def run_conjunction_rules(self, doc: Doc, entity: Entity, text: str) -> list[Entity]:
        ents: list[Entity] = []
        noun_chunks = list(doc.noun_chunks)
        if len(noun_chunks) > 0:
            anchor_chunk = noun_chunks[-1]
            anchor = None
            for tok in anchor_chunk:
                if tok.dep_ == "conj":
                    anchor = tok
            if anchor is not None:
                ents.append(
                    _copy_ent_with_new_spans(
                        spans=[
                            (
                                entity.start + anchor_chunk.start_char,
                                entity.start + anchor_chunk.end_char,
                            )
                        ],
                        text=text,
                        entity=entity,
                        join_str=" ",
                        rule_name="spacy noun chunk tags",
                    )
                )
                for chunk in noun_chunks[:-1]:
                    if anchor in chunk.conjuncts:
                        ents.append(
                            _copy_ent_with_new_spans(
                                spans=[
                                    (
                                        entity.start + chunk.start_char,
                                        entity.start
                                        + (chunk.end_char - chunk.start_char)
                                        + chunk.start_char,
                                    ),
                                    (
                                        entity.start + anchor.idx,
                                        entity.start + len(anchor) + anchor.idx,
                                    ),
                                ],
                                text=text,
                                entity=entity,
                                join_str=" ",
                                rule_name="spacy noun chunk tags",
                            )
                        )

        return ents


class SplitOnNumericalListPatternWithPrefix:
    """split a string of numerically incrementing parts:

    .. testsetup::

        from kazu.steps.ner.entity_post_processing import SplitOnNumericalListPatternWithPrefix
        from kazu.data import Entity

    .. testcode::

        splitter = SplitOnNumericalListPatternWithPrefix()
        ent = Entity.load_contiguous_entity(
            start=0, end=8, namespace="test", entity_class="gene", match="BRCA1/2/3"
        )
        print(splitter(ent, "BRCA1/2/3 are oncogenes"))

    .. testoutput::

        [BRCA1:gene:test:0:5, BRCA2:gene:test:0:7, BRCA3:gene:test:0:9]
    """

    def __init__(self, pattern: str = "/"):
        """Pattern to split the string on (typically "/")

        :param pattern:
        """
        self.pattern = pattern

    def __call__(self, entity: Entity, text: str) -> list[Entity]:
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
                            (
                                input_ent_start + span_offset,
                                input_ent_start + span_offset + len(part),
                            ),
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
    """Some simple rules to split non-contiguous entities into component entities."""

    def __init__(self, entity_conditions: dict[str, list[Callable[[Entity, str], list[Entity]]]]):

        self.entity_conditions = entity_conditions

    def __call__(self, entity: Entity, text: str) -> list[Entity]:
        existing_offsets: set[CharSpan] = set()
        existing_offsets.update(entity.spans)

        new_ents = []
        for rule in self.entity_conditions.get(entity.entity_class, []):
            found_ents = rule(entity, text)
            for found_ent in found_ents:
                # only add new ent if offsets have changed
                if any(span not in existing_offsets for span in found_ent.spans):
                    new_ents.append(found_ent)
                    existing_offsets.update(found_ent.spans)
        return new_ents
