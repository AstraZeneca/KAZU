import logging
import statistics
from typing import Optional, Dict, Tuple

import pydash

from azner.data.data import Entity, EntityMetadata

logger = logging.getLogger(__name__)


class BIOLabelParser:
    entity_start_symbol = "B"
    entity_inside_symbol = "I"
    entity_outside_symbol = "O"

    class EntityParseState:
        def __init__(self, entity_class: str, namespace: str):
            self.namespace = namespace
            self.entity_class = entity_class
            self.start = None
            self.end = None
            self.inside = False
            self.token_confidences = []
            self.entities_found = []

        def clear_entities_found_list(self):
            self.entities_found = []

        def reset_state(self):
            self.start = None
            self.end = None
            self.inside = False
            self.token_confidences = []

        def get_confidence_info(self):
            number_of_tokens = len(self.token_confidences)
            if number_of_tokens > 1:
                confidence_info = {
                    "min_conf": min(self.token_confidences),
                    "max_conf": max(self.token_confidences),
                    "average_conf": statistics.mean(self.token_confidences),
                    "tokens_in_entity": len(self.token_confidences),
                }
            elif number_of_tokens == 1:
                confidence_info = {
                    "average_conf": statistics.mean(self.token_confidences),
                    "tokens_in_entity": 1,
                }
            else:
                confidence_info = {"confidence_score_not_available": True}
            return confidence_info

        def complete_entity(self, text: str):
            if any([not self.inside, self.start is None, self.end is None]):
                logger.warning(
                    f"Tried to complete {self.entity_class} but not properly formed. start: {self.start}, end: {self.end}, inside: {self.inside}, text: {text}"
                )
            else:
                self.entities_found.append(
                    Entity(
                        start=self.start,
                        end=self.end,
                        match=text[self.start : self.end],
                        namespace=self.namespace,
                        entity_class=self.entity_class,
                        hit_metadata=EntityMetadata(entity_meta=self.get_confidence_info()),
                    )
                )
            self.reset_state()

        def update(
            self,
            bio_symbol: str,
            entity_class: Optional[str],
            offsets: Tuple[int, int],
            text: str,
            confidence: Optional[float],
        ):

            if (
                entity_class == self.entity_class
                or bio_symbol == BIOLabelParser.entity_outside_symbol
            ):
                if bio_symbol == BIOLabelParser.entity_start_symbol and self.inside:
                    self.complete_entity(text)
                    self.inside = True
                    self.start = offsets[0]
                    self.end = offsets[1]
                    if isinstance(confidence, float):
                        self.token_confidences.append(confidence)

                elif bio_symbol == BIOLabelParser.entity_start_symbol:
                    self.inside = True
                    self.start = offsets[0]
                    self.end = offsets[1]
                    if isinstance(confidence, float):
                        self.token_confidences.append(confidence)
                    self.token_confidences.append(confidence)

                elif bio_symbol == BIOLabelParser.entity_inside_symbol:
                    self.end = offsets[1]
                    if isinstance(confidence, float):
                        self.token_confidences.append(confidence)
                    self.token_confidences.append(confidence)

                elif bio_symbol == BIOLabelParser.entity_outside_symbol and self.inside:
                    self.complete_entity(text)

    def __init__(self, id_to_label: Dict[int, str], namespace: str):
        self.namespace = namespace
        self.id_to_label = id_to_label
        self.entity_classes = {
            x.split("-")[1] for x in id_to_label.values() if x != self.entity_outside_symbol
        }
        self.entity_state_parsers = [
            BIOLabelParser.EntityParseState(entity_class, namespace)
            for entity_class in self.entity_classes
        ]

    def update_parse_states(
        self, label: str, offsets: Tuple[int, int], text: str, confidence: Optional[float]
    ):
        if label == BIOLabelParser.entity_outside_symbol:
            for entity_parse_state in self.entity_state_parsers:
                entity_parse_state.update(label, None, offsets, text, confidence=None)
        else:
            bio_symbol, entity_class = label.split("-")
            for entity_parse_state in self.entity_state_parsers:
                entity_parse_state.update(
                    bio_symbol, entity_class, offsets, text, confidence=confidence
                )

    def get_entities(self):
        return pydash.flatten([x.entities_found for x in self.entity_state_parsers])

    def reset(self):
        for entity_parse_state in self.entity_state_parsers:
            entity_parse_state.clear_entities_found_list()
