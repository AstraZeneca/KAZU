import logging
import statistics
from typing import Optional, Dict, Tuple, List

import pydash

from azner.data.data import Entity, EntityMetadata

logger = logging.getLogger(__name__)


class BIOLabelParser:
    """
    A parser that turns a BIO style labelling schema into instances of Entity
    """

    entity_start_symbol = "B"
    entity_inside_symbol = "I"
    entity_outside_symbol = "O"

    class EntityParseState:
        def __init__(self, entity_class: str, namespace: str):
            """
            this class tracks the BIO status of a given entity class as all labels are processed, determining
            if an entity has been completed or not. An instance of this is created for each entity class described in
            the constructor to the outer class
            :param entity_class: the entity class to tac
            :param namespace: the namespace of the calling step
            """
            self.namespace = namespace
            self.entity_class = entity_class
            self.start = None
            self.end = None
            self.inside = False
            self.token_confidences = []
            self.entities_found = []

        def clear_entities_found_list(self):
            """
            reset the entities_found list.
            :return:
            """
            self.entities_found = []

        def reset_state(self):
            """
            reset the status to it's initial state
            :return:
            """
            self.start = None
            self.end = None
            self.inside = False
            self.token_confidences = []

        def get_confidence_info(self) -> Dict[str, float]:
            """
            generate a dict of confidence information. The output differs according to whether multiple tokens (and
            therefore confidence scores comprise the entity)
            :return:
            """
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
            """
            called when an entity is completed - i.e. no more tokens are expected to make it up
            :param text: the text string the entity was derived from
            :return:
            """
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
            """
            update the parse state with new information. The logic is described below
            :param bio_symbol: bio symbol to update with
            :param entity_class: class of entity, if any
            :param offsets: offsets of the bio symbol
            :param text: the complete text string from which the update is derived
            :param confidence: any confidence score associated with the symbol
            :return:
            """

            if (
                entity_class == self.entity_class
                or bio_symbol == BIOLabelParser.entity_outside_symbol
            ):  # do nothing if entity class not related to this instance, or is not an outside symbol
                # if inside an instance of an entity, and a new one is starting, complete current entity and start a new
                # one
                if bio_symbol == BIOLabelParser.entity_start_symbol and self.inside:
                    self.complete_entity(text)
                    self.inside = True
                    self.start = offsets[0]
                    self.end = offsets[1]
                    if isinstance(confidence, float):
                        self.token_confidences.append(confidence)
                # if entity is starting and not currently inside, change state to inside
                elif bio_symbol == BIOLabelParser.entity_start_symbol:
                    self.inside = True
                    self.start = offsets[0]
                    self.end = offsets[1]
                    if isinstance(confidence, float):
                        self.token_confidences.append(confidence)
                    self.token_confidences.append(confidence)
                # if currently inside and next BIO symbol is still inside, update state accordingly
                elif bio_symbol == BIOLabelParser.entity_inside_symbol:
                    self.end = offsets[1]
                    if isinstance(confidence, float):
                        self.token_confidences.append(confidence)
                    self.token_confidences.append(confidence)
                # if currently inside and next BIO symbol is outside, complete the entity
                elif bio_symbol == BIOLabelParser.entity_outside_symbol and self.inside:
                    self.complete_entity(text)

    def __init__(self, bio_classes: List[str], namespace: str):
        """

        :param bio_classes: a list of the BIO class labels, in the format B-<entity class>
        :param namespace: the namespace of the calling Step
        """
        self.namespace = namespace
        self.bio_classes = bio_classes
        self.entity_classes = sorted(
            list({x.split("-")[1] for x in bio_classes if x != self.entity_outside_symbol})
        )
        self.entity_state_parsers = [
            BIOLabelParser.EntityParseState(entity_class, namespace)
            for entity_class in self.entity_classes
        ]
        self.active_text = (
            None  # the last text string passed to update_parse_states. Required by finalise
        )

    def update_parse_states(
        self, label: str, offsets: Tuple[int, int], text: str, confidence: Optional[float]
    ):
        """
        the main method of this class. Updates each instance of EntityParseState with new information.

        :param label: the BIO label to process
        :param offsets: the offsets of this label
        :param text: the text from with the above were derived
        :param confidence: if available, any confidence score associated with this label
        :return:
        """

        if label == BIOLabelParser.entity_outside_symbol:
            for entity_parse_state in self.entity_state_parsers:
                entity_parse_state.update(label, None, offsets, text, confidence=None)
        else:
            bio_symbol, entity_class = label.split("-")
            for entity_parse_state in self.entity_state_parsers:
                entity_parse_state.update(
                    bio_symbol, entity_class, offsets, text, confidence=confidence
                )
        self.active_text = text

    def finalise(self, text: str):
        """
        call complete_entity on all EntityParseStates
        :return:
        """
        [x.complete_entity(text) for x in self.entity_state_parsers]

    def get_entities(self):
        """
        return all entities that have been found so far
        :return:
        """
        self.finalise(self.active_text)
        return pydash.flatten([x.entities_found for x in self.entity_state_parsers])

    def reset(self):
        """
        reset the entity found list for each EntityParseState. I.e. - should be called before processing a new string of
        text
        :return:
        """
        for entity_parse_state in self.entity_state_parsers:
            entity_parse_state.clear_entities_found_list()
        self.active_text = None
