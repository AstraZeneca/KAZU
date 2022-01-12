from typing import List
from kazu.data.data import Entity
from kazu.steps.ner.bio_label_parser import BIOLabelParser

class_labels = ["B-ent1", "B-ent2", "B-ent3", "I-ent1", "I-ent2", "I-ent3", "O"]


def test_bio_label_parser():
    parser = BIOLabelParser(bio_classes=class_labels, namespace="test")

    # simple case
    test_parse_1 = [
        {"label": "O", "offsets": [0, 1], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "B-ent1", "offsets": [2, 6], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [7, 9], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [10, 11], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [12, 16], "text": "A EGFR is a gene", "confidence": 1.0},
    ]
    for state in test_parse_1:
        parser.update_parse_states(**state)

    entities: List[Entity] = parser.get_entities()
    assert len(entities) == 1
    assert entities[0].entity_class == "ent1"
    assert entities[0].match == "EGFR"

    # two token entity
    parser.reset()
    test_parse_2 = [
        {"label": "O", "offsets": [0, 1], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "B-ent1", "offsets": [2, 4], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "I-ent1", "offsets": [4, 6], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [7, 9], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [10, 11], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [12, 16], "text": "A EGFR is a gene", "confidence": 1.0},
    ]
    for state in test_parse_2:
        parser.update_parse_states(**state)

    entities: List[Entity] = parser.get_entities()
    assert len(entities) == 1
    assert entities[0].entity_class == "ent1"
    assert entities[0].match == "EGFR"

    # three entities, one is last in string
    parser.reset()
    test_parse_3 = [
        {"label": "O", "offsets": [0, 1], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "B-ent1", "offsets": [2, 4], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "I-ent1", "offsets": [4, 6], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "B-ent2", "offsets": [7, 9], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [10, 11], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "B-ent3", "offsets": [12, 16], "text": "A EGFR is a gene", "confidence": 1.0},
    ]
    for state in test_parse_3:
        parser.update_parse_states(**state)

    entities: List[Entity] = parser.get_entities()
    assert len(entities) == 3
    assert entities[0].entity_class == "ent1"
    assert entities[1].entity_class == "ent2"
    assert entities[2].entity_class == "ent3"
    assert entities[0].match == "EGFR"
    assert entities[1].match == "is"
    assert entities[2].match == "gene"


def test_bio_label_parser_with_mangled_input():
    parser = BIOLabelParser(bio_classes=class_labels, namespace="test")
    # mangled input
    test_parse_4 = [
        {"label": "O", "offsets": [0, 1], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "I-ent1", "offsets": [2, 4], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "I-ent1", "offsets": [4, 6], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "B-ent1", "offsets": [7, 9], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "O", "offsets": [10, 11], "text": "A EGFR is a gene", "confidence": 1.0},
        {"label": "I-ent1", "offsets": [12, 16], "text": "A EGFR is a gene", "confidence": 1.0},
    ]
    for state in test_parse_4:
        parser.update_parse_states(**state)

    entities: List[Entity] = parser.get_entities()
    assert len(entities) == 1
    assert entities[0].entity_class == "ent1"
    assert entities[0].match == "is"
