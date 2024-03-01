import json

import pytest
from kazu.data.data import (
    CharSpan,
    Entity,
    Document,
    SynonymTermWithMetrics,
    Mapping,
    StringMatchConfidence,
    DisambiguationConfidence,
)
from kazu.tests.utils import make_dummy_synonym_term


def test_serialisation():
    original_doc = Document.create_simple_document("Hello")
    term = make_dummy_synonym_term(ids=["1", "2", "3"], parser_name="test", embed_score=0.99)
    original_doc.sections[0].entities = [
        Entity(
            namespace="test",
            match="metastatic liver cancer",
            entity_class="test",
            spans=frozenset([CharSpan(start=16, end=39)]),
            syn_term_to_synonym_terms={term: term},
            mappings={
                Mapping(
                    default_label="a",
                    source="b",
                    parser_name="c",
                    idx="d",
                    string_match_strategy="e",
                    string_match_confidence=StringMatchConfidence.PROBABLE,
                    disambiguation_confidence=DisambiguationConfidence.PROBABLE,
                    disambiguation_strategy="f",
                    xref_source_parser_name="g",
                    metadata={"some": "thing"},
                ),
                Mapping(
                    default_label="aa",
                    source="bb",
                    parser_name="cc",
                    idx="dd",
                    string_match_strategy="ee",
                    string_match_confidence=StringMatchConfidence.PROBABLE,
                    disambiguation_confidence=None,
                    disambiguation_strategy=None,
                    xref_source_parser_name=None,
                    metadata={},
                ),
            },
        )
    ]
    original_doc.sections[0].sentence_spans = [
        CharSpan(start=0, end=28),
        CharSpan(start=29, end=50),
    ]
    json_str = original_doc.to_json()
    # ensure this is valid json
    json_doc = json.loads(json_str)
    assert type(json_doc["sections"][0]["sentence_spans"]) is list

    # since entities are compared by id(self), they cannot be compared directly. We therefore need to compare their dictionary representations
    deserialised_doc = Document.from_json(json_str)
    original_entities = original_doc.sections[0].entities
    deserialised_entities = deserialised_doc.sections[0].entities
    # remove entities, since they will never be equal
    original_doc.sections[0].entities = []
    deserialised_doc.sections[0].entities = []
    assert deserialised_doc == original_doc
    for original_entity, deserialised_entity in zip(original_entities, deserialised_entities):
        assert original_entity.__dict__ == deserialised_entity.__dict__


def test_overlap_logic():
    # e.g. "the patient has metastatic liver cancers"
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=16, end=39)]),
    )
    e2 = Entity(
        namespace="test",
        match="liver cancers",
        entity_class="test",
        spans=frozenset([CharSpan(start=27, end=40)]),
    )

    assert e1.is_partially_overlapped(e2)

    # e.g. 'liver and lung cancer'
    e1 = Entity(
        namespace="test",
        match="liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=0, end=4), CharSpan(start=15, end=21)]),
    )
    e2 = Entity(
        namespace="test",
        match="lung cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=9, end=21)]),
    )
    assert not e1.is_partially_overlapped(e2)


def test_syn_term_manipulation():
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=16, end=39)]),
    )

    # first test syn_terms are merged correctly (same id set, same parser name)
    syn_term_1 = make_dummy_synonym_term(["1", "2", "3"], parser_name="test", search_score=99.5)
    e1.update_terms([syn_term_1])
    syn_term_2 = make_dummy_synonym_term(["1", "2", "3"], parser_name="test", embed_score=99.6)
    e1.update_terms([syn_term_2])
    assert len(e1.syn_term_to_synonym_terms) == 1
    merged_syn_term: SynonymTermWithMetrics = next(iter(e1.syn_term_to_synonym_terms.values()))
    assert merged_syn_term.embed_score == 99.6
    assert merged_syn_term.search_score == 99.5

    # now test syn_terms are differentiated if parser name is different
    syn_term_3 = make_dummy_synonym_term(["1", "2", "3"], parser_name="test_2", search_score=99.5)
    e1.update_terms([syn_term_3])
    assert len(e1.syn_term_to_synonym_terms) == 2

    # now test syn_terms are differentiated if id set is different
    syn_term_4 = make_dummy_synonym_term(["1", "2"], parser_name="test", search_score=99.5)
    e1.update_terms([syn_term_4])
    assert len(e1.syn_term_to_synonym_terms) == 3


def test_section_sentence_spans_is_immutable():
    x = Document.create_simple_document("Hello")
    x.sections[0].sentence_spans = [CharSpan(start=0, end=28), CharSpan(start=29, end=50)]

    # try re-assigning sentence_spans, which should raise an error
    with pytest.raises(AttributeError):
        x.sections[0].sentence_spans = [CharSpan(start=0, end=28), CharSpan(start=29, end=50)]
