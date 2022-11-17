import json
from copy import deepcopy

import pytest

from kazu.data.data import CharSpan, Entity, Document, SynonymTermWithMetrics, DocumentJsonUtils
from kazu.tests.utils import make_dummy_synonym_term


def test_serialisation():
    x = Document.create_simple_document("Hello")
    x.sections[0].offset_map = {CharSpan(start=1, end=2): CharSpan(start=1, end=2)}
    x.sections[0].entities = [
        Entity(
            namespace="test",
            match="metastatic liver cancer",
            entity_class="test",
            spans=frozenset([CharSpan(start=16, end=39)]),
        )
    ]
    x.sections[0].sentence_spans = [CharSpan(start=0, end=28), CharSpan(start=29, end=50)]
    json_str = x.json()
    # ensure this is valid json
    json_doc = json.loads(json_str)
    assert type(json_doc["sections"][0]["sentence_spans"]) == list


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


def test_json_utils():
    x = Document.create_simple_document("Hello")
    x.sections[0].offset_map = {CharSpan(start=1, end=2): CharSpan(start=1, end=2)}
    x.sections[0].entities = [
        Entity(
            namespace="test",
            match="metastatic liver cancer",
            entity_class="test",
            spans=frozenset([CharSpan(start=16, end=39)]),
        )
    ]
    x.sections[0].sentence_spans = [CharSpan(start=0, end=28), CharSpan(start=29, end=50)]
    y = deepcopy(x)
    json_dict = DocumentJsonUtils.doc_to_json_dict(y)
    old_json_dict = json.loads(x.json())
    assert json_dict == old_json_dict

