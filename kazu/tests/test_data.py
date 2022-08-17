import json

from kazu.data.data import CharSpan, Entity, Document, SynonymTermWithMetrics
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
    json_str = x.json()
    # ensure this is valid json
    json.loads(json_str)


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


def test_hit_manipulation():
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=16, end=39)]),
    )

    # first test hits are merged correctly (same id set, same parser name)
    hit_1 = make_dummy_synonym_term(["1", "2", "3"], parser_name="test", search_score=99.5)
    e1.update_terms([hit_1])
    hit_2 = make_dummy_synonym_term(["1", "2", "3"], parser_name="test", embed_score=99.6)
    e1.update_terms([hit_2])
    assert len(e1.syn_term_to_synonym_terms) == 1
    merged_hit: SynonymTermWithMetrics = next(iter(e1.syn_term_to_synonym_terms.values()))
    # make_hits makes "test" the norm_syn that metrics are grouped by
    assert merged_hit.embed_score == 99.6
    assert merged_hit.search_score == 99.5

    # now test hits are differentiated if parser name is different
    hit_3 = make_dummy_synonym_term(["1", "2", "3"], parser_name="test_2", search_score=99.5)
    e1.update_terms([hit_3])
    assert len(e1.syn_term_to_synonym_terms) == 2

    # now test hits are differentiated if id set is different
    hit_4 = make_dummy_synonym_term(["1", "2"], parser_name="test", search_score=99.5)
    e1.update_terms([hit_4])
    assert len(e1.syn_term_to_synonym_terms) == 3
