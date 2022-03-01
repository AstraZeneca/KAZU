import json

from kazu.data.data import CharSpan, Entity, Document


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
