from kazu.data.data import Entity, Document
from kazu.steps.other.entity_dedup import NerDeDuplicationStep


def test_ner_deduplication():
    step = NerDeDuplicationStep(
        depends_on=[], namespace_preferred_order=["best", "better", "worst"]
    )
    best_ent = Entity.load_contiguous_entity(
        start=5, end=10, namespace="best", entity_class="1", match="n/a"
    )
    better_ent = Entity.load_contiguous_entity(
        start=5, end=10, namespace="better", entity_class="1", match="n/a"
    )
    worst_ent = Entity.load_contiguous_entity(
        start=5, end=10, namespace="worst", entity_class="1", match="n/a"
    )
    doc = Document.create_simple_document("this is a test document")
    doc.sections[0].entities = [best_ent, better_ent, worst_ent]
    successes, _ = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 1
    ent = doc.sections[0].entities[0]
    assert ent.namespace == "best"
    doc = Document.create_simple_document("this is a test document")
    doc.sections[0].entities = [worst_ent, better_ent]
    successes, _ = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 1
    ent = doc.sections[0].entities[0]
    assert ent.namespace == "better"
    best_ent2 = Entity.load_contiguous_entity(
        start=6, end=12, namespace="best", entity_class="1", match="n/a"
    )
    better_ent2 = Entity.load_contiguous_entity(
        start=7, end=13, namespace="better", entity_class="1", match="n/a"
    )
    worst_ent2 = Entity.load_contiguous_entity(
        start=6, end=12, namespace="worst", entity_class="1", match="n/a"
    )
    doc = Document.create_simple_document("this is a test document")
    doc.sections[0].entities = [best_ent, worst_ent, better_ent, worst_ent2, better_ent2, best_ent2]
    successes, _ = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 3
    for ent, expected_ns in zip(doc.sections[0].entities, ["best", "better", "best"]):
        assert ent.namespace == expected_ns
