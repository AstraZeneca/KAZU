import json
from typing import Dict, List

from kazu.data.data import (
    CharSpan,
    Entity,
    Document,
    Hit,
    EquivalentIdSet,
    EquivalentIdAggregationStrategy,
)


def make_hit(ids: List[str], parser_name: str, metrics: Dict[str, float]) -> Hit:
    id_set = EquivalentIdSet(
        aggregated_by=EquivalentIdAggregationStrategy.UNAMBIGUOUS, ids=frozenset(ids)
    )
    hit = Hit(id_set=id_set, parser_name=parser_name, per_normalized_syn_metrics={"test": metrics})
    return hit


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
    metrics_1 = {"test_metric_1": 99.5}
    hit_1 = make_hit(["1", "2", "3"], parser_name="test", metrics=metrics_1)
    e1.update_hits([hit_1])
    metrics_2 = {"test_metric_2": 99.6}
    hit_2 = make_hit(["1", "2", "3"], parser_name="test", metrics=metrics_2)
    e1.update_hits([hit_2])
    assert len(e1.hits) == 1
    merged_hit: Hit = next(iter(e1.hits))
    # make_hits makes "test" the norm_syn that metrics are grouped by
    test_metric_items = merged_hit.per_normalized_syn_metrics["test"].items()
    assert metrics_1.items() <= test_metric_items
    assert metrics_2.items() <= test_metric_items

    # now test hits are differentiated if parser name is different
    metrics_3 = {"test_metric_1": 99.5}
    hit_3 = make_hit(["1", "2", "3"], parser_name="test_2", metrics=metrics_3)
    e1.update_hits([hit_3])
    assert len(e1.hits) == 2

    # now test hits are differentiated if id set is different
    metrics_4 = {"test_metric_1": 99.5}
    hit_4 = make_hit(["1", "2"], parser_name="test", metrics=metrics_4)
    e1.update_hits([hit_4])
    assert len(e1.hits) == 3


def test_entity_hit_groups():
    # first group: multiple entities, same hits/matched string and class
    e1_group1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=0, end=6)]),
    )
    e1_group1.update_hits(
        [make_hit(["1", "2", "3"], parser_name="test", metrics={"test_metric_1": 99.5})]
    )
    e1_group1.update_hits(
        [make_hit(["1", "2", "3"], parser_name="test", metrics={"test_metric_2": 99.6})]
    )

    e2_group1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=12, end=14)]),
    )
    e2_group1.update_hits(
        [make_hit(["1", "2", "3"], parser_name="test", metrics={"test_metric_1": 99.5})]
    )
    e2_group1.update_hits(
        [make_hit(["1", "2", "3"], parser_name="test", metrics={"test_metric_2": 99.6})]
    )

    # second group, different matched string and class
    e1_group2 = Entity(
        namespace="test",
        match="skin disease",
        entity_class="test_2",
        spans=frozenset([CharSpan(start=0, end=6)]),
    )
    e1_group2.update_hits(
        [make_hit(["4", "5", "6"], parser_name="test_2", metrics={"test_metric_1": 99.5})]
    )
    e1_group2.update_hits(
        [make_hit(["7", "8", "9"], parser_name="test_2", metrics={"test_metric_2": 99.6})]
    )
    # third group, same as group 2, except different hit set
    e1_group3 = Entity(
        namespace="test",
        match="skin disease",
        entity_class="test_2",
        spans=frozenset([CharSpan(start=0, end=6)]),
    )
    e1_group3.update_hits(
        [make_hit(["4", "5", "6"], parser_name="test_2", metrics={"test_metric_1": 99.5})]
    )

    doc = Document.create_simple_document("Hello")
    doc.sections[0].entities = [e1_group1, e2_group1, e1_group2, e1_group3]

    groups = list(doc.sections[0].group_entities_on_hits)
    assert len(groups) == 3


def make_entity_with_hits(
    start: int,
    end: int,
):
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=start, end=end)]),
    )
    # first test hits are merged correctly (same id set, same parser name)
    metrics_1 = {"test_metric_1": 99.5}
    hit_1 = make_hit(["1", "2", "3"], parser_name="test", metrics=metrics_1)
    e1.update_hits([hit_1])
    metrics_2 = {"test_metric_2": 99.6}
    hit_2 = make_hit(["1", "2", "3"], parser_name="test", metrics=metrics_2)
    e1.update_hits([hit_2])
    return e1
