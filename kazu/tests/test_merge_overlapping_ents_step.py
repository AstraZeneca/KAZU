import pytest
from omegaconf import DictConfig
from hydra.utils import instantiate

from kazu.data.data import Entity, CharSpan, Document, Mapping, StringMatchConfidence
from kazu.steps import (
    MergeOverlappingEntsStep,
    ExplosionStringMatchingStep,
    TransformersModelForTokenClassificationNerStep,
)


@pytest.fixture
def merge_step(kazu_test_config: DictConfig) -> MergeOverlappingEntsStep:
    merge_step: MergeOverlappingEntsStep = instantiate(kazu_test_config.MergeOverlappingEntsStep)
    return merge_step


EXPLOSION_NAMESPACE = ExplosionStringMatchingStep.namespace()
TRANSFORMERS_NAMESPACE = TransformersModelForTokenClassificationNerStep.namespace()


def test_merge_overlapping_step_case_1(merge_step):
    # should filter longer span with no mappings
    explosion_ent = Entity(
        namespace=EXPLOSION_NAMESPACE,
        match="Baclofen",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings=set(),
    )

    doc = Document.create_simple_document("Baclofen drug")
    doc.sections[0].entities = [explosion_ent, transformer_ent]
    processed, failures = merge_step([doc])
    assert len(processed) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == explosion_ent


def test_merge_overlapping_step_case_2(merge_step):
    # should filter shorter span, as longer span has a mapping
    explosion_ent = Entity(
        namespace=EXPLOSION_NAMESPACE,
        match="Baclofen",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings={
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    doc = Document.create_simple_document("Baclofen drug")
    doc.sections[0].entities = [explosion_ent, transformer_ent]
    processed, failures = merge_step([doc])
    assert len(processed) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == transformer_ent


def test_merge_overlapping_step_case_3(merge_step):
    # two spans the same length. One should be kept as it's a preferred class (according to config)
    explosion_ent = Entity(
        namespace=EXPLOSION_NAMESPACE,
        match="Baclofen",
        entity_class="anatomy",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings={
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="Baclofen",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_strategy="test",
                disambiguation_strategy=None,
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
            )
        },
    )

    doc = Document.create_simple_document("Baclofen drug")
    doc.sections[0].entities = [explosion_ent, transformer_ent]
    processed, failures = merge_step([doc])
    assert len(processed) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == transformer_ent


def test_merge_overlapping_step_case_4(merge_step):
    # multiple overlapping non contained spans. Longest should be kept
    explosion_ent = Entity(
        namespace=EXPLOSION_NAMESPACE,
        match="Baclofen",
        entity_class="anatomy",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings={
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings={
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )
    transformer_ent_2 = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="drug treatment",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=8, end=22)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    doc = Document.create_simple_document("Baclofen drug treatment")
    doc.sections[0].entities = [explosion_ent, transformer_ent, transformer_ent_2]
    processed, failures = merge_step([doc])
    assert len(processed) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == transformer_ent_2


def test_merge_overlapping_step_case_5(merge_step):
    # a more complex case involving multiple locations
    explosion_ent = Entity(
        namespace=EXPLOSION_NAMESPACE,
        match="Baclofen",
        entity_class="anatomy",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings={
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings={
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )
    transformer_ent_2 = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="drug treatment",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=8, end=22)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent_3 = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="inpatients-",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=23, end=34)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    transformer_ent_4 = Entity(
        namespace=TRANSFORMERS_NAMESPACE,
        match="assistance",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=34, end=44)]),
        mappings={
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                string_match_strategy="test",
                disambiguation_strategy=None,
            )
        },
    )

    doc = Document.create_simple_document("Baclofen drug treatment inpatients-assistance")
    doc.sections[0].entities = [
        explosion_ent,
        transformer_ent,
        transformer_ent_2,
        transformer_ent_3,
        transformer_ent_4,
    ]
    processed, failures = merge_step([doc])
    assert len(processed) == 1
    assert len(doc.sections[0].entities) == 3
    assert doc.sections[0].entities[0] == transformer_ent_2
    assert doc.sections[0].entities[1] == transformer_ent_3
    assert doc.sections[0].entities[2] == transformer_ent_4
