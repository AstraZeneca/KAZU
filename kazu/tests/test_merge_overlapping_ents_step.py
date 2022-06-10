from hydra.utils import instantiate
from kazu.data.data import Entity, CharSpan, Document, Mapping, LinkRanks
from kazu.tests.utils import requires_model_pack


@requires_model_pack
def test_merge_overlapping_step_case_1(kazu_test_config):
    # should filter longer span with no mappings
    step = instantiate(kazu_test_config.MergeOverlappingEntsStep)
    explosion_ent = Entity(
        namespace="ExplosionNERStep",
        match="Baclofen",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings=[],
    )

    doc = Document.create_simple_document("Baclofen drug")
    doc.sections[0].entities = [explosion_ent, transformer_ent]
    successes, failures = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == explosion_ent


@requires_model_pack
def test_merge_overlapping_step_case_2(kazu_test_config):
    # should filter shorter span, as longer span has a mapping
    step = instantiate(kazu_test_config.MergeOverlappingEntsStep)
    explosion_ent = Entity(
        namespace="ExplosionNERStep",
        match="Baclofen",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings=[
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    doc = Document.create_simple_document("Baclofen drug")
    doc.sections[0].entities = [explosion_ent, transformer_ent]
    successes, failures = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == transformer_ent


@requires_model_pack
def test_merge_overlapping_step_case_3(kazu_test_config):
    # two spans the same length. One should be kept as it's a preferred class (according to config)
    step = instantiate(kazu_test_config.MergeOverlappingEntsStep)
    explosion_ent = Entity(
        namespace="ExplosionNERStep",
        match="Baclofen",
        entity_class="anatomy",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings=[
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="Baclofen",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    doc = Document.create_simple_document("Baclofen drug")
    doc.sections[0].entities = [explosion_ent, transformer_ent]
    successes, failures = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == transformer_ent


@requires_model_pack
def test_merge_overlapping_step_case_4(kazu_test_config):
    # multiple overlapping non contained spans. Longest should be kept
    step = instantiate(kazu_test_config.MergeOverlappingEntsStep)
    explosion_ent = Entity(
        namespace="ExplosionNERStep",
        match="Baclofen",
        entity_class="anatomy",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings=[
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings=[
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )
    transformer_ent_2 = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="drug treatment",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=8, end=22)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    doc = Document.create_simple_document("Baclofen drug treatment")
    doc.sections[0].entities = [explosion_ent, transformer_ent, transformer_ent_2]
    successes, failures = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 1
    assert doc.sections[0].entities[0] == transformer_ent_2


@requires_model_pack
def test_merge_overlapping_step_case_5(kazu_test_config):
    # a more complex case involving multiple locations
    step = instantiate(kazu_test_config.MergeOverlappingEntsStep)
    explosion_ent = Entity(
        namespace="ExplosionNERStep",
        match="Baclofen",
        entity_class="anatomy",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=0, end=8)]),
        mappings=[
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="Baclofen drug",
        entity_class="drug",
        spans=frozenset([CharSpan(start=0, end=13)]),
        mappings=[
            Mapping(
                default_label="ignore me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )
    transformer_ent_2 = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="drug treatment",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=8, end=22)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent_3 = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="inpatients-",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=23, end=34)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    transformer_ent_4 = Entity(
        namespace="TransformersModelForTokenClassificationNerStep",
        match="assistance",
        entity_class="disease",  # <- deliberately wrong
        spans=frozenset([CharSpan(start=34, end=44)]),
        mappings=[
            Mapping(
                default_label="pick me!",
                source="test",
                parser_name="test",
                idx="test",
                confidence=LinkRanks.HIGH_CONFIDENCE,
            )
        ],
    )

    doc = Document.create_simple_document("Baclofen drug treatment inpatients-assistance")
    doc.sections[0].entities = [
        explosion_ent,
        transformer_ent,
        transformer_ent_2,
        transformer_ent_3,
        transformer_ent_4,
    ]
    successes, failures = step([doc])
    assert len(successes) == 1
    assert len(doc.sections[0].entities) == 3
    assert doc.sections[0].entities[0] == transformer_ent_2
    assert doc.sections[0].entities[1] == transformer_ent_3
    assert doc.sections[0].entities[2] == transformer_ent_4
