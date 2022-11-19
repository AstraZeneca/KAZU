from kazu.steps.other.cleanup import CleanupStep
from kazu.data.data import Document, Entity, Section, Mapping, LinkRanks
from kazu.steps.joint_ner_and_linking.explosion import ExplosionStringMatchingStep

from hydra.utils import instantiate

doc_text = "XYZ1 is picked up as entity by explosion step but not mapped to a kb."
"ABC9 is picked up by a different NER step and also not mapped."
"But EFGR was picked up by explosion step and mapped."


def test_configured_mapping_cleanup_discards_ambiguous_mappings(kazu_test_config):
    action = instantiate(kazu_test_config.CleanupActions.MappingFilterCleanupAction)
    doc = Document.create_simple_document(doc_text)
    ents = [
        Entity.load_contiguous_entity(
            start=135,
            end=139,
            match="EFGR",
            entity_class="gene",
            namespace="test",
            mappings={
                Mapping(
                    default_label="EFGR",
                    source="test",
                    parser_name="test",
                    idx="test",
                    confidence=LinkRanks.HIGHLY_LIKELY,
                    mapping_strategy="test",
                    disambiguation_strategy=None,
                ),
                Mapping(
                    default_label="EFGR",
                    source="test",
                    parser_name="test",
                    idx="test",
                    confidence=LinkRanks.AMBIGUOUS,
                    mapping_strategy="test",
                    disambiguation_strategy=None,
                ),
            },
        ),
    ]

    doc.sections[0].entities.extend(ents)
    assert len(doc.get_entities()) == 1
    action.cleanup(doc)
    ent = doc.get_entities()[0]
    assert len(ent.mappings) == 1
    mapping = next(iter(ent.mappings))
    assert mapping.confidence == LinkRanks.HIGHLY_LIKELY


def test_configured_entity_cleanup_discards_unmapped_explosion_ents(kazu_test_config):
    explosion_step_namespace = ExplosionStringMatchingStep.namespace()
    mock_other_ner_namespace = "mock_other_ner_namespace"

    action = instantiate(kazu_test_config.CleanupActions.EntityFilterCleanupAction)
    doc = Document.create_simple_document(doc_text)
    ents = [
        Entity.load_contiguous_entity(
            start=0, end=4, match="XYZ1", entity_class="gene", namespace=explosion_step_namespace
        ),
        Entity.load_contiguous_entity(
            start=69, end=73, match="ABC9", entity_class="gene", namespace=mock_other_ner_namespace
        ),
        Entity.load_contiguous_entity(
            start=135,
            end=139,
            match="EFGR",
            entity_class="gene",
            namespace=explosion_step_namespace,
            mappings={
                Mapping(
                    default_label="EFGR",
                    source="test",
                    parser_name="test",
                    idx="test",
                    confidence=LinkRanks.HIGHLY_LIKELY,
                    mapping_strategy="test",
                    disambiguation_strategy=None,
                )
            },
        ),
    ]

    doc.sections[0].entities.extend(ents)
    assert len(doc.get_entities()) == 3
    action.cleanup(doc)
    assert len(doc.get_entities()) == 2


def test_cleanup_step(kazu_test_config):
    class MockCleanupAction1:
        def cleanup(self, doc: Document):
            doc.sections = [section for section in doc.sections if len(section.text) >= 3]

    class MockCleanupAction2:
        def cleanup(self, doc: Document):
            for ent in doc.get_entities():
                if ent.namespace == "tricky_ent_step":
                    raise Exception(f"{self.__class__} fails on ents from {ent.namespace}!")
                else:
                    ent.match = ent.match.upper()

    cleanup_step = CleanupStep(cleanup_actions=[MockCleanupAction1(), MockCleanupAction2()])
    doc1 = Document(
        idx="test1",
        sections=[
            Section(text="hi", name="doc1_section1"),
            Section(text="2nd section in doc1", name="doc1_section2"),
        ],
    )
    doc2 = Document.create_simple_document(text="cursed document with a gremlin entity")
    doc2.sections[0].entities.append(
        Entity.load_contiguous_entity(
            start=23,
            end=30,
            match="gremlin",
            entity_class="tricky_entity",
            namespace="tricky_ent_step",
        )
    )

    assert len(doc1.sections) == 2
    docs, failed_docs = cleanup_step([doc1, doc2])
    assert len(docs) == 2
    assert len(failed_docs) == 1
    assert len(doc1.sections) == 1
