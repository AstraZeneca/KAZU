from kazu.steps.other.cleanup import CleanupStep, CleanupAction
from kazu.data.data import Document, Entity, Section, Mapping, LinkRanks
from kazu.steps.ner.explosion import ExplosionNERStep

from hydra.utils import instantiate


def test_DropUnmappedExplosionEnts_action(kazu_test_config):
    explosion_step_namespace = ExplosionNERStep.namespace()
    mock_other_ner_namespace = "mock_other_ner_namespace"

    action = instantiate(kazu_test_config.CleanupActions.DropUnmappedExplosionEnts)
    doc = Document.create_simple_document(
        "XYZ1 is picked up as entity by explosion step but not mapped to a kb."
        "ABC9 is picked up by a different NER step and also not mapped."
        "But EFGR was picked up by explosion step and mapped."
    )
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
    action(doc)
    assert len(doc.get_entities()) == 2


def test_cleanup_step(kazu_test_config):
    class MockCleanupAction1(CleanupAction):
        def __call__(self, doc: Document):
            doc_sections = set(doc.sections)
            drop_sections = set([section for section in doc_sections if len(section.text) < 3])
            doc_sections.difference_update(drop_sections)
            doc.sections = list(doc_sections)

    class MockCleanupAction2(CleanupAction):
        def __call__(self, doc: Document):
            for ent in doc.get_entities():
                if ent.namespace == "tricky_ent_step":
                    raise Exception(f"{self.__class__} fails on ents from {ent.namespace}!")
                else:
                    ent.match = ent.match.upper()

    cleanup_step = CleanupStep(
        depends_on=[], cleanup_actions=[MockCleanupAction1(), MockCleanupAction2()]
    )
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
