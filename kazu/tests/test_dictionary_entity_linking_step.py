from kazu.steps.linking.dictionary import DictionaryEntityLinkingStep, DictionaryIndex
from kazu.data.data import Entity, Document
from kazu.tests.utils import DummyParser


def test_skips_prelinked_entities(tmp_path):
    parser = DummyParser(str(tmp_path / "dummy_parser"))
    index = DictionaryIndex(parser)
    mock_skip_namespace = "mock_skip_namespace"
    mock_noskip_namespace = "mock_noskip_namespace"
    step = DictionaryEntityLinkingStep(
        depends_on=[],
        indices=[index],
        entity_class_to_ontology_mappings={"int": [parser.name]},
        skip_ner_namespaces={mock_skip_namespace},
    )

    text = 'The word "one", recognised in a joint ner-linking step. But also the word "one" somehow only recognised in a different, purely ner, step.'
    ents = [
        Entity.load_contiguous_entity(
            start=10, end=13, match="one", entity_class="int", namespace=mock_skip_namespace
        ),
        Entity.load_contiguous_entity(
            start=75, end=78, match="one", entity_class="int", namespace=mock_noskip_namespace
        ),
    ]
    doc = Document.create_simple_document(text)
    doc.sections[0].entities.extend(ents)

    step([doc])
    assert (
        len(ents[0].syn_term_to_synonym_terms) == 0
        and len(ents[1].syn_term_to_synonym_terms) > 0
    )
