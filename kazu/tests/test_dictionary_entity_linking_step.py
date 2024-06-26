from kazu.steps.linking.dictionary import DictionaryEntityLinkingStep, DictionaryIndex
from kazu.data import Entity, Document
from kazu.tests.utils import DummyParser


def test_skips_prelinked_entities(mock_kazu_disk_cache_on_parsers):
    parser = DummyParser(in_path="", entity_class="int")
    parser.populate_databases(force=True)
    index = DictionaryIndex(parser)
    mock_skip_namespace = "mock_skip_namespace"
    mock_noskip_namespace = "mock_noskip_namespace"
    step = DictionaryEntityLinkingStep(
        indices=[index],
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
    assert len(ents[0].linking_candidates) == 0 and len(ents[1].linking_candidates) > 0
