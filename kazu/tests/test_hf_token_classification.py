from hydra.utils import instantiate
from kazu.data.data import SimpleDocument
from kazu.tests.utils import (
    ner_simple_test_cases,
    ner_long_document_test_cases,
    requires_model_pack,
)


@requires_model_pack
def test_TransformersModelForTokenClassificationNerStep(kazu_test_config):
    step = instantiate(kazu_test_config.TransformersModelForTokenClassificationNerStep)
    docs = [SimpleDocument(x[0]) for x in ner_simple_test_cases()]
    classes = [x[1] for x in ner_simple_test_cases()]
    successes, failures = step(docs)
    assert len(successes) == len(docs)
    for doc, target_class in zip(successes, classes):
        assert doc.sections[0].entities[0].entity_class == target_class

    docs = [SimpleDocument(x[0]) for x in ner_long_document_test_cases()]
    target_entity_counts = [x[1] for x in ner_long_document_test_cases()]
    target_classes = [x[2] for x in ner_long_document_test_cases()]
    successes, failures = step(docs)
    assert len(successes) == len(docs)
    for doc, target_entity_count, target_class in zip(
        successes, target_entity_counts, target_classes
    ):
        assert len(doc.sections[0].entities) == target_entity_count
        for ent in doc.sections[0].entities:
            assert ent.entity_class == target_class
