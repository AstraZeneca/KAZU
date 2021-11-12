import os
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from azner.data.data import SimpleDocument
from azner.tests.utils import (
    ner_simple_test_cases,
    ner_long_document_test_cases,
)

skip_msg = (
    "skipping acceptance test as KAZU_TEST_CONFIG_DIR is not provided as an environment variable. This "
    "should be the path to a hydra config directory, configured with paths to the various resources/models to "
    "run the production pipeline"
)


@pytest.mark.skipif(os.environ.get("KAZU_TEST_CONFIG_DIR") is None, reason=skip_msg)
def test_TransformersModelForTokenClassificationNerStep():

    with initialize_config_dir(config_dir=os.environ.get("KAZU_TEST_CONFIG_DIR")):
        cfg = compose(
            config_name="config",
        )

        step = instantiate(cfg.TransformersModelForTokenClassificationNerStep)
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
