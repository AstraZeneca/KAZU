from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.tests.utils import ner_simple_test_cases, requires_model_pack


@requires_model_pack
def test_TransformersModelForTokenClassificationNerStep(kazu_test_config):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test
    step = instantiate(kazu_test_config.TransformersModelForTokenClassificationNerStep)
    docs = [Document.create_simple_document(x[0]) for x in ner_simple_test_cases()]
    processed, failures = step(docs)
    assert len(processed) == len(docs)


@requires_model_pack
def test_multilabel_transformer_token_classification(override_kazu_test_config):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test
    cfg = override_kazu_test_config(
        overrides=["TransformersModelForTokenClassificationNerStep=multilabel"],
    )
    step = instantiate(cfg.TransformersModelForTokenClassificationNerStep)
    docs = [Document.create_simple_document(x[0]) for x in ner_simple_test_cases()]
    processed, failures = step(docs)
    assert len(processed) == len(docs)
