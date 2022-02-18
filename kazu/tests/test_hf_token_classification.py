from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.tests.utils import (
    ner_simple_test_cases,
    requires_model_pack,
)


@requires_model_pack
def test_TransformersModelForTokenClassificationNerStep(kazu_test_config):
    # note, here we just test that the step is fucnctional. Model performance should be handled via an acceptance test
    step = instantiate(kazu_test_config.TransformersModelForTokenClassificationNerStep)
    docs = [Document.create_simple_document(x[0]) for x in ner_simple_test_cases()]
    successes, failures = step(docs)
    assert len(successes) == len(docs)
