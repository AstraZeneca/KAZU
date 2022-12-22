from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.tests.utils import ner_simple_test_cases, requires_model_pack


@requires_model_pack
def test_spacy_ner_step(kazu_test_config):
    # note, here we just test that the step is functional. Model performance should be tested via an acceptance test
    step = instantiate(kazu_test_config.SpacyNerStep)
    docs = [Document.create_simple_document(x[0]) for x in ner_simple_test_cases()]
    processed, failures = step(docs)
    assert len(processed) == len(docs)
