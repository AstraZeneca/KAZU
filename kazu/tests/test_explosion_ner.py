from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.tests.utils import ner_simple_test_cases, requires_model_pack


@requires_model_pack
def test_ExplosionStringMatchingStep(kazu_test_config):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test

    # note: loading the dictionary linking step is necessary as it caches the synonym db (and metadata db)
    # we have a task to address this on GitHub as issue #312
    instantiate(kazu_test_config.DictionaryEntityLinkingStep)
    step = instantiate(kazu_test_config.ExplosionStringMatchingStep)
    docs = [Document.create_simple_document(x[0]) for x in ner_simple_test_cases()]
    processed, failures = step(docs)
    assert len(processed) == len(docs) and len(failures) == 0
