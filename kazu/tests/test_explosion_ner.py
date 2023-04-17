from hydra.utils import instantiate

from kazu.tests.utils import requires_model_pack


@requires_model_pack
def test_ExplosionStringMatchingStep(kazu_test_config, ner_simple_test_cases):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test

    # note: loading the dictionary linking step is necessary as it caches the synonym db (and metadata db)
    # we have a task to address this on GitHub as issue #312
    instantiate(kazu_test_config.DictionaryEntityLinkingStep)
    step = instantiate(kazu_test_config.ExplosionStringMatchingStep)
    processed, failures = step(ner_simple_test_cases)
    assert len(processed) == len(ner_simple_test_cases) and len(failures) == 0
