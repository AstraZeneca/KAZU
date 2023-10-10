import pytest
from hydra.utils import instantiate

from kazu.tests.utils import requires_model_pack


@pytest.mark.skip(reason="semi deprecated - not in default model pack and is slow")
@requires_model_pack
def test_ExplosionStringMatchingStep(kazu_test_config, ner_simple_test_cases):
    # note, here we just test that the step is functional. Model performance is tested via an acceptance test
    step = instantiate(kazu_test_config.ExplosionStringMatchingStep)
    processed, failures = step(ner_simple_test_cases)
    assert len(processed) == len(ner_simple_test_cases) and len(failures) == 0
