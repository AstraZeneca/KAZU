from kazu.pipeline import Pipeline, load_steps
from kazu.tests.utils import requires_model_pack


@requires_model_pack
def test_configs(kazu_test_config):
    # test the default pipeline can load/configs are all correct
    Pipeline(steps=load_steps(kazu_test_config))
