import pytest
from hydra.utils import instantiate

from kazu.pipeline import Pipeline, load_steps


def test_docker_configs(override_kazu_test_config):
    cfg = override_kazu_test_config(
        overrides=[
            "DictionaryEntityLinkingStep=docker",
            "SapBertForEntityLinkingStep=docker",
            "TransformersModelForTokenClassificationNerStep=docker",
        ],
    )
    # should raise OSErrors from missing files
    with pytest.raises(OSError):
        instantiate(cfg.DictionaryEntityLinkingStep)
    with pytest.raises(OSError):
        instantiate(cfg.SapBertForEntityLinkingStep)
    with pytest.raises(OSError):
        instantiate(cfg.TransformersModelForTokenClassificationNerStep)
    with pytest.raises(OSError):
        Pipeline(steps=load_steps(cfg))
