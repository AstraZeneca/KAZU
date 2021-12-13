import pytest
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from azner.pipeline.pipeline import Pipeline, load_steps
from azner.tests.utils import CONFIG_DIR


def test_docker_configs():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
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
