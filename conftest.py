from typing import List

import pytest
from hydra import compose, initialize_config_dir

from kazu.tests.utils import CONFIG_DIR


@pytest.fixture(scope="session")
def override_kazu_test_config():
    def _override_kazu_test_config(overrides: List[str]):
        """Return an optionally overriden copy of the kazu test config.

        :return: DictConfig
        """

        # needs a str, can't take a Path
        with initialize_config_dir(config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=overrides)
        return cfg

    return _override_kazu_test_config


@pytest.fixture(scope="session")
def kazu_test_config(override_kazu_test_config):
    return override_kazu_test_config(overrides=[])
