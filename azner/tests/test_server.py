import os

import pytest
import requests
from hydra import compose, initialize_config_dir

from azner.data.data import Document
from azner.tests.utils import (
    SKIP_MESSAGE,
)
from azner.web.routes import AZNER
from azner.web.server import start


@pytest.mark.skipif(os.environ.get("KAZU_TEST_CONFIG_DIR") is None, reason=SKIP_MESSAGE)
def test_api():

    with initialize_config_dir(config_dir=os.environ.get("KAZU_TEST_CONFIG_DIR")):
        cfg = compose(
            config_name="config",
            overrides=[
                "ray=local",
            ],
        )
        start(cfg)
        response = requests.post(
            f"http://127.0.0.1:{cfg.ray.serve.port}/api/{AZNER}/",
            json={"text": "Why do we always test EGFR in " "these applications?"},
        ).json()
        result = Document(**response)
        assert isinstance(result, Document)
        section = result.sections[0]
        assert len(section.entities) > 0
        assert section.entities[0].match == "EGFR"
