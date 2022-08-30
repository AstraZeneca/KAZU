import json

import pytest
import requests

from kazu.tests.utils import requires_model_pack
from kazu.web.routes import KAZU
from kazu.web.server import start


@requires_model_pack
@pytest.mark.skip
def test_api(override_kazu_test_config):

    cfg = override_kazu_test_config(
        overrides=["ray=local", "ray.detached=True"],
    )
    start(cfg)
    response = requests.post(
        f"http://127.0.0.1:{cfg.ray.serve.port}/api/{KAZU}/",
        json={"text": "Why do we always test EGFR in these applications?"},
    ).json()
    data = json.loads(response)
    section = data["sections"][0]
    assert len(section["entities"]) > 0
    assert section["entities"][0]["match"] == "EGFR"
