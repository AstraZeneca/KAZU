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
    data = requests.post(
        f"http://127.0.0.1:{cfg.ray.serve.port}/api/{KAZU}/",
        json={"text": "Why do we always test EGFR in these applications?"},
    ).json()
    section = data["sections"][0]
    assert len(section["entities"]) > 0
    assert section["entities"][0]["match"] == "EGFR"


@requires_model_pack
@pytest.mark.skip
def test_batch_api(override_kazu_test_config):

    cfg = override_kazu_test_config(
        overrides=["ray=local", "ray.detached=True"],
    )
    start(cfg)
    data = requests.post(
        f"http://127.0.0.1:{cfg.ray.serve.port}/api/{KAZU}/batch",
        json=[{"text": "Why do we always test EGFR in these applications?"},
              {"sections": ["hello this is the first section", "the second section mentions BRCA1"]}],
    ).json()
    doc0_section0 = data[0]["sections"][0]
    assert len(doc0_section0["entities"]) > 0
    assert doc0_section0["entities"][0]["match"] == "EGFR"

    doc1_section1 = data[1]["sections"][1]
    assert doc1_section1["entities"][0]["match"] == "BRCA1"
