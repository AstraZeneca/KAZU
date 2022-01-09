import requests

from azner.data.data import Document
from azner.web.routes import AZNER
from azner.web.server import start
from azner.tests.utils import requires_model_pack


@requires_model_pack
def test_api(override_kazu_test_config):

    cfg = override_kazu_test_config(
        overrides=["ray=local", "ray.detached=True"],
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
