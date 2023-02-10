import pytest
import requests

from kazu.tests.utils import requires_model_pack
from kazu.web.routes import KAZU


@requires_model_pack
def test_api(ray_server, kazu_test_config):
    data = requests.post(
        f"http://127.0.0.1:{kazu_test_config.ray.serve.http_options.port}/api/{KAZU}/",
        json={"text": "EGFR is an important gene in breast cancer"},
    ).json()
    section = data["sections"][0]
    assert len(section["entities"]) > 0
    assert section["entities"][0]["match"] == "EGFR"


@requires_model_pack
@pytest.mark.parametrize(
    (
        "server_type",
        "should_fail_auth",
    ),
    [
        pytest.param("ray_server", False),
        pytest.param("ray_server_with_jwt_auth", False),
        pytest.param("ray_server_with_jwt_auth", True),
    ],
)
def test_batch_api(server_type, should_fail_auth, request, kazu_test_config):
    headers = request.getfixturevalue(server_type)
    if should_fail_auth:
        headers = {}

    response = requests.post(
        f"http://127.0.0.1:{kazu_test_config.ray.serve.http_options.port}/api/{KAZU}/batch",
        headers=headers,
        json=[
            {"text": "EGFR is an important gene in breast cancer"},
            {
                "sections": {
                    "sec1": "hello this is the first section",
                    "sec2": "the second section mentions BRCA1",
                }
            },
        ],
    )
    if should_fail_auth:
        assert response.status_code == 401
    else:
        data = response.json()
        doc0_section0 = data[0]["sections"][0]
        assert len(doc0_section0["entities"]) > 0
        assert doc0_section0["entities"][0]["match"] == "EGFR"

        doc1_section1 = data[1]["sections"][1]
        assert doc1_section1["entities"][0]["match"] == "BRCA1"
