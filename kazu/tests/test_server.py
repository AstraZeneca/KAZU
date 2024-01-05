import pytest
import requests
from omegaconf import DictConfig

from kazu.tests.utils import requires_model_pack
from kazu.web.routes import KAZU, API


_single_single_document_json = {"text": "EGFR is an important gene in breast cancer"}

_multi_document_example_json = [
    _single_single_document_json,
    {
        "sections": {
            "sec1": "hello this is the first section",
            "sec2": "the second section mentions BRCA1",
        }
    },
]

pytestmark = requires_model_pack


@pytest.fixture(scope="module")
def api_base_url(kazu_test_config: DictConfig) -> str:
    return f"http://127.0.0.1:{kazu_test_config.ray.serve.http_options.port}/{API}/{KAZU}"


def test_single_document(ray_server, api_base_url):
    data = requests.post(
        f"{api_base_url}/ner_and_linking",
        json=_single_single_document_json,
        headers=ray_server,
    ).json()
    document = data[0]
    section = document["sections"][0]
    assert len(section["entities"]) > 0
    assert section["entities"][0]["match"] == "EGFR"


def test_single_doc_deprecated_api(ray_server, api_base_url):
    # this is for the old, deprecated API - we still want to check it works until we remove it
    data = requests.post(
        api_base_url,
        json=_single_single_document_json,
        headers=ray_server,
    ).json()
    section = data["sections"][0]
    assert len(section["entities"]) > 0
    assert section["entities"][0]["match"] == "EGFR"


# ner_and_linking is the new api, batch is the old deprecated api, but we still
# want to check that it works as expected
@pytest.mark.parametrize("endpoint", ["ner_and_linking", "batch"])
def test_multiple_documents(endpoint, ray_server, api_base_url):
    url = f"{api_base_url}/{endpoint}"

    response = requests.post(
        url,
        headers=ray_server,
        json=_multi_document_example_json,
    )

    data = response.json()
    print(data)
    doc0_section0 = data[0]["sections"][0]
    assert len(doc0_section0["entities"]) > 0
    assert doc0_section0["entities"][0]["match"] == "EGFR"

    doc1_section1 = data[1]["sections"][1]
    assert doc1_section1["entities"][0]["match"] == "BRCA1"


@pytest.mark.parametrize("endpoint", ["ner_and_linking", "batch"])
def test_failed_auth(endpoint, ray_server, api_base_url):
    if ray_server:  # if it's not an empty dict, it requires auth
        url = f"{api_base_url}/{endpoint}"

        response = requests.post(
            url,
            headers={},
            json=_multi_document_example_json,
        )
        assert response.status_code == 401
