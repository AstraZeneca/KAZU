import pytest
import requests
from omegaconf import DictConfig

from kazu.data import Document
from kazu.tests.utils import requires_model_pack, maybe_skip_server_tests
from kazu.web.routes import (
    KAZU,
    API,
    NER_AND_LINKING,
    BATCH,
    API_DOCS_URLS,
    NO_AUTH_ENDPOINTS,
    AUTH_ENDPOINTS,
)


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

pytestmark = [requires_model_pack, maybe_skip_server_tests]


@pytest.fixture(scope="module")
def api_root_url(kazu_test_config: DictConfig) -> str:
    return f"http://127.0.0.1:{kazu_test_config.ray.serve.http_options.port}"


def test_single_document(ray_server, api_root_url):
    data = requests.post(
        f"{api_root_url}{NER_AND_LINKING}",
        json=_single_single_document_json,
        headers=ray_server,
    ).json()
    document = Document.from_dict(data[0])
    section = document.sections[0]
    assert len(section.entities) > 0
    assert section.entities[0].match == "EGFR"


def test_single_doc_deprecated_api(ray_server, api_root_url):
    # this is for the old, deprecated API - we still want to check it works until we remove it
    data = requests.post(
        f"{api_root_url}/{API}/{KAZU}",
        json=_single_single_document_json,
        headers=ray_server,
    ).json()
    document = Document.from_dict(data)
    section = document.sections[0]
    assert len(section.entities) > 0
    assert section.entities[0].match == "EGFR"


# ner_and_linking is the new api, batch is the old deprecated api, but we still
# want to check that it works as expected
@pytest.mark.parametrize("endpoint", [NER_AND_LINKING, BATCH])
def test_multiple_documents(endpoint, ray_server, api_root_url):
    url = f"{api_root_url}{endpoint}"

    response = requests.post(
        url,
        headers=ray_server,
        json=_multi_document_example_json,
    )

    data = response.json()
    doc0_section0 = Document.from_dict(data[0]).sections[0]
    assert len(doc0_section0.entities) > 0
    assert doc0_section0.entities[0].match == "EGFR"

    doc1_section1 = Document.from_dict(data[1]).sections[1]
    assert doc1_section1.entities[0].match == "BRCA1"


@pytest.mark.parametrize("endpoint", [NER_AND_LINKING, BATCH])
def test_failed_auth(endpoint, ray_server, api_root_url):
    if ray_server:  # if it's not an empty dict, it requires auth
        url = f"{api_root_url}{endpoint}"

        response = requests.post(
            url,
            headers={},
            json=_multi_document_example_json,
        )
        assert response.status_code == 401


_AUTH_ENDPOINTS_SET = set(AUTH_ENDPOINTS)
# the openai ui and json plus redoc don't show up in the openapi output (understandably!)
_ALL_DOCUMENTED_ENDPOINTS_SET = _AUTH_ENDPOINTS_SET.union(NO_AUTH_ENDPOINTS).difference(
    API_DOCS_URLS.values()
)


@pytest.fixture
def openapi_json(ray_server, api_root_url):
    openapi_json_url = f"{api_root_url}{API_DOCS_URLS['openapi_url']}"
    return requests.get(openapi_json_url).json()


def test_all_paths_present_and_documented(openapi_json):
    assert set(openapi_json["paths"]) == _ALL_DOCUMENTED_ENDPOINTS_SET


def test_auth_paths_as_expected(ray_server, openapi_json):
    endpoints_with_security = {
        path
        for path, path_data in openapi_json["paths"].items()
        for endpoint_data in path_data.values()
        if "security" in endpoint_data
    }
    if ray_server:  # if it's not an empty dict, it should have auth on it
        assert endpoints_with_security == _AUTH_ENDPOINTS_SET
    else:
        assert not endpoints_with_security
