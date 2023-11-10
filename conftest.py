import os

import jwt
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from kazu.data.data import Document, SynonymTermWithMetrics
from kazu.annotation.label_studio import (
    LabelStudioManager,
)
from kazu.database.in_memory_db import SynonymDatabase
from kazu.ontology_preprocessing.base import (
    IDX,
    DEFAULT_LABEL,
    SYN,
    MAPPING_TYPE,
    OntologyParser,
)
from kazu.tests.utils import CONFIG_DIR, DummyParser
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.web.server import start, stop
from kazu.utils.caching import kazu_disk_cache
from kazu.steps.linking.post_processing.disambiguation.context_scoring import TfIdfScorer
from kazu.utils.utils import Singleton
from kazu.steps.joint_ner_and_linking.memory_efficient_string_matching import (
    MemoryEfficientStringMatchingStep,
)


@pytest.fixture(scope="session")
def override_kazu_test_config():
    def _override_kazu_test_config(overrides: list[str]) -> DictConfig:
        """Return an optionally overriden copy of the kazu test config.

        :return:
        """

        # needs a str, can't take a Path
        with initialize_config_dir(version_base=HYDRA_VERSION_BASE, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=overrides)
        return cfg

    return _override_kazu_test_config


@pytest.fixture(scope="session")
def kazu_test_config(override_kazu_test_config):
    return override_kazu_test_config(overrides=[])


_SIMPLE_TEST_CASE_DATA = [
    ("EGFR is a gene", "gene"),
    ("CAT1 is a gene", "gene"),
    ("my cat sat on the mat", "species"),
    ("For the treatment of anorexia nervosa.", "disease"),
]


@pytest.fixture(scope="function")
def ner_simple_test_cases() -> list[Document]:
    """Return simple Documents testing NER functionality.

    This needs to be function-scoped because Documents can be mutated.
    """
    docs = [Document.create_simple_document(x[0]) for x in _SIMPLE_TEST_CASE_DATA]
    return docs


@pytest.fixture(scope="session")
def set_up_p27_test_case() -> tuple[set[SynonymTermWithMetrics], DummyParser]:

    dummy_data = {
        IDX: ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
        DEFAULT_LABEL: [
            "CDKN1B",
            "CDKN1B",
            "CDKN1B",
            "PAK2",
            "PAK2",
            "PAK2",
            "ZNRD2",
            "ZNRD2",
            "ZNRD2",
        ],
        SYN: [
            "cyclin-dependent kinase inhibitor 1B (p27, Kip1)",
            "CDKN1B",
            "p27",
            "PAK-2p27",
            "p27",
            "PAK2",
            "Autoantigen p27",
            "ZNRD2",
            "p27",
        ],
        MAPPING_TYPE: ["", "", "", "", "", "", "", "", ""],
    }
    parser = DummyParser(data=dummy_data, name="test_tfidf_parser", source="test_tfidf_parser")
    parser.populate_databases()
    terms_with_metrics = set(
        SynonymTermWithMetrics.from_synonym_term(term)
        for term in SynonymDatabase().get_all(parser.name).values()
    )
    return terms_with_metrics, parser


@pytest.fixture(scope="session")
def make_label_studio_manager():
    # using a 'factory fixture' here gives us the ability to share code when
    # we need to make a label studio manager which is the same except for a custom project.
    def _make_label_studio_manager(
        project_name: str = os.environ["LS_PROJECT_NAME"],
    ) -> LabelStudioManager:

        label_studio_url_and_port = os.environ["LS_URL_PORT"]
        headers = {
            "Authorization": f"Token {os.environ['LS_TOKEN']}",
            "Content-Type": "application/json",
        }
        manager = LabelStudioManager(
            project_name=project_name, headers=headers, url=label_studio_url_and_port
        )
        return manager

    return _make_label_studio_manager


@pytest.fixture(scope="function")
def ray_server(override_kazu_test_config):
    # clear any residual singleton info, as ray runs separate processes and
    # hanging resources can cause OOM
    Singleton.clear_all()
    cfg = override_kazu_test_config(
        overrides=["ray=local", "ray.serve.detached=true"],
    )
    start(cfg)
    yield {}
    stop()


@pytest.fixture(scope="function")
def ray_server_with_jwt_auth(override_kazu_test_config):
    # clear any residual singleton info, as ray runs separate processes and
    # hanging resources can cause OOM
    Singleton.clear_all()
    os.environ["KAZU_JWT_KEY"] = "this secret key is not secret"
    cfg = override_kazu_test_config(
        overrides=["ray=local", "ray.serve.detached=true", "Middlewares=jwt"],
    )
    start(cfg)
    yield {
        "Authorization": f'Bearer {jwt.encode({"username": "user"}, os.environ["KAZU_JWT_KEY"], algorithm="HS256")}'
    }
    stop()


@pytest.fixture(scope="function")
def mock_kazu_disk_cache_on_parsers(monkeypatch):
    """Disables the caching functions on OntologyParsers during testing.

    Since we employ diskcache in a slightly unusual way, we need to use some python
    tricks to turn the caching on/off during tests.

    :param monkeypatch:
    :return:
    """

    def do_nothing(*args, **kwargs):
        return

    # list of memoize functions
    funcs = [
        OntologyParser._populate_databases,
        OntologyParser.export_metadata,
        OntologyParser.export_synonym_terms,
    ]
    # ...mapped to the underlying function
    # type ignore needed because it would be a pain to try and tell mypy that
    # these are all 'wrapped' decorated functions so they have this attribute.
    original_funcs = {func: func.__wrapped__ for func in funcs}  # type: ignore[attr-defined]

    for func, original_func in original_funcs.items():
        # set the __cache_key__ to do nothing
        original_func.__cache_key__ = do_nothing
        # set the memoized function to the original function
        monkeypatch.setattr(OntologyParser, func.__name__, original_func)  # type: ignore[attr-defined] # doesn't know it will have __name__
    # also prevent the original cache from deleting anything
    monkeypatch.setattr(kazu_disk_cache, "delete", do_nothing)
    # run the calling test
    yield
    # delete the "do_nothing" function for __cache_key__ at the end of the test
    for original_func in original_funcs.values():
        del original_func.__cache_key__
    # What can possibly go wrong?


@pytest.fixture(scope="function")
def mock_build_vectoriser_cache(monkeypatch):
    # type ignore as above - mypy doesn't know this function is 'wrapped'
    monkeypatch.setattr(TfIdfScorer, "build_vectorizers", TfIdfScorer.build_vectorizers.__wrapped__)  # type: ignore[attr-defined]


@pytest.fixture(scope="function")
def mock_build_fast_string_matcher_cache(monkeypatch):
    # type ignore as above - mypy doesn't know this function is 'wrapped'
    monkeypatch.setattr(
        MemoryEfficientStringMatchingStep,
        "_create_automaton",
        MemoryEfficientStringMatchingStep._create_automaton.__wrapped__,  # type: ignore[attr-defined]
    )
