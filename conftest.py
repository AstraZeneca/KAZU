import os
from typing import List, Tuple, Set

import pytest
from hydra import compose, initialize_config_dir

from kazu.data.data import SynonymTermWithMetrics
from kazu.modelling.annotation.label_studio import (
    LabelStudioManager,
)
from kazu.modelling.database.in_memory_db import SynonymDatabase
from kazu.modelling.ontology_preprocessing.base import IDX, DEFAULT_LABEL, SYN, MAPPING_TYPE
from kazu.tests.utils import CONFIG_DIR, DummyParser, make_dummy_parser


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


@pytest.fixture(scope="session")
def set_up_p27_test_case() -> Tuple[Set[SynonymTermWithMetrics], DummyParser]:

    dummy_data = {
        IDX: ["1", "1", "2", "2", "2", "3", "3", "3"],
        DEFAULT_LABEL: ["CDKN1B", "CDKN1B", "PAK2", "PAK2", "PAK2", "ZNRD2", "ZNRD2", "ZNRD2"],
        SYN: [
            "cyclin-dependent kinase inhibitor 1B (p27, Kip1)",
            "CDKN1B",
            "PAK-2p27",
            "p27",
            "PAK2",
            "Autoantigen p27",
            "ZNRD2",
            "p27",
        ],
        MAPPING_TYPE: ["", "", "", "", "", "", "", ""],
    }
    parser = make_dummy_parser(
        in_path="", data=dummy_data, name="test_tfidf_parsr", source="test_tfidf_parsr"
    )
    parser.populate_databases()
    terms_with_metrics = set(
        SynonymTermWithMetrics.from_synonym_term(term)
        for term in SynonymDatabase().get_all(parser.name).values()
    )
    return terms_with_metrics, parser


@pytest.fixture(scope="session")
def label_studio_manager() -> LabelStudioManager:

    label_studio_url_and_port = os.environ["LS_URL_PORT"]
    headers = {
        "Authorization": f"Token {os.environ['LS_TOKEN']}",
        "Content-Type": "application/json",
    }
    ls_project = os.environ["LS_PROJECT_NAME"]
    manager = LabelStudioManager(
        project_name=ls_project, headers=headers, url=label_studio_url_and_port
    )
    return manager
