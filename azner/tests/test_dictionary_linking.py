import os

import pydash
from hydra import initialize_config_module, compose
from hydra.utils import instantiate

from tests.utils import entity_linking_hard_cases


class AcceptanceTestError(Exception):
    def __init__(self, message):
        self.message = message


def test_dictionary_entity_linking():
    pass
