import requests
from hydra import initialize_config_module, compose

from azner.web.routes import AZNER
from azner.web.server import start


def test_api():

    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(config_name='config',overrides=[
            'ray=local'
        ])
        start(cfg)
        response = requests.post(f"http://127.0.0.1:8000/api/{AZNER}/",json={"text":"hello"}).text
        assert response == '"not implemented"'
