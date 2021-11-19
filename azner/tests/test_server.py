import requests
from hydra import initialize_config_module, compose

from azner.data.data import Document
from azner.tests.utils import get_TransformersModelForTokenClassificationNerStep_model_path
from azner.web.routes import AZNER
from azner.web.server import start


def test_api():

    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "ray=local",
                f"TransformersModelForTokenClassificationNerStep.path={get_TransformersModelForTokenClassificationNerStep_model_path()}",
            ],
        )
        start(cfg)
        response = requests.post(
            f"http://127.0.0.1:{cfg.ray.serve.port}/api/{AZNER}/", json={"text": "hello"}
        ).json()
        result = Document(**response).rehydrate()
        assert isinstance(result, Document)
