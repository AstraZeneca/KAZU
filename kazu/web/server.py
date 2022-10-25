import logging
import time
from typing import Callable, Union, List

import hydra
import ray
from fastapi import FastAPI
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve

from kazu.data.data import Document
from kazu.pipeline import Pipeline, load_steps
from kazu.web.routes import KAZU

logger = logging.getLogger("ray")
app = FastAPI()


class WebDocument(BaseModel):
    sections: List[str]


class SimpleWebDocument(BaseModel):
    text: str


@serve.deployment(route_prefix="/api")
@serve.ingress(app)
class KazuWebApp:
    """
    Web app to serve results
    """

    deploy: Callable

    def __init__(self, cfg: DictConfig):
        """
        :param cfg: DictConfig from Hydra
        """
        self.pipeline = Pipeline(load_steps(cfg))

    @app.get("/")
    def get(self):
        logger.info("received request to root /")
        return "Welcome to KAZU."

    @app.post(f"/{KAZU}")
    def ner(self, doc: Union[SimpleWebDocument, WebDocument]):
        logger.info(f"received request: {doc}")
        if isinstance(doc, SimpleWebDocument):
            result = self.pipeline([Document.create_simple_document(doc.text)])
        else:
            result = self.pipeline([Document.from_section_texts(doc.sections)])
        return result[0].json()


@hydra.main(config_path="../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    """
    deploy the web app to Ray Serve

    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    ray.init(address=cfg.ray.address, namespace="serve")
    # Bind on 0.0.0.0 to expose the HTTP server on external IPs.
    serve.start(
        detached=cfg.ray.detached, http_options={"host": "0.0.0.0", "location": "EveryNode"}
    )
    KazuWebApp.deploy(cfg)
    if not cfg.ray.detached:
        while True:
            logger.info(serve.list_deployments())
            time.sleep(10)


if __name__ == "__main__":
    start()
