import logging
import time
from typing import Callable, Union, List

import hydra
import ray
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve

from kazu.data.data import Document
from kazu.pipeline import Pipeline, load_steps
from kazu.web.routes import KAZU

logger = logging.getLogger("ray")
app = FastAPI()


class SectionedWebDocument(BaseModel):
    sections: Dict[str, str]

    def to_kazu_document(self) -> Document:
         idx = uuid.uuid4().hex
         sections = [Section(text=text, name=name) for text, name in sections.items()]
         return Document(idx=idx, sections=sections)


class SimpleWebDocument(BaseModel):
    text: str

    def to_kazu_document(self) -> Document:
        return Document.create_simple_document(self.text)


WebDocument = Union[SimpleWebDocument, SectionedWebDocument]


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
    def ner(self, doc: WebDocument):
        logger.info(f"received request: {doc}")
        result = self.pipeline([doc.to_kazu_document()])
        resp_dict = result[0].as_minified_dict()
        return JSONResponse(content=resp_dict)

    @app.post(f"/{KAZU}/batch")
    def batch_ner(self, docs: List[WebDocument]):
        logger.info(f"received request: {[docs]}")
        result = self.pipeline([doc.to_kazu_document() for doc in docs])
        return JSONResponse(content=[res.as_minified_dict() for res in result])


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


def stop():
    if ray.is_initialized():
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    start()
