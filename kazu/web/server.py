import logging
import time
from typing import Callable, Union, List, Dict

import hydra
import ray
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve

from starlette.requests import Request, HTTPConnection
from kazu.data.data import Document
from kazu.pipeline import Pipeline
from kazu.web.routes import KAZU

logger = logging.getLogger("ray")
app = FastAPI()


def get_request_id(request: HTTPConnection) -> Union[str, None]:
    """Utility function for extracting custom header from HTTPConnection Object.

    :param request: Starlette HTTPConnection object
    :returns: ID string
    """
    key, req_id = request.headers.__dict__["_list"][-1]
    if key.decode("latin-1").lower() != "x-request-id":
        return None
    return req_id.decode("latin-1")


class SectionedWebDocument(BaseModel):
    sections: Dict[str, str]

    def to_kazu_document(self) -> Document:
        return Document.from_named_section_texts(self.sections)


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
        self.pipeline: Pipeline = instantiate(cfg.Pipeline)

    @app.get("/")
    def get(self):
        logger.info("received request to root /")
        return "Welcome to KAZU."

    @app.post(f"/{KAZU}")
    def ner(self, doc: WebDocument, request: Request):
        req_id = get_request_id(request)
        logger.info("ID: %s Request to kazu endpoint" % req_id)
        logger.info(f"ID: {req_id} Document: {doc}")
        result = self.pipeline([doc.to_kazu_document()])
        resp_dict = result[0].as_minified_dict()
        return JSONResponse(content=resp_dict)

    @app.post(f"/{KAZU}/batch")
    def batch_ner(self, docs: List[WebDocument]):
        result = self.pipeline([doc.to_kazu_document() for doc in docs])
        return JSONResponse(content=[res.as_minified_dict() for res in result])

    @app.middleware("http")
    async def add_id_header(
        request: Request,
        call_next,
    ):
        """Add request ID to response header.

        :param request: Request object
        :param call_next: Function for passing middleware to FASTAPI

        :returns: Response object
        """
        req_id = get_request_id(request)
        response = await call_next(request)
        response.headers.append(
            "X-request-id",
            req_id,
        )
        logger.info(f"ID: {req_id} Response Code: {response.status_code}")
        return response


@hydra.main(config_path="../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    """
    deploy the web app to Ray Serve

    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    ray.init(address=cfg.ray.address, namespace="serve")
    middlewares = instantiate(cfg.Middlewares)
    # Bind on 0.0.0.0 to expose the HTTP server on external IPs.
    serve.start(
        detached=cfg.ray.detached,
        http_options={"host": "0.0.0.0", "location": "EveryNode", **middlewares},
    )

    KazuWebApp.deploy(cfg)
    if not cfg.ray.detached:
        while True:
            logger.info(serve.list_deployments())
            time.sleep(600)


def stop():
    if ray.is_initialized():
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    start()
