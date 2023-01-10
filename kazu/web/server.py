import logging
import subprocess
import time
from typing import Callable, Dict, List, Union

import hydra
import ray
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve
from starlette.requests import HTTPConnection, Request

from kazu.data.data import Document
from kazu.pipeline import Pipeline
from kazu.web.routes import KAZU

description = """
Welcome to the Web API of Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads. This library aims to simplify the process of using state of the art NLP research in production systems. Some of the
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

The Web API is designed for light usage, if you need to run kazu for a heavy workload, please use the library directly. The Documentaion for the library is available
*[here](https://astrazeneca.github.io/KAZU/_build/html/index.html)*.
"""
logger = logging.getLogger("ray")
kazu_version = (
    subprocess.check_output("pip show kazu | grep Version", shell=True)
    .decode("utf-8")
    .split(" ")[1]
    .strip()
)
app = FastAPI(
    title="Kazu - Biomedical NLP Framework",
    description=description,
    version=kazu_version,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

oauth2_scheme = HTTPBearer(auto_error=False)


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
    def ner(self, doc: WebDocument, request: Request, token=Depends(oauth2_scheme)):
        req_id = get_request_id(request)
        logger.info("ID: %s Request to kazu endpoint" % req_id)
        logger.info(f"ID: {req_id} Document: {doc}")
        result = self.pipeline([doc.to_kazu_document()])
        resp_dict = result[0].as_minified_dict()
        return JSONResponse(content=resp_dict)

    @app.post(f"/{KAZU}/batch")
    def batch_ner(self, docs: List[WebDocument], token=Depends(oauth2_scheme)):
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
        if req_id is not None:
            response.headers.append(
                "X-request-id",
                req_id,
            )
        logger.info(f"ID: {req_id} Response Code: {response.status_code}")
        return response


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
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
