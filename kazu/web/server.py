import logging
import subprocess
import time
from typing import Any, Callable, Dict, List, Union

import hydra
import ray
from fastapi import Depends, FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve
from starlette.requests import HTTPConnection, Request
from starlette.middleware.authentication import AuthenticationMiddleware

from kazu.data.data import Document
from kazu.pipeline import Pipeline
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.web.routes import KAZU
from kazu.web.ls_web_utils import LSWebUtils


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


def openapi_no_auth() -> Dict[str, Any]:
    """Remove the bits of the openapi schema that put auth buttons on the Swagger UI.

    When we don't configure any Authentication middleware, we otherwise still get the
    buttons, even though they aren't necessary to fill in and don't do anything, which
    may be confusing to users."""
    if not app.openapi_schema:
        base_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            terms_of_service=app.terms_of_service,
            contact=app.contact,
            license_info=app.license_info,
            routes=app.routes,
            tags=app.openapi_tags,
            servers=app.servers,
        )
        del base_schema["components"]["securitySchemes"]
        for _path, path_data in base_schema["paths"].items():
            for _method, endpoint_data in path_data.items():
                # security in the openapi json is only present
                # to start with on those endpoints with
                # token=Depends(oauth2_scheme)
                if "security" in endpoint_data:
                    del endpoint_data["security"]
        app.openapi_schema = base_schema
    return app.openapi_schema


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


def get_id_log_prefix_if_available(request: HTTPConnection) -> str:
    """Utility function for generating the appropriate prefix for logs.

    :param request: Starlette HTTPConnection object
    :returns: Prefix to pre-pend to log messages containing the request id.
    """
    req_id = get_request_id(request)
    if req_id is not None:
        return "ID: " + req_id + " "
    else:
        return ""


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

    def __init__(self, cfg: DictConfig, auth_required: bool):
        """
        :param cfg: DictConfig from Hydra
        :param auth_required: Does this require authentication to use?
            If not, we'll make sure the openapi docs don't generate auth buttons.
        """
        self.pipeline: Pipeline = instantiate(cfg.Pipeline)
        self.ls_web_utils: LSWebUtils = instantiate(cfg.LSWebUtils)
        if not auth_required:
            app.openapi = openapi_no_auth  # type: ignore[assignment]
            # we need the 'type: ignore' as otherwise mypy gives an error:
            # mypy doesn't like assigning to a method - it isn't able
            # to infer whether typing is consistent if you do this.
            # However, note that '@decorator' syntax is just syntactic
            # sugar for doing this, so this is a *fairly* reasonable
            # thing to do.

    @app.get("/")
    def get(self):
        logger.info("received request to root /")
        return "Welcome to KAZU."

    @app.post(f"/{KAZU}")
    def ner(self, doc: WebDocument, request: Request, token=Depends(oauth2_scheme)):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu endpoint")
        logger.info(id_log_prefix + "Document: %s", doc)
        result = self.pipeline([doc.to_kazu_document()])
        resp_dict = result[0].as_minified_dict()
        return JSONResponse(content=resp_dict)

    @app.post(f"/{KAZU}/batch")
    def batch_ner(self, docs: List[WebDocument], request: Request, token=Depends(oauth2_scheme)):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu/batch endpoint")
        logger.info(id_log_prefix + "Documents sent: %s", len(docs))
        result = self.pipeline([doc.to_kazu_document() for doc in docs])
        return JSONResponse(content=[res.as_minified_dict() for res in result])

    @app.post(f"/{KAZU}/ls-annotations")
    def ls_annotations(self, doc: WebDocument, request: Request):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu/ls-annotations endpoint")
        logger.info(id_log_prefix + "Document: %s", doc)
        result = self.pipeline([doc.to_kazu_document()])[0]
        ls_view, ls_tasks = self.ls_web_utils.kazu_doc_to_ls(result)
        return JSONResponse(
            content={"ls_view": ls_view, "ls_tasks": ls_tasks, "doc": result.as_minified_dict()}
        )


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    """
    deploy the web app to Ray Serve

    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    ray.init(address=cfg.ray.address, namespace="serve", num_cpus=cfg.ray.num_cpus, object_store_memory=cfg.ray.object_store_memory)
    middlewares = instantiate(cfg.Middlewares)
    # Bind on 0.0.0.0 to expose the HTTP server on external IPs.
    serve.start(
        detached=cfg.ray.detached,
        http_options={"host": "0.0.0.0", "location": "EveryNode", **middlewares},
    )

    if any(m.cls is AuthenticationMiddleware for m in middlewares["middlewares"]):
        auth_required = True
    else:
        auth_required = False

    KazuWebApp.deploy(cfg, auth_required=auth_required)
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
