import logging
import subprocess
import time
from typing import Any, Callable, Dict, List, Union, Optional

import hydra
import ray
from fastapi import Depends, FastAPI, HTTPException, Body
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from hydra.utils import instantiate, call
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve
from starlette.requests import HTTPConnection, Request
from starlette.responses import RedirectResponse

from kazu.data.data import Document, Entity
from kazu.pipeline import Pipeline, PipelineValueError
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.web.routes import KAZU
from kazu.web.ls_web_utils import LSWebUtils


description = """
Welcome to the Web API of Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads. This library aims to simplify the process of using state of the art NLP research in production systems. Some of the
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

The Web API is designed for light usage, if you need to run kazu for a heavy workload, please use the library directly. The Documentation for the library is available
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


_simple_doc_example = {
    "text": "A single string document that you want to recognise entities in."
    " Using the default kazu pipeline, this will recognise things like asthma, acetaminophen,"
    " EGFR and many others."
}
_sectioned_doc_example = {
    "sections": {
        "title": "a study about HER2 in breast cancer",
        "abstract": "We carried out a study on trastuzumab.",
        "fulltext": "A much longer text that probably mentions all of HER2,"
        " breast and gastric cancer, and trastuzumab.",
    }
}
_multiple_docs_mixed_type_example = [
    _simple_doc_example,
    _sectioned_doc_example,
    {"text": "Another simple doc, this one is about hypertension"},
    {
        "sections": {
            "first section": "A section about non-small cell lung cancer",
            "second section": "A section with nothing interesting in it",
            "final section": "A section about drugs: paracetamol, naproxin, ibuprofen.",
        }
    },
]


class SectionedWebDocument(BaseModel):
    sections: Dict[str, str]

    def to_kazu_document(self) -> Document:
        return Document.from_named_section_texts(self.sections)

    class Config:
        schema_extra = {"example": _sectioned_doc_example}


class SimpleWebDocument(BaseModel):
    text: str

    def to_kazu_document(self) -> Document:
        return Document.create_simple_document(self.text)

    class Config:
        schema_extra = {"example": _simple_doc_example}


WebDocument = Union[SimpleWebDocument, SectionedWebDocument]


class DocumentCollection(BaseModel):
    __root__: Union[List[WebDocument], WebDocument]

    def convert_to_kazu_documents(self) -> List[Document]:
        if isinstance(self.__root__, list):
            return [doc.to_kazu_document() for doc in self.__root__]
        else:
            return [self.__root__.to_kazu_document()]

    def __len__(self) -> int:
        if isinstance(self.__root__, list):
            return len(self.__root__)
        else:
            return 1

    class Config:
        schema_extra = {"example": _multiple_docs_mixed_type_example}


# You appear not to be able to provide multiple examples in the
# pydantic Config.schema_extra, so save this var for the Body
document_collection_examples = {
    "single_simple_doc": {
        "summary": "A single simple doc",
        "value": _simple_doc_example,
    },
    "multiple_simple_docs": {
        "summary": "Multiple simple docs",
        "value": [
            _simple_doc_example,
            {"text": "Another doc, this one is about hypertension"},
            {
                "text": "The last doc. This one doesn't have anything that should be picked up as an entity."
            },
        ],
    },
    "single_sectioned_doc": {"summary": "A single sectioned doc", "value": _sectioned_doc_example},
    "multiple_sectioned_docs": {
        "summary": "Multiple sectioned docs",
        "value": [
            _sectioned_doc_example,
            {
                "sections": {
                    "first section": "A section about non-small cell lung cancer",
                    "second section": "A section with nothing interesting in it",
                    "final section": "A section about drugs: paracetamol, naproxin, ibuprofen.",
                }
            },
        ],
    },
    "multiple_docs_mixed_types": {
        "summary": "Multiple docs, both simple and sectioned",
        "value": _multiple_docs_mixed_type_example,
    },
}
# the examples above don't make much sense for linking
linking_only_examples = {
    "single_simple_doc": {
        "summary": "A single simple doc",
        "value": {"text": "paracetamol"},
    },
    "multiple_simple_docs": {
        "summary": "Multiple simple docs",
        "value": [
            {"text": "paracetamol"},
            {"text": "acetaminophen"},
            {"text": "naproxin"},
        ],
    },
    "single_sectioned_doc": {
        "summary": "A single sectioned doc",
        "value": {"sections": {"first": "paracetamol", "second": "ibuprofen"}},
    },
    "multiple_sectioned_docs": {
        "summary": "Multiple sectioned docs",
        "value": [
            {"sections": {"first": "paracetamol", "second": "ibuprofen"}},
            {"sections": {"sec1": "naproxin", "sec2": "morphine"}},
        ],
    },
    "multiple_docs_mixed_types": {
        "summary": "Multiple docs, both simple and sectioned",
        "value": [
            {"text": "paracetamol"},
            {"sections": {"sec1": "naproxin", "sec2": "morphine"}},
            {"text": "acetaminophen"},
        ],
    },
}


class SingleEntityDocumentConverter:
    """Add an entity for the whole of every section in every document you provide.

    Essentially a subclass of :class:`DocumentCollection`, but pydantic makes this a pain to do with inheritance,
    so use composition instead of inheritance here."""

    def __init__(self, entity_free_doc_collection: DocumentCollection, entity_class: str) -> None:
        self.entity_class = entity_class
        self.entity_free_doc_collection = entity_free_doc_collection

    def convert_to_kazu_documents(self) -> List[Document]:
        documents = self.entity_free_doc_collection.convert_to_kazu_documents()
        for doc in documents:
            for section in doc.sections:
                section.entities = [
                    Entity.load_contiguous_entity(
                        # NOTE: our mapping logic is currently very dependent on the namespace
                        # I think though that we intend to get away from this with the 'mention_confidence' attribute
                        # on Entity and Curation
                        namespace="TransformersModelForTokenClassificationNerStep",
                        match=section.get_text(),
                        entity_class=self.entity_class,
                        start=0,
                        end=len(section.get_text()),
                    )
                ]
        return documents

    def __len__(self) -> int:
        return self.entity_free_doc_collection.__len__()


@serve.deployment(route_prefix="/api")
@serve.ingress(app)
class KazuWebAPI:
    """
    Web app to serve results
    """

    deploy: Callable

    def __init__(self, cfg: DictConfig):
        """
        :param cfg: DictConfig from Hydra
        """
        self.pipeline: Pipeline = instantiate(cfg.Pipeline)
        self.ls_web_utils: LSWebUtils = instantiate(cfg.LSWebUtils)
        if not cfg.Middlewares.auth_required:
            # override the openapi doc generating method
            # to remove authentication buttons.
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

    @app.get("/steps")
    def steps(self, request: Request):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu/steps endpoint")
        return JSONResponse(content=list(self.pipeline._namespace_to_step))

    @app.get("/step_groups")
    def step_groups(self, request: Request):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu/step_groups endpoint")
        if self.pipeline.step_groups is None:
            return JSONResponse(content=None)
        return JSONResponse(
            content={
                group_name: [step.namespace() for step in steps]
                for group_name, steps in self.pipeline.step_groups.items()
            }
        )

    def base_pipeline_request(
        self,
        doc_collection: Union[DocumentCollection, SingleEntityDocumentConverter],
        request: Request,
        step_namespaces: Optional[List[str]] = None,
        step_group: Optional[str] = None,
    ):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu/batch_or_not endpoint")
        logger.info(id_log_prefix + "Documents sent: %s", len(doc_collection))
        try:
            result = self.pipeline(
                doc_collection.convert_to_kazu_documents(),
                step_namespaces=step_namespaces,
                step_group=step_group,
            )
        except PipelineValueError as e:
            raise HTTPException(status_code=422, detail=e.args[0]) from e
        return JSONResponse(content=[res.as_minified_dict() for res in result])

    @app.post(f"/{KAZU}/ner_and_linking")
    def ner_and_linking(
        self,
        request: Request,
        doc_collection: DocumentCollection = Body(examples=document_collection_examples),
        token=Depends(oauth2_scheme),
    ):
        return self.base_pipeline_request(
            doc_collection=doc_collection, request=request, step_namespaces=None
        )

    @app.post(f"/{KAZU}/custom_pipeline_steps")
    def custom_pipeline_steps(
        self,
        request: Request,
        doc_collection: DocumentCollection = Body(examples=document_collection_examples),
        step_group: Optional[str] = Body(default=None),
        token=Depends(oauth2_scheme),
        steps: Optional[List[str]] = None,
    ):
        return self.base_pipeline_request(
            doc_collection=doc_collection,
            request=request,
            step_namespaces=steps,
            step_group=step_group,
        )

    @app.post(f"/{KAZU}/ner_only")
    def ner_only(
        self,
        request: Request,
        doc_collection: DocumentCollection = Body(examples=document_collection_examples),
        token=Depends(oauth2_scheme),
    ):
        """An endpoint that only calls steps that do Named Entity Recognition (NER).

        Note that this functionality was already available via the custom_steps endpoint,
        this just provided a convenient and visible/discoverable endpoint for users."""
        return self.base_pipeline_request(
            doc_collection=doc_collection, request=request, step_group="ner_only"
        )

    @app.post(f"/{KAZU}/linking_only")
    def linking_only(
        self,
        entity_class: str,
        request: Request,
        doc_collection: DocumentCollection = Body(examples=linking_only_examples),
        token=Depends(oauth2_scheme),
    ):
        linking_doc_collection = SingleEntityDocumentConverter(
            doc_collection, entity_class=entity_class
        )
        return self.base_pipeline_request(
            doc_collection=linking_doc_collection, request=request, step_group="linking_only"
        )

    # To be deprecated, after we check how often it gets called after being 'hidden'
    @app.post(f"/{KAZU}", deprecated=True)
    def ner(self, doc: WebDocument, request: Request, token=Depends(oauth2_scheme)):
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu endpoint")
        logger.info(id_log_prefix + "Document: %s", doc)
        result = self.pipeline([doc.to_kazu_document()])
        resp_dict = result[0].as_minified_dict()
        return JSONResponse(content=resp_dict)

    # To be deprecated, after we check how often it gets called after being 'hidden'
    @app.post(f"/{KAZU}/batch", deprecated=True)
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


@serve.deployment(route_prefix="/ui")
@serve.ingress(app)
class KazuWebUI:

    deploy: Callable

    def __init__(self, ui_cfg: DictConfig):
        app.mount(
            ui_cfg.baseURL,
            StaticFiles(directory=ui_cfg.staticFilesPath, html=True),
            name="static-ui-content",
        )

    @app.get("/")
    def root(self):
        return RedirectResponse("/index.html")


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    """
    deploy the web app to Ray Serve

    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    call(cfg.ray.init)
    call(cfg.ray.serve)

    KazuWebAPI.deploy(cfg)
    if (ui_conf := cfg.ray.get("ui")) is not None:
        KazuWebUI.deploy(ui_conf)

    if not cfg.ray.serve.detached:
        while True:
            logger.info(serve.list_deployments())
            time.sleep(600)


def stop():
    if ray.is_initialized():
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    start()
