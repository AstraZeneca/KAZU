import logging
import os
import time
import packaging.version
from typing import Any, Union, Optional
from collections.abc import Callable

import hydra

KAZU_WEBSERVER_SPINUP_TIMEOUT = os.environ.setdefault(
    "RAY_SERVE_PROXY_READY_CHECK_TIMEOUT_S", "180"
)
"""A timeout limit for spinning up the kazu server, including pipeline load time.

Defaults to 3 minutes, but will read the ``RAY_SERVE_PROXY_READY_CHECK_TIMEOUT_S``
environment variable and use that value if present.

If you have a custom pipeline that is very slow to spin up, you may need to increase this timeout.
If this is the case, you will get an error like::

   ray.exceptions.RayActorError: The actor died unexpectedly before finishing this task.
           class_name: HTTPProxyActor
           actor_id: d241a36c326e982cac79013101000000
           pid: 274
           name: SERVE_CONTROLLER_ACTOR:FmBUWo:SERVE_PROXY_ACTOR-71c5a59a2c003ba52ae3f190e537d552af676c9d895ad98c7942cfe0
           namespace: serve
           ip: 172.29.34.98
   The actor is dead because it was killed by `ray.kill`

.. note::

   Normally we provide options to control environment variables within the hydra config,
   but unfortunately we can't set this with hydra because ray reads this at import time.

   Attempts to work around this with local imports interfere with the ray
   serve/FastAPI integration and break the server.
"""

import ray
from fastapi import Depends, FastAPI, HTTPException, Body
from fastapi import __version__ as fastapi_version
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from hydra.utils import instantiate, call
from omegaconf import DictConfig
from pydantic import BaseModel
from ray import serve
from starlette.requests import HTTPConnection, Request

from kazu import __version__ as kazu_version
from kazu.data import Document, Entity
from kazu.pipeline import Pipeline, PipelineValueError
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.web.routes import (
    KAZU,
    API,
    API_DOCS_URLS,
    STEPS,
    STEP_GROUPS,
    NER_AND_LINKING,
    CUSTOM_PIPELINE_STEPS,
    NER_ONLY,
    LINKING_ONLY,
    BATCH,
    LS_ANNOTATIONS,
    STEP_GROUP_WITH_VAR,
)
from kazu.web.ls_web_utils import LSWebUtils


if packaging.version.parse(fastapi_version) < packaging.version.parse("0.99.0"):
    _OPENAPI_EXAMPLES_FIELD = "examples"
else:
    _OPENAPI_EXAMPLES_FIELD = "openapi_examples"

description = """
Welcome to the Web API of Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads. This library aims to simplify the process of using state of the art NLP research in production systems. Some of the
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

The Web API is designed for light usage, if you need to run kazu for a heavy workload, please use the library directly. The Documentation for the library is available
*[here](https://astrazeneca.github.io/KAZU/index.html)*.
"""
logger = logging.getLogger("ray")
app = FastAPI(
    title="Kazu - Biomedical NLP Framework",
    description=description,
    version=kazu_version,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    **API_DOCS_URLS,
)


def openapi_no_auth() -> dict[str, Any]:
    """Remove the bits of the openapi schema that put auth buttons on the Swagger UI.

    When we don't configure any Authentication middleware, we otherwise still get the
    buttons, even though they aren't necessary to fill in and don't do anything, which
    may be confusing to users.
    """
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
    possible_request_id = request.headers.getlist("x-request-id")
    num_request_id_headers = len(possible_request_id)
    if num_request_id_headers == 0:
        return None

    # we only expect a single request id header
    assert num_request_id_headers == 1
    return possible_request_id[0]


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


def log_request_to_path_with_prefix(
    request: HTTPConnection, log_prefix: Optional[str] = None
) -> None:
    """Utility function to log the log prefix plus the endpoint the request was sent to.

    :param request: Starlette HTTPConnection object
    :param log_prefix: the prefix to log. Provided in case the log prefix has already been
        calculated in order to save re-calculation. Will call
        :func:`get_id_log_prefix_if_available` if not provided (or None is provided).
    """
    if log_prefix is None:
        log_prefix = get_id_log_prefix_if_available(request)
    logger.info(log_prefix + "Request to " + str(request.scope["path"]) + " endpoint")


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
    sections: dict[str, str]

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
    __root__: Union[list[WebDocument], WebDocument]

    def convert_to_kazu_documents(self) -> list[Document]:
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

    Essentially a subclass of :class:`DocumentCollection`, but pydantic
    makes this a pain to do with inheritance, so use composition instead
    of inheritance here.
    """

    def __init__(self, entity_free_doc_collection: DocumentCollection, entity_class: str) -> None:
        self.entity_class = entity_class
        self.entity_free_doc_collection = entity_free_doc_collection

    def convert_to_kazu_documents(self) -> list[Document]:
        documents = self.entity_free_doc_collection.convert_to_kazu_documents()
        for doc in documents:
            for section in doc.sections:
                section.entities = [
                    Entity.load_contiguous_entity(
                        namespace="KAZUAPILinkingOnlyRequest",
                        match=section.text,
                        entity_class=self.entity_class,
                        start=0,
                        end=len(section.text),
                    )
                ]
        return documents

    def __len__(self) -> int:
        return self.entity_free_doc_collection.__len__()


@serve.deployment
@serve.ingress(app)
class KazuWebAPI:
    """Web app to serve results."""

    bind: Callable

    def __init__(self, cfg: DictConfig):
        """
        :param cfg: DictConfig from Hydra
        """
        self.pipeline: Pipeline = instantiate(cfg.Pipeline)
        self.ls_web_utils: LSWebUtils = instantiate(cfg.LSWebUtils)
        if not cfg.Middlewares.auth_required:
            # override the openapi doc generating method
            # to remove authentication buttons.
            app.openapi = openapi_no_auth  # type: ignore[method-assign]
            # we need the 'type: ignore' as otherwise mypy gives an error:
            # mypy doesn't like assigning to a method - it isn't able
            # to infer whether typing is consistent if you do this.
            # However, note that '@decorator' syntax is just syntactic
            # sugar for doing this, so this is a *fairly* reasonable
            # thing to do.

        if (ui_conf := cfg.ray.get("ui")) is not None:
            app.mount(
                ui_conf.baseURL,
                StaticFiles(directory=ui_conf.staticFilesPath, html=True),
                name="static-ui-content",
            )

    @app.get("/")
    @app.get(f"/{API}")
    @app.get(f"/{API}/")
    def get(self):
        logger.info("received request to root /")
        return "Welcome to KAZU."

    @app.get(STEPS)
    def steps(self, request: Request) -> JSONResponse:
        """Get a list of the steps in the the deployed pipeline."""
        log_request_to_path_with_prefix(request)
        return JSONResponse(content=list(self.pipeline._namespace_to_step))

    @app.get(STEP_GROUPS)
    def step_groups(self, request: Request) -> JSONResponse:
        """Get the step groups configured in the deployed pipeline, (including showing
        the steps in each group)."""
        log_request_to_path_with_prefix(request)
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
        step_namespaces: Optional[list[str]] = None,
        step_group: Optional[str] = None,
    ) -> JSONResponse:
        id_log_prefix = get_id_log_prefix_if_available(request)
        log_request_to_path_with_prefix(request, log_prefix=id_log_prefix)
        logger.info(id_log_prefix + "Documents sent: %s", len(doc_collection))
        try:
            result = self.pipeline(
                doc_collection.convert_to_kazu_documents(),
                step_namespaces=step_namespaces,
                step_group=step_group,
            )
        except PipelineValueError as e:
            raise HTTPException(status_code=422, detail=e.args[0]) from e
        return JSONResponse(content=[res.to_dict() for res in result])

    @app.post(NER_AND_LINKING)
    def ner_and_linking(
        self,
        request: Request,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
        doc_collection: DocumentCollection = Body(
            **{_OPENAPI_EXAMPLES_FIELD: document_collection_examples}  # type:ignore[arg-type]
            # type ignore is needed because the dict unwrapping means mypy doesn't know
            # which argument is being used, and therefore what type
            # it should have.
        ),
    ) -> JSONResponse:
        """Run NER and Linking over the input document or documents.

        This is the default behaviour of the kazu library.
        """
        return self.base_pipeline_request(
            doc_collection=doc_collection, request=request, step_namespaces=None
        )

    @app.post(CUSTOM_PIPELINE_STEPS)
    def custom_pipeline_steps(
        self,
        request: Request,
        doc_collection: DocumentCollection,
        steps: Optional[list[str]] = None,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    ) -> JSONResponse:
        """Run specific steps over the provided document or documents.

        This is advanced functionality not expected to be needed by most ordinary users.

        The steps must be contained in the current pipeline (call the steps api to see
        what is available). The order the steps run in is the order provided in the API,
        not the order specified in the pipeline. Note that this means you can customize
        the order steps run in (although doing so effectively requires a good
        understanding of the steps and their behaviour).
        """
        return self.base_pipeline_request(
            doc_collection=doc_collection,
            request=request,
            step_namespaces=steps,
        )

    @app.post(NER_ONLY)
    def ner_only(
        self,
        request: Request,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
        doc_collection: DocumentCollection = Body(
            **{_OPENAPI_EXAMPLES_FIELD: linking_only_examples}  # type:ignore[arg-type]
            # type ignore as above
        ),
    ) -> JSONResponse:
        """Call only steps that do Named Entity Recognition (NER).

        Note that this functionality is already available via the custom_steps endpoint,
        this just provides a convenient and visible/discoverable endpoint for users.
        """
        return self.base_pipeline_request(
            doc_collection=doc_collection, request=request, step_group="ner_only"
        )

    @app.post(LINKING_ONLY)
    def linking_only(
        self,
        entity_class: str,
        request: Request,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
        doc_collection: DocumentCollection = Body(
            **{_OPENAPI_EXAMPLES_FIELD: document_collection_examples}  # type:ignore[arg-type]
            # type ignore as above
        ),
    ) -> JSONResponse:
        """Call only steps that do Entity Linking (EL). Also known as 'entity
        normalization'.

        This API assumes that the whole of each section in each document is an entity
        mention. It will also try to link all entities to a single type of entity,
        specified in the entity_class query parameter. You can of course call the API
        multiple times with different entity_class values with the relevant documents
        for each entity_class.

        If you don't know what the entity type of the entities is, you can just call the
        ner_and_linking endpoint instead. It may however give you back entities that
        don't span the whole section.
        """
        linking_doc_collection = SingleEntityDocumentConverter(
            doc_collection, entity_class=entity_class
        )
        return self.base_pipeline_request(
            doc_collection=linking_doc_collection, request=request, step_group="linking_only"
        )

    # To be removed, after we check how often it gets called after being 'hidden'
    @app.post(f"/{API}/{KAZU}", deprecated=True)
    def ner(
        self,
        doc: WebDocument,
        request: Request,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    ) -> JSONResponse:
        """Deprecated endpoint: use ner_and_linking.

        This endpoint will be removed in future.

        Behaves the same as ner_and_linking, but now we have the
        ner_only and linking_only endpoints, the naming here had the
        potential to be confusing.
        """
        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu endpoint")
        logger.info(id_log_prefix + "Document: %s", doc)
        result = self.pipeline([doc.to_kazu_document()])
        resp_dict = result[0].to_dict()
        return JSONResponse(content=resp_dict)

    # To be removed, after we check how often it gets called after being 'hidden'
    @app.post(BATCH, deprecated=True)
    def batch_ner(
        self,
        docs: list[WebDocument],
        request: Request,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    ) -> JSONResponse:
        """Deprecated endpoint: use ner_and_linking.

        This endpoint will be removed in future.

        Behaves the same as ner_and_linking, but now we have the
        ner_only and linking_only endpoints, the naming here had the
        potential to be confusing.

        In addition, we refactored ner_and_linking so that we no longer
        need separate 'batch' and single document endpoints, now
        ner_and_linking handles both.
        """

        id_log_prefix = get_id_log_prefix_if_available(request)
        logger.info(id_log_prefix + "Request to kazu/batch endpoint")
        logger.info(id_log_prefix + "Documents sent: %s", len(docs))
        result = self.pipeline([doc.to_kazu_document() for doc in docs])
        return JSONResponse(content=[res.to_dict() for res in result])

    @app.post(LS_ANNOTATIONS)
    def ls_annotations(self, doc: WebDocument, request: Request) -> JSONResponse:
        """Provide LabelStudio annotations from the kazu results on the given document.

        This is not expected to be called by ordinary users. This endpoint was added in
        order to support the Kazu frontend, which makes uses of LabelStudio's frontend
        code, so relies on this endpoint.
        """

        id_log_prefix = get_id_log_prefix_if_available(request)
        log_request_to_path_with_prefix(request, log_prefix=id_log_prefix)
        logger.info(id_log_prefix + "Document: %s", doc)
        result = self.pipeline([doc.to_kazu_document()])[0]
        ls_view, ls_tasks = self.ls_web_utils.kazu_doc_to_ls(result)
        return JSONResponse(
            content={"ls_view": ls_view, "ls_tasks": ls_tasks, "doc": result.to_dict()}
        )

    # Note: this needs to be defined last so that we don't try and
    # interpret the ls-annotations or batch API calls as step groups.
    @app.post(STEP_GROUP_WITH_VAR)
    def step_group(
        self,
        step_group: str,
        request: Request,
        token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
        doc_collection: DocumentCollection = Body(
            **{_OPENAPI_EXAMPLES_FIELD: document_collection_examples}  # type:ignore[arg-type]
            # type ignore as above
        ),
    ) -> JSONResponse:
        """Run the pipeline with a specific step group.

        This will run only a pre-defined subset of steps. Call the step_groups endpoint
        to find the configured set of step_groups for this API deployment, and the steps
        they include.

        If you are interested in running a different set of steps, you can use the
        custom_pipeline_steps endpoint.
        """
        return self.base_pipeline_request(
            doc_collection=doc_collection,
            request=request,
            step_group=step_group,
        )


@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="../conf", config_name="config")
def start(cfg: DictConfig) -> None:
    """Deploy the web app to Ray Serve.

    :param cfg: DictConfig from Hydra
    :return: None
    """
    # Connect to the running Ray cluster, or run as single node
    call(cfg.ray.init)
    call(cfg.ray.serve)

    serve.run(KazuWebAPI.bind(cfg))


def stop():
    if ray.is_initialized():
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    start()

    # for compatibility - list_deployments was removed in 2.8,
    # 'status' play a similar role in 2.7 onwards
    deployment_status_method_name = "status" if hasattr(serve, "status") else "list_deployments"
    while True:
        deployment_status_method = getattr(serve, deployment_status_method_name)
        logger.info(deployment_status_method())
        time.sleep(600)
