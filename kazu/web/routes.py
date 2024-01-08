from typing import TypedDict

KAZU = "kazu"
API = "api"


class _FastAPIDocsUrls(TypedDict):
    docs_url: str
    openapi_url: str
    redoc_url: str


API_DOCS_URLS: _FastAPIDocsUrls = {
    "docs_url": f"/{API}/docs",  # this is the Swagger page
    "openapi_url": f"/{API}/openapi.json",
    "redoc_url": f"/{API}/redoc",
}

_API_KAZU_PREFIX = f"/{API}/{KAZU}/"

STEPS = f"/{API}/steps"
STEP_GROUPS = f"/{API}/step_groups"
NER_AND_LINKING = f"{_API_KAZU_PREFIX}ner_and_linking"
CUSTOM_PIPELINE_STEPS = f"{_API_KAZU_PREFIX}custom_pipeline_steps"
NER_ONLY = f"{_API_KAZU_PREFIX}ner_only"
LINKING_ONLY = f"{_API_KAZU_PREFIX}linking_only"
BATCH = f"{_API_KAZU_PREFIX}batch"
LS_ANNOTATIONS = f"{_API_KAZU_PREFIX}ls-annotations"
STEP_GROUP_WITH_VAR = f"{_API_KAZU_PREFIX}{{step_group}}"

NO_AUTH_ENDPOINTS = [
    "/",
    f"/{API}",
    f"/{API}/",
    LS_ANNOTATIONS,
    STEPS,
    STEP_GROUPS,
]

# type ignore necessary because the way TypedDict currently works,
# one of the values could be something other than a string, as
# we could be using a subclass of _FastAPIDocsUrls which adds a field
# with some other type. See https://github.com/python/mypy/issues/7981
# for discussion and possible solution
NO_AUTH_ENDPOINTS.extend(API_DOCS_URLS.values())  # type: ignore[arg-type]

NO_AUTH_DIRS = [
    # nothing under /ui requires auth
    # the main page is /ui/index, but
    # it needs to be able to load static resources
    # which are places under /ui as well.
    "/ui"
]

AUTH_ENDPOINTS = [
    NER_AND_LINKING,
    CUSTOM_PIPELINE_STEPS,
    NER_ONLY,
    LINKING_ONLY,
    BATCH,
    STEP_GROUP_WITH_VAR,
    f"/{API}/{KAZU}",  # old legacy single-doc api
]
