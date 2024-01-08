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

NO_AUTH_ENDPOINTS = [
    "/",
    f"/{API}",
    f"/{API}/",
    f"/{API}/{KAZU}/ls-annotations",
    f"/{API}/steps",
    f"/{API}/step_groups",
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
