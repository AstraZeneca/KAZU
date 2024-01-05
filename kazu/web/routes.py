KAZU = "kazu"
API = "api"
NO_AUTH_ENDPOINTS = [
    "/",
    f"/{API}",
    f"/{API}/",
    f"/{API}/docs",
    f"/{API}/openapi.json",
    f"/{API}/{KAZU}/ls-annotations",
]
NO_AUTH_DIRS = [
    # nothing under /ui requires auth
    # the main page is /ui/index, but
    # it needs to be able to load static resources
    # which are places under /ui as well.
    "/ui"
]
