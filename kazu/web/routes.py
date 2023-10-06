KAZU = "kazu"
NO_AUTH_ENDPOINTS = [
    "/api",
    "/api/",
    "/api/docs",
    "/api/openapi.json",
    "/api/kazu/ls-annotations",
]
NO_AUTH_DIRS = [
    # nothing under /ui requires auth
    # the main page is /ui/index, but
    # it needs to be able to load static resources
    # which are places under /ui as well.
    "/ui"
]
