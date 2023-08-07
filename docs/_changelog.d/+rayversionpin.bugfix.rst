Added upper version limit for ray[serve] for the webserver dependencies.
In ray 2.5, HTTP Proxy Health checks were introduced which by default kill slow-deploying servers.
There are environments variables that can override this behaviour, but specifying them at the right time
is a pain in our setup, so until we've decided on the best way of handling this, just pin to a version of
ray that works.
