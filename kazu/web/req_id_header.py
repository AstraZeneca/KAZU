"""Modified from 'CustomHeaderMiddleware' in the `Starlette Middleware docs
<https://www.starlette.io/middleware/#basehttpmiddleware>`_.

Licensed under BSD 3-Clause along with the rest of Starlette.

Copyright © 2018, `Encode OSS Ltd <https://www.encode.io/>`_.

.. raw:: html

    <details>
    <summary>Full License</summary>

Copyright © 2018, `Encode OSS Ltd <https://www.encode.io/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

.. raw:: html

    </details>
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from kazu.web.server import get_request_id

logger = logging.getLogger("ray")


class AddRequestIdMiddleware(BaseHTTPMiddleware):
    """A middleware that puts a request-id from the request's header onto the response.

    This was written to be used in conjunction with
    :class:`.JWTAuthenticationBackend`, which adds a request id header
    to the request as part of authentication.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        req_id = get_request_id(request)
        response = await call_next(request)
        if req_id is not None:
            response.headers["X-request-id"] = req_id

        logger.info("ID: %s Response Code: %s", req_id, response.status_code)
        return response
