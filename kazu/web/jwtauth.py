"""Original source:

https://raw.githubusercontent.com/amitripshtos/starlette-jwt/master/starlette_jwt/middleware.py


All code in this file is originally licensed under the BSD 3-Clause License as part of starlette-jwt above.

Copyright (c) 2018, Amit Ripshtos All rights reserved.

.. raw:: html

    <details>
    <summary>Full License</summary>

Copyright (c) 2018, Amit Ripshtos
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
import uuid
from typing import Optional, Union

import jwt
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    BaseUser,
)
from starlette.requests import Request, HTTPConnection
from starlette.responses import JSONResponse

from kazu.web.routes import NO_AUTH_ENDPOINTS, NO_AUTH_DIRS

logger = logging.getLogger("ray")


class JWTUser(BaseUser):
    def __init__(self, username: str, token: str, payload: dict) -> None:
        self.username = username
        self.token = token
        self.payload = payload

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return self.username


class JWTAuthenticationBackend(AuthenticationBackend):
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        prefix: str = "Bearer",
        username_field: str = "username",
        audience: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> None:
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.prefix = prefix
        self.username_field = username_field
        self.audience = audience
        self.options = options or dict()

    @classmethod
    def get_token_from_header(cls, authorization: str, prefix: str) -> str:
        """Parses the Authorization header and returns only the token."""
        try:
            scheme, token = authorization.split()
        except ValueError:
            raise AuthenticationError("Could not separate Authorization scheme and token")
        if scheme.lower() != prefix.lower():
            raise AuthenticationError(f"Authorization scheme {scheme} is not supported")
        return token

    async def authenticate(
        self, request: HTTPConnection
    ) -> Union[None, tuple[AuthCredentials, BaseUser]]:
        req_id = str(uuid.uuid4())
        request.headers.__dict__["_list"].append(
            (
                "X-request-id".encode(),
                req_id.encode(),
            )
        )
        path: str = request.scope["raw_path"].decode()
        if path in NO_AUTH_ENDPOINTS or any(path.startswith(x) for x in NO_AUTH_DIRS):
            logger.info("ID: %s Request to %s, no authentication required", req_id, path)
            return None

        if "Authorization" not in request.headers:
            message = (
                f"ID: {req_id} No 'Authorization' header specified: please use a valid Bearer token"
            )
            logger.warn(message)
            raise AuthenticationError(message)
        auth = request.headers["Authorization"]
        token = self.get_token_from_header(authorization=auth, prefix=self.prefix)
        try:
            payload = jwt.decode(
                token,
                key=self.secret_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                options=self.options,
            )
        except jwt.InvalidTokenError as e:
            logger.warn("ID: %s %s", req_id, e)
            raise AuthenticationError(f"ID: {req_id} {e}")
        username = payload[self.username_field]
        logger.info("ID: %s Received request from %s", req_id, username)
        return AuthCredentials(["authenticated"]), JWTUser(
            username=username, token=token, payload=payload
        )


def on_auth_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=401)
