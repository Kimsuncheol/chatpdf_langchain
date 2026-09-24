"""Firebase ID-token authentication dependency."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Header, HTTPException, Request
from firebase_admin import auth

logger = logging.getLogger(__name__)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


# Sync on purpose: FastAPI runs it in a threadpool, so verify_id_token's
# occasional certificate fetch never blocks the event loop.
def get_current_user(request: Request, authorization: Optional[str] = Header(default=None)) -> str:
    """Verify `Authorization: Bearer <Firebase ID token>` and return the uid.

    The uid is also stored on request.state for the rate limiter's key function.
    """
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Missing or malformed Authorization header")
    try:
        decoded = auth.verify_id_token(token.strip())
    except auth.CertificateFetchError:
        logger.exception("Could not fetch Firebase signing certificates")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    except (ValueError, auth.InvalidIdTokenError):  # includes expired tokens
        raise _unauthorized("Invalid or expired token")
    uid = decoded["uid"]
    request.state.uid = uid
    return uid
