"""slowapi limiter keyed by authenticated uid (never by IP)."""
from __future__ import annotations

import math
import os
import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

ANALYZE_LIMIT = "5/hour"
CHAT_LIMIT = "30/minute"


def uid_key(request: Request) -> str:
    # get_current_user runs before the limiter and sets this. slowapi silently skips
    # limits with an empty key, so refuse outright rather than ever falling back to IP.
    uid = getattr(request.state, "uid", None)
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication required")
    return uid


limiter = Limiter(
    key_func=uid_key,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", os.environ.get("REDIS_URL", "redis://localhost:6379/1")),
    swallow_errors=True,  # storage outage: fail open, slowapi logs the error
)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    item, args = request.state.view_rate_limit
    reset_at, _ = limiter.limiter.get_window_stats(item, *args)
    retry = max(1, math.ceil(reset_at - time.time()))
    return JSONResponse(
        {"error": "rate_limited", "retryAfterSeconds": retry},
        status_code=429,
        headers={"Retry-After": str(retry)},
    )
