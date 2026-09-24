"""Fixed-window, per-client rate limiter backed by Redis (shared across workers).

Each limiter has its own key namespace, so /chat and /analyze never share a bucket.
Fails open if Redis is unreachable: availability beats strictness here.
"""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    return _redis


def rate_limiter(name: str, limit: int, window_s: int):
    async def dependency(request: Request) -> None:
        # Keyed by peer IP; behind a proxy, configure uvicorn --proxy-headers so this is the real client.
        client = request.client.host if request.client else "unknown"
        key = f"ratelimit:{name}:{client}"
        try:
            r = get_redis()
            count = await r.incr(key)
            if count == 1:
                await r.expire(key, window_s)
            ttl = await r.ttl(key) if count > limit else 0
        except Exception:
            logger.exception("Rate limiter unavailable; allowing request")
            return
        if count > limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(max(ttl, 1))},
            )

    return dependency
