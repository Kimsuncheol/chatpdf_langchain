"""Describe charts/tables/diagrams on a PDF page with Qwen-VL via Ollama."""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from ollama import AsyncClient

from pdf_parser import PageContent

logger = logging.getLogger(__name__)

MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
TIMEOUT_S = 30
RETRY_DELAY_S = 2

SYSTEM_PROMPT = (
    "You are analyzing one page of a technical document. Focus only on tables, "
    "charts, graphs, and diagrams. Do not re-describe plain paragraph text."
)
USER_PROMPT = (
    "Page text extracted so far: {page_text}\n\n"
    "Describe any tables, charts, or diagrams on this page in detail, "
    "including all numeric values you can read."
)

# Host comes from OLLAMA_HOST (default http://127.0.0.1:11434); Ollama Cloud
# works by setting it plus an Authorization header via client kwargs.
_client: AsyncClient | None = None


def _get_client() -> AsyncClient:
    global _client
    if _client is None:
        _client = AsyncClient()
    return _client


async def _chat(image_path: str, page_text: str) -> str:
    resp = await _get_client().chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(page_text=page_text),
                "images": [image_path],
            },
        ],
    )
    return (resp.message.content or "").strip()


async def _chat_with_retry(image_path: str, page_text: str) -> str:
    try:
        return await _chat(image_path, page_text)
    except (ConnectionError, httpx.ConnectError) as e:
        logger.warning("Ollama connection failed (%s); retrying once", e)
        await asyncio.sleep(RETRY_DELAY_S)
        return await _chat(image_path, page_text)


async def describe_page_visuals(image_path: str, page_text: str) -> str:
    """Return a text description of the page's visuals, or "" on timeout.

    The 30s budget covers the whole call including the one connection retry.
    A second connection failure (or any other Ollama error) propagates.
    """
    try:
        return await asyncio.wait_for(_chat_with_retry(image_path, page_text), TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("Qwen-VL timed out after %ss for %s; skipping", TIMEOUT_S, image_path)
        return ""


async def enrich_page(page: PageContent) -> str:
    """Page text merged with visual descriptions; skips the model when has_images is False."""
    if not page.has_images:
        return page.text
    descriptions = [d for p in page.image_paths if (d := await describe_page_visuals(p, page.text))]
    return "\n\n".join([page.text, *descriptions])
