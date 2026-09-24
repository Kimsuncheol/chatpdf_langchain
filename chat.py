"""POST /chat: retrieval-grounded answers streamed as Server-Sent Events."""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import AsyncIterator, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ingest import make_vectorstore
from rate_limit import rate_limiter

logger = logging.getLogger(__name__)

CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:14b")
TOP_K = 6
MIN_RELEVANCE = 0.3
MAX_HISTORY_MESSAGES = 6
HISTORY_TOKEN_BUDGET = 3000
CHARS_PER_TOKEN = 2  # conservative (Korean text tokenizes densely)
NO_ANSWER = "I couldn't find relevant content in this document for that question."
CHAT_RATE_LIMIT = (30, 60)  # requests per window seconds, per client; separate bucket from /analyze

SYSTEM_PROMPT = (
    "You answer questions about a single document using only the excerpts provided. "
    "Each excerpt is labeled with its page number. If the excerpts do not contain the answer, say so. "
    "When you use information from an excerpt, cite it as (page N)."
)
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
    ]
)

router = APIRouter()


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    docId: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)


@lru_cache
def get_vectorstore():
    return make_vectorstore(os.environ.get("CHROMA_DIR", "./chroma_db"))


@lru_cache
def get_llm():
    # astream() streams token by token; ChatOllama has no separate streaming flag.
    return ChatOllama(model=CHAT_MODEL, temperature=0.2)


def trim_history(history: list[ChatTurn]) -> list[ChatTurn]:
    """Last MAX_HISTORY_MESSAGES messages, then drop oldest until under the token budget.

    The newest message is always kept (clipped if it alone exceeds the budget).
    """
    recent = history[-MAX_HISTORY_MESSAGES:]
    budget_chars = HISTORY_TOKEN_BUDGET * CHARS_PER_TOKEN
    kept: list[ChatTurn] = []
    used = 0
    for turn in reversed(recent):
        remaining = budget_chars - used
        if remaining <= 0:
            break
        if len(turn.content) > remaining:
            if kept:
                break
            turn = ChatTurn(role=turn.role, content=turn.content[-remaining:])
        kept.append(turn)
        used += len(turn.content)
    return kept[::-1]


def build_messages(question: str, docs, history: list[ChatTurn]) -> list[BaseMessage]:
    context = "\n\n".join(f"[Page {d.metadata['page_number']}]\n{d.page_content}" for d in docs)
    hist = [
        (HumanMessage if t.role == "user" else AIMessage)(content=t.content) for t in trim_history(history)
    ]
    return PROMPT.format_messages(history=hist, context=context, question=question)


_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?。])\s+|\n+")


def cited_pages(answer: str, docs) -> list[int]:
    """Pages the answer references: explicit "page N" mentions, or a chunk that
    covers most of the words of some answer sentence."""
    explicit = {
        int(n)
        for m in re.finditer(r"(?:page|pages|p\.|페이지)\s*(\d+)|(\d+)\s*(?:페이지|쪽)", answer, re.I)
        for n in [m.group(1) or m.group(2)]
    }
    sentences = [
        [w for w in _WORD.findall(s.lower()) if len(w) >= 2] for s in _SENTENCE.split(answer) if s.strip()
    ]
    sentences = [s for s in sentences if len(s) >= 5]
    pages = set()
    for d in docs:
        page = d.metadata["page_number"]
        if page in explicit:
            pages.add(page)
            continue
        chunk_words = set(_WORD.findall(d.page_content.lower()))
        if any(sum(w in chunk_words for w in s) / len(s) >= 0.6 for s in sentences):
            pages.add(page)
    return sorted(pages)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream(req: ChatRequest, vectorstore, llm) -> AsyncIterator[str]:
    try:
        scored = await vectorstore.asimilarity_search_with_relevance_scores(
            req.question, k=TOP_K, filter={"docId": req.docId}
        )
    except Exception:
        logger.exception("Retrieval failed for docId=%s", req.docId)
        yield sse("error", {"message": "Something went wrong. Please try again."})
        return
    docs = [d for d, score in scored if score >= MIN_RELEVANCE]
    if not docs:
        yield sse("token", {"text": NO_ANSWER})
        yield sse("done", {"citedPages": []})
        return

    answer: list[str] = []
    try:
        async for chunk in llm.astream(build_messages(req.question, docs, req.history)):
            text = chunk.content
            if text:
                answer.append(text)
                yield sse("token", {"text": text})
    except Exception:
        logger.exception("LLM streaming failed for docId=%s", req.docId)
        yield sse("error", {"message": "Something went wrong. Please try again."})
        return
    yield sse("done", {"citedPages": cited_pages("".join(answer), docs)})


@router.post("/chat", dependencies=[Depends(rate_limiter("chat", *CHAT_RATE_LIMIT))])
async def chat(req: ChatRequest, vectorstore=Depends(get_vectorstore), llm=Depends(get_llm)):
    return StreamingResponse(
        _stream(req, vectorstore, llm),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
