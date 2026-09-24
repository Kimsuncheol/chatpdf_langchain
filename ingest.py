"""Chunk, embed (batches of 20) and store pages in Chroma; resumable per page."""
from __future__ import annotations

import inspect
import logging
from typing import Awaitable, Callable, Iterable, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_parser import PageContent
from vision import page_visual_descriptions

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 20
EMBEDDING_MODEL = "nomic-embed-text"

# (doc_id, processingStage, progressPercent) -- wire the real Firestore admin helper here.
StatusUpdater = Callable[[str, str, int], Union[None, Awaitable[None]]]
Describer = Callable[[PageContent], Awaitable[str]]


def make_vectorstore(persist_directory: str, collection_name: str = "chatpdf") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=persist_directory,
    )


def _completed_pages(vectorstore: Chroma, doc_id: str) -> set[int]:
    """Pages whose chunks are all stored (stored count >= recorded total_chunks)."""
    metas = vectorstore.get(where={"docId": doc_id}, include=["metadatas"])["metadatas"]
    stored: dict[int, int] = {}
    totals: dict[int, int] = {}
    for m in metas:
        p = m["page_number"]
        stored[p] = stored.get(p, 0) + 1
        totals[p] = m["total_chunks"]
    return {p for p, n in stored.items() if n >= totals[p]}


async def embed_and_store_document(
    doc_id: str,
    pages: Iterable[PageContent],
    total_pages: int,
    vectorstore: Chroma,
    update_status: StatusUpdater,
    describe: Describer = page_visual_descriptions,
) -> None:
    """Embed every page not already in the store, then mark the doc done.

    Resume: pages with a complete set of chunks for (docId, page_number) are
    skipped before any Qwen-VL or embedding call. A page interrupted mid-way is
    re-embedded; chunk ids are deterministic so the upsert overwrites cleanly.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    done = _completed_pages(vectorstore, doc_id)

    for page in pages:
        if page.page_number in done:
            continue
        visual = await describe(page)
        full_text = "\n\n".join(filter(None, [page.text, visual]))
        texts = splitter.split_text(full_text)
        chunks = [
            Document(
                page_content=t,
                metadata={
                    "docId": doc_id,
                    "page_number": page.page_number,
                    "hasVisualContent": bool(visual),
                    "total_chunks": len(texts),
                },
            )
            for t in texts
        ]
        ids = [f"{doc_id}:{page.page_number}:{i}" for i in range(len(chunks))]
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            await vectorstore.aadd_documents(
                chunks[i : i + EMBED_BATCH_SIZE], ids=ids[i : i + EMBED_BATCH_SIZE]
            )

    result = update_status(doc_id, "done", 100)
    if inspect.isawaitable(result):
        await result
    logger.info("Embedded document %s (%d pages)", doc_id, total_pages)
