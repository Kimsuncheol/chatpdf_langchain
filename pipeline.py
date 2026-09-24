"""Steps 1-3 end to end: download -> parse (+vision) -> chunk/embed/store."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Callable, Optional

import pymupdf

import firebase_io
from ingest import embed_and_store_document, make_vectorstore
from pdf_parser import FileCheckpointStore, parse_pdf_pages
from vision import page_visual_descriptions

logger = logging.getLogger(__name__)

Report = Callable[[str, int], None]  # (processingStage, progressPercent)


class PipelineError(Exception):
    """Carries a message that is safe to show to clients."""


def run_pipeline(doc_id: str, storage_path: str, report: Report, *, vectorstore=None, describe=page_visual_descriptions) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "doc.pdf")
        try:
            firebase_io.download_pdf(storage_path, pdf_path)
        except Exception as e:
            logger.exception("Download failed for %s", storage_path)
            raise PipelineError("Could not retrieve file from storage") from e
        report("uploaded", 5)

        try:
            with pymupdf.open(pdf_path) as d:
                total = d.page_count
        except Exception as e:
            logger.exception("Could not open PDF for %s", doc_id)
            raise PipelineError("PDF parsing failed: corrupted file") from e
        report("parsing", 10)

        # Parsing and embedding are pipelined page by page (memory-safe), so
        # "embedding" covers both; progress runs 10 -> 95 as pages complete.
        report("embedding", 10)
        pages = parse_pdf_pages(
            pdf_path,
            doc_id=doc_id,
            image_dir=os.path.join(tmp, "images"),
            checkpoint=FileCheckpointStore(os.path.join(tempfile.gettempdir(), "pdf_checkpoints")),
        )
        vs = vectorstore or make_vectorstore(os.environ.get("CHROMA_DIR", "./chroma_db"))
        asyncio.run(
            embed_and_store_document(
                doc_id, pages, total, vs, firebase_io.update_status, describe,
                on_page=lambda n: report("embedding", 10 + int(85 * n / max(total, 1))),
            )
        )
