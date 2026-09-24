"""Memory-safe, resumable, page-by-page PDF parsing.

The document is opened once (PyMuPDF memory-maps / lazily reads pages) and each
page is converted on its own via ``pymupdf4llm.to_markdown(doc, pages=[n])``, so
only one page's worth of data is ever resident.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Protocol

import pymupdf
import pymupdf4llm

IMAGE_DPI = 150
MIN_IMAGE_SIDE_PT = 8  # ignore icons/rules smaller than this (in PDF points)


@dataclass
class PageContent:
    page_number: int  # 1-based
    text: str
    has_images: bool
    image_paths: list[str] = field(default_factory=list)


class CheckpointStore(Protocol):
    def get(self, doc_id: str) -> int: ...
    def set(self, doc_id: str, page_number: int) -> None: ...
    def clear(self, doc_id: str) -> None: ...


class FileCheckpointStore:
    """Local JSON checkpoint; writes are atomic (tmp file + os.replace)."""

    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        return self.dir / f"{_safe(doc_id)}.checkpoint.json"

    def get(self, doc_id: str) -> int:
        try:
            return int(json.loads(self._path(doc_id).read_text())["last_page"])
        except (FileNotFoundError, KeyError, ValueError):
            return 0  # missing or corrupt -> start over (safe: parsing is idempotent)

    def set(self, doc_id: str, page_number: int) -> None:
        path = self._path(doc_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_page": page_number}))
        os.replace(tmp, path)

    def clear(self, doc_id: str) -> None:
        self._path(doc_id).unlink(missing_ok=True)


class RedisCheckpointStore:
    """Pass a ``redis.Redis`` client."""

    def __init__(self, client, prefix: str = "pdf_parse:last_page:"):
        self.client, self.prefix = client, prefix

    def get(self, doc_id: str) -> int:
        v = self.client.get(self.prefix + doc_id)
        return int(v) if v is not None else 0

    def set(self, doc_id: str, page_number: int) -> None:
        self.client.set(self.prefix + doc_id, page_number)

    def clear(self, doc_id: str) -> None:
        self.client.delete(self.prefix + doc_id)


def _safe(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode()).hexdigest()[:32]


def _extract_page_images(page: pymupdf.Page, out_dir: Path, page_number: int) -> list[str]:
    """Render each image's on-page region at IMAGE_DPI to a PNG. Deterministic names."""
    paths: list[str] = []
    # get_image_info() returns bboxes without decoding image data.
    for idx, info in enumerate(page.get_image_info()):
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if rect.is_empty or min(rect.width, rect.height) < MIN_IMAGE_SIDE_PT:
            continue
        path = out_dir / f"p{page_number:05d}_i{idx:03d}.png"
        pix = page.get_pixmap(dpi=IMAGE_DPI, clip=rect, alpha=False)
        try:
            tmp = path.with_suffix(".tmp.png")
            pix.save(tmp)
            os.replace(tmp, path)  # never leave a half-written image
        finally:
            pix = None
        paths.append(str(path))
    return paths


def parse_pdf_pages(
    file_path: str,
    *,
    doc_id: Optional[str] = None,
    image_dir: Optional[str] = None,
    checkpoint: Optional[CheckpointStore] = None,
) -> Generator[PageContent, None, None]:
    """Lazily yield one PageContent per page, resuming after the last checkpoint.

    The checkpoint for page N is committed only after the consumer has finished
    with it (i.e. when it requests page N+1 or exhausts the generator), giving
    at-least-once delivery: a crash may re-yield the page in flight, never skip it.
    Image files are overwritten deterministically, so re-processing is idempotent.

    ``doc_id`` defaults to a hash of the absolute path. ``checkpoint`` defaults to
    a file store in the system temp dir. The checkpoint is cleared on completion.
    """
    doc_id = doc_id or hashlib.sha256(os.path.abspath(file_path).encode()).hexdigest()
    checkpoint = checkpoint or FileCheckpointStore(Path(tempfile.gettempdir()) / "pdf_checkpoints")
    out_root = Path(image_dir) if image_dir else Path(tempfile.gettempdir()) / "pdf_images"
    out_dir = out_root / _safe(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = checkpoint.get(doc_id)  # last completed 1-based page
    doc = pymupdf.open(file_path)
    try:
        total = doc.page_count
        for page_number in range(start + 1, total + 1):
            page = doc.load_page(page_number - 1)
            try:
                chunks = pymupdf4llm.to_markdown(
                    doc, pages=[page_number - 1], page_chunks=True,
                    write_images=False, show_progress=False,
                )
                text = chunks[0]["text"] if chunks else ""
                chunks = None
                image_paths = _extract_page_images(page, out_dir, page_number)
                content = PageContent(
                    page_number=page_number,
                    text=text,
                    has_images=bool(image_paths),
                    image_paths=image_paths,
                )
            finally:
                page = None  # drop the page reference before yielding
            yield content
            content = None
            checkpoint.set(doc_id, page_number)
        checkpoint.clear(doc_id)
    finally:
        doc.close()
