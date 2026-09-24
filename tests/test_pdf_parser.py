import io
import os
import tracemalloc

import pymupdf
import pytest
from PIL import Image

from pdf_parser import FileCheckpointStore, parse_pdf_pages

NUM_PAGES = 50
# Configurable via env; peak Python-heap allocation while parsing all pages.
PEAK_MB_LIMIT = float(os.environ.get("PDF_PARSE_PEAK_MB", "50"))


def _png_bytes(seed: int) -> bytes:
    img = Image.effect_noise((600, 400), 60 + seed).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "synthetic.pdf"
    doc = pymupdf.open()
    for i in range(NUM_PAGES):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} heading", fontsize=18)
        page.insert_textbox(pymupdf.Rect(72, 100, 520, 300), "lorem ipsum " * 60)
        if i % 3 == 0:  # every third page has an image
            page.insert_image(pymupdf.Rect(72, 320, 400, 560), stream=_png_bytes(i))
    doc.save(path)
    doc.close()
    return str(path)


def _run(pdf_path, tmp_path, doc_id="doc1", stop_after=None):
    store = FileCheckpointStore(tmp_path / "ckpt")
    pages = []
    gen = parse_pdf_pages(pdf_path, doc_id=doc_id, image_dir=str(tmp_path / "img"), checkpoint=store)
    for p in gen:
        pages.append(p)
        if stop_after and len(pages) == stop_after:
            gen.close()  # simulate a crash mid-document
            break
    return pages, store


def test_content_and_images(pdf_path, tmp_path):
    pages, _ = _run(pdf_path, tmp_path)
    assert [p.page_number for p in pages] == list(range(1, NUM_PAGES + 1))
    for p in pages:
        assert f"Page {p.page_number} heading" in p.text
        assert p.has_images == ((p.page_number - 1) % 3 == 0)
        assert bool(p.image_paths) == p.has_images
        assert all(os.path.exists(x) for x in p.image_paths)


def test_peak_memory_under_threshold(pdf_path, tmp_path):
    tracemalloc.start()
    try:
        count = sum(1 for _ in parse_pdf_pages(
            pdf_path, doc_id="mem", image_dir=str(tmp_path / "img"),
            checkpoint=FileCheckpointStore(tmp_path / "ckpt")))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == NUM_PAGES
    assert peak / 1e6 < PEAK_MB_LIMIT, f"peak {peak / 1e6:.1f}MB >= {PEAK_MB_LIMIT}MB"


def test_resume_after_crash(pdf_path, tmp_path):
    first, store = _run(pdf_path, tmp_path, stop_after=10)
    # Page 10 was handed out but not confirmed processed -> re-delivered on resume.
    assert store.get("doc1") == 9
    rest = list(parse_pdf_pages(pdf_path, doc_id="doc1", image_dir=str(tmp_path / "img"), checkpoint=store))
    assert [p.page_number for p in rest] == list(range(10, NUM_PAGES + 1))
    assert store.get("doc1") == 0  # cleared on completion
    # Idempotent: image paths are deterministic across runs.
    assert first[9].image_paths == rest[0].image_paths
