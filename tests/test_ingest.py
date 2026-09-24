import asyncio

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from ingest import embed_and_store_document
from pdf_parser import PageContent


class CountingEmbeddings(DeterministicFakeEmbedding):
    size: int = 8
    batches: list = []

    def __init__(self):
        super().__init__(size=8)
        self.batches = []

    def embed_documents(self, texts):
        self.batches.append(len(texts))
        return super().embed_documents(texts)


@pytest.fixture
def store(tmp_path):
    emb = CountingEmbeddings()
    return Chroma(collection_name="test_col", embedding_function=emb, persist_directory=str(tmp_path)), emb


def make_pages(n, long_text=("word " * 400)):
    return [PageContent(i, f"page {i} " + long_text, i % 2 == 0, [f"{i}.png"] if i % 2 == 0 else []) for i in range(1, n + 1)]


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *a):
        self.calls.append(a)


def run(doc_id, pages, store, status, describe):
    asyncio.run(embed_and_store_document(doc_id, pages, len(pages), store, status, describe))


def test_metadata_batching_and_done(store):
    vs, emb = store
    status = Recorder()

    async def describe(page):
        return "Bar chart Q1: 120M" if page.has_images else ""

    run("d1", make_pages(4), vs, status, describe)
    got = vs.get(where={"docId": "d1"}, include=["metadatas", "documents"])
    assert got["metadatas"]
    for m in got["metadatas"]:
        assert set(m) >= {"docId", "page_number", "hasVisualContent"}
        assert m["hasVisualContent"] == (m["page_number"] % 2 == 0)
    assert any("Bar chart" in d for d in got["documents"])
    assert all(len(d) <= 800 for d in got["documents"])
    assert max(emb.batches) <= 20 and any(b > 1 for b in emb.batches)
    assert status.calls == [("d1", "done", 100)]


def test_docid_scoping(store):
    vs, _ = store

    async def describe(page):
        return ""

    run("a", make_pages(2), vs, Recorder(), describe)
    run("b", make_pages(2), vs, Recorder(), describe)
    hits = vs.similarity_search("page", k=50, filter={"docId": "a"})
    assert hits and all(h.metadata["docId"] == "a" for h in hits)


def test_resume_skips_embedded_and_redoes_partial(store):
    vs, emb = store
    described = []

    async def describe(page):
        described.append(page.page_number)
        return ""

    pages = make_pages(6)
    run("d", pages[:3], vs, Recorder(), describe)  # "crash" after page 3
    # simulate a half-written page 3: drop one of its chunks
    ids = vs.get(where={"$and": [{"docId": "d"}, {"page_number": 3}]})["ids"]
    vs.delete(ids=ids[-1:])
    described.clear()
    emb.batches.clear()

    status = Recorder()
    run("d", pages, vs, status, describe)
    assert described == [3, 4, 5, 6]  # 1-2 skipped, partial 3 redone
    n_page3 = len(vs.get(where={"$and": [{"docId": "d"}, {"page_number": 3}]})["ids"])
    assert n_page3 == len(ids)  # no duplicates, restored
    assert status.calls == [("d", "done", 100)]
