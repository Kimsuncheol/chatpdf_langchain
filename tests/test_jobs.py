import shutil

import pymupdf
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

import firebase_io
import main
import pipeline
import tasks
from pipeline import PipelineError


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    calls = []
    monkeypatch.setattr(firebase_io, "update_status", lambda *a, **k: calls.append((a, k)))
    return calls


def test_analyze_enqueues_and_returns_202(client, monkeypatch):
    seen = {}

    class T:
        id = "job-1"

    monkeypatch.setattr(main.process_document_task, "delay", lambda d, s: seen.update(a=(d, s)) or T())
    r = client.post("/analyze", json={"docId": "abc123", "storagePath": "uploads/u1/abc123.pdf"})
    assert r.status_code == 202
    assert r.json() == {"jobId": "job-1", "status": "queued"}
    assert seen["a"] == ("abc123", "uploads/u1/abc123.pdf")


def test_analyze_validates_and_handles_broker_down(client, monkeypatch):
    assert client.post("/analyze", json={"docId": "", "storagePath": "x"}).status_code == 422

    def boom(*a):
        raise ConnectionError("redis down")

    monkeypatch.setattr(main.process_document_task, "delay", boom)
    assert client.post("/analyze", json={"docId": "a", "storagePath": "b"}).status_code == 503


def test_failure_sets_error_and_does_not_leak(client, monkeypatch, eager, caplog):
    def bad(*a, **k):
        raise ValueError("secret /internal/path")

    monkeypatch.setattr(tasks, "run_pipeline", bad)
    res = tasks.process_document_task.delay("d1", "p.pdf")
    assert res.state == "FAILURE"
    assert eager[-1] == (("d1", "error", 0), {"error": "Processing failed"})
    assert "secret /internal/path" in caplog.text  # trace logged server-side
    body = client.get(f"/jobs/{res.id}/status").json()
    assert body == {"state": "FAILURE", "error": "Processing failed"}


def test_failure_shows_safe_pipeline_message(client, monkeypatch):
    def bad(*a, **k):
        raise PipelineError("PDF parsing failed: corrupted file")

    monkeypatch.setattr(tasks, "run_pipeline", bad)
    res = tasks.process_document_task.delay("d1", "p.pdf")
    assert client.get(f"/jobs/{res.id}/status").json() == {
        "state": "FAILURE", "error": "PDF parsing failed: corrupted file"}


def test_status_started_and_pending(client, monkeypatch):
    class R:
        state, info = "STARTED", {"progressPercent": 42}

    monkeypatch.setattr(main, "AsyncResult", lambda *a, **k: R())
    assert client.get("/jobs/x/status").json() == {"state": "STARTED", "progressPercent": 42}
    R.state = "PENDING"
    assert client.get("/jobs/x/status").json() == {"state": "PENDING"}


def test_full_pipeline_progress_and_done(client, monkeypatch, eager, tmp_path):
    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    for i in range(4):
        doc.new_page().insert_text((72, 72), f"hello page {i + 1}")
    doc.save(src)
    doc.close()
    monkeypatch.setattr(firebase_io, "download_pdf", lambda sp, dest: shutil.copy(src, dest))
    vs = Chroma(collection_name="job_col", embedding_function=DeterministicFakeEmbedding(size=8),
                persist_directory=str(tmp_path / "c"))
    monkeypatch.setattr(pipeline, "make_vectorstore", lambda *_: vs)

    res = tasks.process_document_task.delay("docX", "uploads/x.pdf")
    assert res.state == "SUCCESS"
    stages = [a[1] for a, _ in eager]
    assert stages[:3] == ["uploaded", "parsing", "embedding"]
    assert stages[-1] == "done" and eager[-1][0][2] == 100
    assert len(vs.get(where={"docId": "docX"})["ids"]) >= 4
    assert client.get(f"/jobs/{res.id}/status").json() == {"state": "SUCCESS", "progressPercent": 100}
