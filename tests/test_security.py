import pytest

import chat
import main
from ratelimit import limiter
from test_chat import FakeLLM, FakeStore, override

CHAT = {"docId": "d", "question": "q", "history": []}
ANALYZE = {"docId": "a", "storagePath": "uploads/a.pdf"}


def as_user(uid):
    return {"Authorization": f"Bearer {uid}"}


@pytest.fixture(autouse=True)
def stub_queue(monkeypatch):
    monkeypatch.setattr(main.process_document_task, "delay", lambda *a: type("T", (), {"id": "job"})())
    override(FakeStore([]), FakeLLM([]))


def test_firebase_initialised_once_at_startup(client):
    client.get("/health")
    client.post("/analyze", json=ANALYZE)
    assert client.inits == [1]


def test_health_is_public_everything_else_requires_auth(client):
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200
    calls = [
        ("get", "/", None),
        ("post", "/analyze", ANALYZE),
        ("post", "/chat", CHAT),
        ("get", "/jobs/x/status", None),
    ]
    for method, path, body in calls:
        for headers in ({"Authorization": ""}, {"Authorization": "Bearer bad"}, {"Authorization": "Bearer expired"}, {"Authorization": "Basic abc"}):
            r = client.request(method, path, json=body, headers=headers)
            assert r.status_code == 401, (path, headers)
            assert r.headers["www-authenticate"] == "Bearer"


def test_valid_token_passes(client):
    assert client.get("/").status_code == 200


def test_analyze_limit_5_per_hour_per_uid(client):
    for _ in range(5):
        assert client.post("/analyze", json=ANALYZE).status_code == 202
    r = client.post("/analyze", json=ANALYZE)
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "rate_limited" and 0 < body["retryAfterSeconds"] <= 3600
    assert r.headers["retry-after"] == str(body["retryAfterSeconds"])
    # another user on the same IP is unaffected
    assert client.post("/analyze", json=ANALYZE, headers=as_user("user-b")).status_code == 202


def test_chat_limit_30_per_minute_and_buckets_are_separate(client):
    for _ in range(30):
        assert client.post("/chat", json=CHAT).status_code == 200
    r = client.post("/chat", json=CHAT)
    assert r.status_code == 429
    assert r.json()["error"] == "rate_limited" and r.json()["retryAfterSeconds"] <= 60
    assert client.post("/chat", json=CHAT, headers=as_user("user-b")).status_code == 200
    assert client.post("/analyze", json=ANALYZE).status_code == 202  # /analyze bucket untouched
