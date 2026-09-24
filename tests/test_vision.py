import asyncio
from types import SimpleNamespace

import pytest

import vision
from pdf_parser import PageContent


class FakeClient:
    def __init__(self, behaviors):
        self.behaviors, self.calls = list(behaviors), []

    async def chat(self, **kw):
        self.calls.append(kw)
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        if callable(b):
            return await b()
        return SimpleNamespace(message=SimpleNamespace(content=b))


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(vision, "RETRY_DELAY_S", 0)


def use(monkeypatch, *behaviors):
    fake = FakeClient(behaviors)
    monkeypatch.setattr(vision, "_client", fake)
    return fake


def test_success_sends_prompt_and_image(monkeypatch):
    fake = use(monkeypatch, " Bar chart. Q1: 120M ")
    out = asyncio.run(vision.describe_page_visuals("p.png", "some text"))
    assert out == "Bar chart. Q1: 120M"
    call = fake.calls[0]
    assert call["model"] == "qwen2.5vl:7b"
    assert call["messages"][0]["role"] == "system"
    assert "Page text extracted so far: some text" in call["messages"][1]["content"]
    assert call["messages"][1]["images"] == ["p.png"]


def test_retries_once_on_connection_error(monkeypatch):
    fake = use(monkeypatch, ConnectionError("cold"), "ok")
    assert asyncio.run(vision.describe_page_visuals("p.png", "t")) == "ok"
    assert len(fake.calls) == 2


def test_second_connection_error_propagates(monkeypatch):
    use(monkeypatch, ConnectionError("a"), ConnectionError("b"))
    with pytest.raises(ConnectionError):
        asyncio.run(vision.describe_page_visuals("p.png", "t"))


def test_timeout_returns_empty_and_warns(monkeypatch, caplog):
    async def hang():
        await asyncio.sleep(10)

    use(monkeypatch, hang)
    monkeypatch.setattr(vision, "TIMEOUT_S", 0.05)
    assert asyncio.run(vision.describe_page_visuals("p.png", "t")) == ""
    assert "timed out" in caplog.text


def test_enrich_skips_model_without_images(monkeypatch):
    fake = use(monkeypatch)
    page = PageContent(1, "plain", False, [])
    assert asyncio.run(vision.enrich_page(page)) == "plain"
    assert fake.calls == []


def test_enrich_merges_descriptions(monkeypatch):
    use(monkeypatch, "Chart A", "")
    page = PageContent(1, "txt", True, ["a.png", "b.png"])
    assert asyncio.run(vision.enrich_page(page)) == "txt\n\nChart A"
