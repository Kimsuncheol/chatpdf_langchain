import json

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import chat
import main
from chat import ChatTurn, cited_pages, trim_history


class FakeStore:
    def __init__(self, scored):
        self.scored, self.kwargs = scored, None

    async def asimilarity_search_with_relevance_scores(self, q, **kw):
        self.kwargs = kw
        return self.scored


class FakeLLM:
    def __init__(self, tokens):
        self.tokens, self.messages = tokens, None

    async def astream(self, messages):
        self.messages = messages
        for t in self.tokens:
            yield AIMessageChunk(content=t)


def doc(page, text):
    return Document(page_content=text, metadata={"docId": "d", "page_number": page})


def override(store, llm):
    main.app.dependency_overrides[chat.get_vectorstore] = lambda: store
    main.app.dependency_overrides[chat.get_llm] = lambda: llm


def events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        ev, data = block.split("\n")
        out.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


BODY = {"docId": "d", "question": "What is the mixing ratio?", "history": []}


def test_streams_tokens_and_cited_pages(client):
    chunk12 = "The mixing ratio of the solvent to the resin is three to one by volume."
    store = FakeStore([(doc(12, chunk12), 0.8), (doc(13, "Unrelated safety notes."), 0.5)])
    llm = FakeLLM(["The mixing ratio ", "is three to one by volume for the resin."])
    override(store, llm)
    r = client.post("/chat", json=BODY)
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    assert ev[0] == ("token", {"text": "The mixing ratio "})
    assert ev[-1] == ("done", {"citedPages": [12]})
    assert store.kwargs == {"k": 6, "filter": {"docId": "d"}}
    prompt = "\n".join(m.content for m in llm.messages)
    assert "[Page 12]" in prompt and "[Page 13]" in prompt


def test_below_threshold_skips_llm(client):
    llm = FakeLLM(["should not run"])
    override(FakeStore([(doc(1, "x"), 0.29)]), llm)
    ev = events(client.post("/chat", json=BODY))
    assert ev == [("token", {"text": chat.NO_ANSWER}), ("done", {"citedPages": []})]
    assert llm.messages is None


def test_history_trimmed_before_llm(client):
    llm = FakeLLM(["ok"])
    override(FakeStore([(doc(1, "x"), 0.9)]), llm)
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    client.post("/chat", json={**BODY, "history": hist})
    sent = [m.content for m in llm.messages[1:-1]]
    assert sent == [f"m{i}" for i in range(4, 10)]


def test_trim_history_drops_oldest_by_token_budget():
    big = "x" * 5000  # 2500 tokens at 2 chars/token
    h = [ChatTurn(role="user", content=big), ChatTurn(role="assistant", content=big), ChatTurn(role="user", content="newest")]
    out = trim_history(h)
    assert [t.content for t in out] == [big, "newest"] or out[-1].content == "newest"
    assert sum(len(t.content) for t in out) <= 6000
    assert out[-1].content == "newest"
    assert len(out) == 2  # oldest big turn dropped
    # a single oversized newest message is clipped, not dropped
    out = trim_history([ChatTurn(role="user", content="y" * 8000)])
    assert len(out) == 1 and len(out[0].content) == 6000


def test_cited_pages_explicit_reference():
    assert cited_pages("See (page 7) and 9페이지.", [doc(7, "a"), doc(8, "b"), doc(9, "c")]) == [7, 9]


def test_cosine_relevance_with_real_chroma(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    vs = Chroma(collection_name="cos_col", embedding_function=DeterministicFakeEmbedding(size=16),
                persist_directory=str(tmp_path), collection_configuration={"hnsw": {"space": "cosine"}})
    vs.add_documents([Document(page_content="hello world", metadata={"docId": "d", "page_number": 1})], ids=["1"])
    (d, score), = vs.similarity_search_with_relevance_scores("hello world", k=1, filter={"docId": "d"})
    assert score > 0.99
