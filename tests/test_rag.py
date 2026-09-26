"""Tests for the RAG pipeline that run fully offline (no API key, no downloads)."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import anthropic
import joblib
import pytest

from docqa import ask, config, embed, generate, ingest
from docqa.chunk import chunk_text, load_and_chunk
from docqa.embed import TransformerEmbedder
from docqa.generate import extractive_answer, has_credentials
from docqa.pipeline import answer
from docqa.store import VectorStore, build_index

LEAKAGE_Q = "What is data leakage and how do you detect it?"


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture
def tfidf_store():
    return build_index(save=False, backend="tfidf")


def test_chunking_overlap():
    text = " ".join(str(i) for i in range(400))  # one long "paragraph"
    chunks = chunk_text(text, "nums.txt")
    assert len(chunks) > 1
    assert all(c.source == "nums.txt" for c in chunks)
    # consecutive chunks overlap (windowing), so total words exceed the original
    assert sum(len(c.text.split()) for c in chunks) > 400


def test_load_and_chunk_reads_docs():
    chunks = load_and_chunk()
    assert len(chunks) >= 4
    sources = {c.source for c in chunks}
    assert "data_leakage.md" in sources


def test_retrieval_finds_relevant_doc():
    store = build_index(save=False, backend="tfidf")
    hits = store.search("What is data leakage and how to detect it?", k=3)
    assert hits[0].score > 0
    # the top hit should come from the leakage document
    assert hits[0].chunk.source == "data_leakage.md"


def test_retrieval_transfer_learning():
    store = build_index(save=False, backend="tfidf")
    hits = store.search("freezing a pretrained backbone for a new task", k=3)
    assert hits[0].chunk.source == "transfer_learning.md"


def test_extractive_answer_shape():
    store = build_index(save=False, backend="tfidf")
    hits = store.search("evaluation metrics", k=3)
    result = extractive_answer("evaluation metrics", hits)
    assert result["mode"] == "extractive"
    assert result["answer"]
    assert result["sources"][0]["n"] == 1


def test_save_and_load(tmp_path):
    store = build_index(save=False, backend="tfidf")
    p = tmp_path / "index.joblib"
    store.save(p)
    loaded = VectorStore.load(p)
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.search("RAG", k=1)[0].score >= 0


def test_has_credentials_is_bool():
    assert isinstance(has_credentials(), bool)


def test_transformer_embedder_pickles_without_model(tmp_path):
    # A loaded model/tokenizer (here: unpicklable stand-ins) must not end up in
    # index.joblib; only the model name is persisted and weights reload lazily.
    emb = TransformerEmbedder("some/model")
    emb._model, emb._tok = (lambda: None), (lambda: None)
    path = tmp_path / "emb.joblib"
    joblib.dump(emb, path)
    loaded = joblib.load(path)
    assert loaded.model_name == "some/model"
    assert loaded._model is None and loaded._tok is None


def test_off_topic_question_finds_nothing(tfidf_store, fake_key, monkeypatch):
    def no_claude(*args, **kwargs):
        raise AssertionError("Claude should not be called with no relevant passages")

    monkeypatch.setattr(generate, "synthesize", no_claude)
    result = answer("Who won the 2022 World Cup?", store=tfidf_store)
    assert result["sources"] == []
    assert "couldn't find" in result["answer"]


def test_on_topic_question_passes_threshold(tfidf_store):
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["sources"][0]["source"] == "data_leakage.md"


def test_claude_api_failure_is_reported(tfidf_store, fake_key, monkeypatch):
    def fail(*args, **kwargs):
        # the base class of every SDK error; constructing a specific one needs the SDK's
        # HTTP client types, which differ between anthropic 0.x (httpx) and 1.x (httpx2)
        raise anthropic.AnthropicError("connection failed")

    monkeypatch.setattr(generate, "synthesize", fail)
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["mode"] == "extractive"
    assert "AnthropicError: connection failed" in result["note"]
    # the key *is* set, so don't tell the user to set it
    assert "Set ANTHROPIC_API_KEY" not in result["answer"]


def test_empty_claude_response_is_reported(tfidf_store, fake_key, monkeypatch):
    empty = SimpleNamespace(content=[], stop_reason="refusal")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: empty))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["mode"] == "extractive"
    assert "stop_reason=refusal" in result["note"]


def test_non_api_errors_propagate(tfidf_store, fake_key, monkeypatch):
    def bug(*args, **kwargs):
        raise ValueError("a real bug")

    monkeypatch.setattr(generate, "synthesize", bug)
    with pytest.raises(ValueError):
        answer(LEAKAGE_Q, store=tfidf_store)


def test_ask_cli_prints_note(monkeypatch, capsys):
    result = {"question": "q", "answer": "a", "sources": [], "mode": "extractive",
              "note": "Claude call failed (AuthenticationError)"}
    monkeypatch.setattr(ask, "answer", lambda question, k=None: result)
    monkeypatch.setattr("sys.argv", ["ask", "q"])
    ask.main()
    assert "AuthenticationError" in capsys.readouterr().err


# A small labeled set: each question should retrieve its source document first.
RETRIEVAL_EVAL = {
    "data_leakage.md": [
        LEAKAGE_Q, "how to detect leakage", "label permutation test",
    ],
    "transfer_learning.md": [
        "When does transfer learning help?", "freezing a pretrained backbone for a new task",
        "fine-tuning vs feature extraction",
    ],
    "evaluation_metrics.md": [
        "Why is accuracy alone misleading?", "What is precision and recall?",
        "F1 score on imbalanced classes",
    ],
    "rag_systems.md": [
        "How does RAG reduce hallucination?", "what is a vector store",
        "retrieval augmented generation citations",
    ],
}


def test_retrieval_eval_recall_at_1(tfidf_store):
    threshold = config.min_score("tfidf")
    misses = []
    for doc, questions in RETRIEVAL_EVAL.items():
        for q in questions:
            top = tfidf_store.search(q, k=1)[0]
            if top.chunk.source != doc or top.score < threshold:
                misses.append(f"{q!r} -> {top.chunk.source} ({top.score:.3f}), want {doc}")
    assert not misses, "\n".join(misses)


def _fake_claude(monkeypatch, text, stop_reason="end_turn"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                               stop_reason=stop_reason)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
    return calls


def test_claude_answer_path(tfidf_store, fake_key, monkeypatch):
    calls = _fake_claude(monkeypatch, "Leakage is ... [1]")
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["mode"] == "claude"
    assert result["answer"] == "Leakage is ... [1]"
    assert "note" not in result
    (req,) = calls
    assert req["model"] == config.GEN_MODEL
    assert req["max_tokens"] == config.MAX_ANSWER_TOKENS
    assert req["system"] == generate.SYSTEM_PROMPT
    prompt = req["messages"][0]["content"]
    assert "[1] (source: data_leakage.md)" in prompt and LEAKAGE_Q in prompt


def test_truncated_claude_answer_is_noted(tfidf_store, fake_key, monkeypatch):
    _fake_claude(monkeypatch, "Leakage is", stop_reason="max_tokens")
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["mode"] == "claude"
    assert "RAG_MAX_TOKENS" in result["note"]


def test_tokenizer_name(monkeypatch):
    monkeypatch.setattr(config, "EMBED_TOKENIZER_NAME", None)
    assert embed.tokenizer_name("google/bert_uncased_L-4_H-256_A-4") == "bert-base-uncased"
    assert embed.tokenizer_name("some/other-model") == "some/other-model"
    monkeypatch.setattr(config, "EMBED_TOKENIZER_NAME", "my/tokenizer")
    assert embed.tokenizer_name("some/other-model") == "my/tokenizer"


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    try:
        importlib.reload(config)
        assert config.DOCS_DIR == tmp_path / "docs"
        assert config.INDEX_PATH == tmp_path / "index.joblib"
    finally:
        monkeypatch.delenv("RAG_DATA_DIR")
        importlib.reload(config)


def test_ingest_reports_docs_dir_used(monkeypatch, tmp_path, capsys):
    docs = tmp_path / "mydocs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nSome text about apples.", encoding="utf-8")
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "index.joblib")
    monkeypatch.setattr("sys.argv", ["ingest", "--docs", str(docs), "--backend", "tfidf"])
    ingest.main()
    assert str(docs) in capsys.readouterr().out
    assert (tmp_path / "index.joblib").exists()
