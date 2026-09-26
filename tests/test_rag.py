"""Tests for the RAG pipeline. They run offline with no API key; the transformer
tests also need the `transformer` extra (and download the embedding model), and
are skipped without it.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from types import SimpleNamespace

import anthropic
import joblib
import numpy as np
import pytest

from docqa import ask, config, generate, ingest
from docqa.chunk import chunk_text, load_and_chunk
from docqa.embed import TransformerEmbedder
from docqa.generate import extractive_answer, has_credentials
from docqa.pipeline import answer
from docqa.store import VectorStore, build_index

NOTES = config.DATA_DIR / "ml_notes"  # small corpus: fast, and its answers are known
LEAKAGE_Q = "What is data leakage and how do you detect it?"


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture
def tfidf_store():
    return build_index(NOTES, backend="tfidf", save=False)


def test_chunking_overlap():
    text = " ".join(str(i) for i in range(400))  # one long "paragraph"
    chunks = chunk_text(text, "nums.txt")
    assert len(chunks) > 1
    assert all(c.source == "nums.txt" for c in chunks)
    # consecutive chunks overlap (windowing), so total words exceed the original
    assert sum(len(c.text.split()) for c in chunks) > 400


def test_load_and_chunk_reads_docs():
    chunks = load_and_chunk(NOTES)
    assert len(chunks) >= 4
    sources = {c.source for c in chunks}
    assert "data_leakage.md" in sources


def test_retrieval_finds_relevant_doc():
    store = build_index(NOTES, backend="tfidf", save=False)
    hits = store.search("What is data leakage and how to detect it?", k=3)
    assert hits[0].score > 0
    # the top hit should come from the leakage document
    assert hits[0].chunk.source == "data_leakage.md"


def test_retrieval_transfer_learning():
    store = build_index(NOTES, backend="tfidf", save=False)
    hits = store.search("freezing a pretrained backbone for a new task", k=3)
    assert hits[0].chunk.source == "transfer_learning.md"


def test_extractive_answer_shape():
    store = build_index(NOTES, backend="tfidf", save=False)
    hits = store.search("evaluation metrics", k=3)
    result = extractive_answer("evaluation metrics", hits)
    assert result["mode"] == "extractive"
    assert result["answer"]
    assert result["sources"][0]["n"] == 1


def test_save_and_load(tmp_path):
    store = build_index(NOTES, backend="tfidf", save=False)
    p = tmp_path / "index.joblib"
    store.save(p)
    loaded = VectorStore.load(p)
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.search("RAG", k=1)[0].score >= 0


def test_has_credentials_is_bool():
    assert isinstance(has_credentials(), bool)


def test_transformer_embedder_pickles_without_model(tmp_path):
    # A loaded model (here: an unpicklable stand-in) must not end up in
    # index.joblib; only the model name is persisted and weights reload lazily.
    emb = TransformerEmbedder("some/model")
    emb._model = lambda: None
    path = tmp_path / "emb.joblib"
    joblib.dump(emb, path)
    loaded = joblib.load(path)
    assert loaded.model_name == "some/model"
    assert loaded._model is None


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


# The same topics asked in words the docs mostly don't use; TF-IDF gets 9 of these.
PARAPHRASE_EVAL = {
    "data_leakage.md": [
        "My model scored brilliantly offline but flopped once deployed. Why?",
        "Should I normalize features before or after dividing train and test?",
        "copies of the same example ended up on both sides of my split",
    ],
    "transfer_learning.md": [
        "reuse an ImageNet network for my small image dataset",
        "I only have a few hundred labelled examples, can a pretrained model help?",
        "should I retrain all the layers or keep them fixed",
    ],
    "evaluation_metrics.md": [
        "my fraud model is 99% right but catches nothing",
        "how often are the positive predictions actually correct",
        "which measure should I report for a regression model",
    ],
    "rag_systems.md": [
        "how do you stop a chatbot from making things up",
        "grounding LLM responses in my company's documents",
        "keyword search versus neural embeddings for finding passages",
    ],
}


def _recall_misses(store, labeled: dict, threshold: float = 0.0) -> list[str]:
    misses = []
    for doc, questions in labeled.items():
        for q in questions:
            top = store.search(q, k=1)[0]
            if top.chunk.source != doc or top.score < threshold:
                misses.append(f"{q!r} -> {top.chunk.source} ({top.score:.3f}), want {doc}")
    return misses


def test_retrieval_eval_recall_at_1(tfidf_store):
    misses = _recall_misses(tfidf_store, RETRIEVAL_EVAL, config.min_score("tfidf"))
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


def test_search_encodes_query_as_a_query():
    # Some embedding models use different prompts for queries and documents.
    class Embedder:
        name = "fake"

        def encode(self, texts):
            raise AssertionError("queries must go through encode_queries")

        def encode_queries(self, texts):
            return np.array([[1.0, 0.0]])

    chunks = load_and_chunk(NOTES)[:2]
    store = VectorStore(Embedder(), np.array([[0.0, 1.0], [1.0, 0.0]]), chunks)
    assert store.search("q", k=1)[0].chunk is chunks[1]


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    try:
        importlib.reload(config)
        assert config.DOCS_DIR == tmp_path / "sklearn"
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


@pytest.fixture(scope="module")
def transformer_store():
    # Skipped unless the `transformer` extra is installed. CI's transformer job sets
    # RAG_REQUIRE_TRANSFORMER, so a broken install fails there instead of skipping.
    if not os.environ.get("RAG_REQUIRE_TRANSFORMER"):
        pytest.importorskip("sentence_transformers")
    return build_index(NOTES, backend="transformer", save=False)


def test_transformer_retrieval_eval(transformer_store):
    # One miss is tolerated: the chunker packs a whole section into one chunk, which
    # measurably helps on the 100-question scikit-learn eval (see README) but blurs
    # "duplicate leakage" into the other bullets of its section on these notes.
    misses = _recall_misses(transformer_store, RETRIEVAL_EVAL)
    misses += _recall_misses(transformer_store, PARAPHRASE_EVAL)
    assert len(misses) <= 1, "\n".join(misses)


def test_transformer_index_round_trip(transformer_store, tmp_path):
    path = transformer_store.save(tmp_path / "index.joblib")
    assert path.stat().st_size < 1_000_000  # vectors and chunks, not model weights
    loaded = VectorStore.load(path)
    assert loaded.search(LEAKAGE_Q, k=1)[0].chunk.source == "data_leakage.md"


# --- chunking, corpus conversion and the retrieval eval --------------------------------

def test_chunks_carry_their_section_path():
    text = "# Guide\n\nIntro.\n\n## Part A\n\nAbout A.\n\n### Detail\n\nDeep.\n\n## Part B\n\nAbout B."
    chunks = chunk_text(text, "g.md")
    assert [c.section for c in chunks] == [
        "Guide", "Guide > Part A", "Guide > Part A > Detail", "Guide > Part B"]
    assert chunks[2].text == "Guide > Part A > Detail. Deep."


def test_chunker_packs_short_paragraphs_and_keeps_code_whole():
    text = ("<!-- source comment -->\n\n# T\n\none two.\n\nthree four.\n\n"
            "```\ncode line\n\nmore code\n```")
    chunks = chunk_text(text, "t.md")
    assert len(chunks) == 1                     # one section, well under CHUNK_WORDS
    assert "source comment" not in chunks[0].text
    assert "```\ncode line\n\nmore code\n```" in chunks[0].text


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "fetch_sklearn_docs", config.REPO_ROOT / "scripts" / "fetch_sklearn_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rst_conversion():
    rst = """.. _label:

=====
Title
=====

.. currentmodule:: sklearn.svm

Use :class:`~sklearn.svm.SVC` or :func:`train_test_split`, see
:ref:`the guide <grid_search>` and `docs <https://x.org>`_; math :math:`x^2`.

Section
=======

.. note::

   Careful with ``C``.

Example::

    >>> clf.fit(X, y)

Sub
---

.. image:: foo.png
   :width: 10px

Text.
"""
    md = _load_converter().convert(rst)
    assert md.splitlines()[0] == "# Title"
    assert "## Section" in md and "### Sub" in md
    assert "Use `SVC` or `train_test_split`, see\nthe guide and docs; math $x^2$." in md
    assert "**Note:**" in md and "Careful with `C`." in md
    assert "```\n>>> clf.fit(X, y)\n```" in md
    assert "currentmodule" not in md and "foo.png" not in md and "label" not in md


def test_eval_metrics():
    from docqa.chunk import Chunk
    from docqa.evaluate import Question, score

    q = Question("q", source="a.md", section="A > B")
    right = Chunk("t", "a.md", 0, section="A > B > C")    # subsection counts
    same_page = Chunk("t", "a.md", 1, section="A > D")
    other = Chunk("t", "b.md", 0, section="A > B")
    m = score([[same_page, other, right]], [q])
    assert m["page@1"] == 1.0 and m["section@1"] == 0.0 and m["section@3"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 3)


def test_eval_set_points_at_real_sections():
    from docqa.evaluate import load_questions

    questions = load_questions()
    sections = {(c.source, c.section) for c in load_and_chunk()}
    missing = [q.question for q in questions if (q.source, q.section) not in sections]
    assert len(questions) >= 100 and not missing, missing
