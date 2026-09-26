"""Tests for the RAG pipeline. They run offline with no API key; the transformer
tests also need the `transformer` extra (and download the embedding model), and
are skipped without it.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
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
from docqa.retrieve import BM25Scorer, CosineScorer, rrf
from docqa.store import VectorStore, build_index, resolve_backend

NOTES = config.DATA_DIR / "ml_notes"  # small corpus: fast, and its answers are known
LEAKAGE_Q = "What is data leakage and how do you detect it?"


@pytest.fixture(autouse=True)
def no_rerank_by_default(monkeypatch):
    # RAG_RERANK=auto turns reranking on whenever sentence-transformers is installed;
    # tests opt in explicitly so they stay fast and don't depend on the machine.
    monkeypatch.setattr(config, "RERANK", "0")


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
    monkeypatch.setattr(ask, "answer_stream",
                        lambda question, k=None: iter([{"type": "done", "result": result}]))
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


def _cite(doc_index, quote):
    """A citation as the API returns it for a plain-text document."""
    return SimpleNamespace(type="char_location", document_index=doc_index, cited_text=quote,
                           start_char_index=0, end_char_index=len(quote))


def _fake_claude(monkeypatch, text, stop_reason="end_turn", citations=(), fail_after=None):
    """A fake client for both messages.create and messages.stream.

    The stream yields the documented event shapes (content_block_delta with
    text_delta / citations_delta); ``fail_after`` raises an SDK error mid-stream.
    """
    calls = []
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text, citations=list(citations))],
        stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=100, output_tokens=20))

    def delta(**kw):
        return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(**kw))

    class Stream:
        def __init__(self, kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            yield SimpleNamespace(type="content_block_start")
            for i, word in enumerate(text.split(" ")):
                if fail_after is not None and i == fail_after:
                    raise anthropic.AnthropicError("connection dropped")
                yield delta(type="text_delta", text=word + " ")
            for c in citations:
                yield delta(type="citations_delta", citation=c)

        def get_final_message(self):
            return message

    def create(**kwargs):
        calls.append(kwargs)
        return message

    client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=lambda **kw: Stream(kw)))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
    return calls


def test_claude_answer_path(tfidf_store, fake_key, monkeypatch):
    calls = _fake_claude(monkeypatch, "Leakage is ...", citations=[_cite(0, "Data leakage happens.")])
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["mode"] == "claude"
    assert result["answer"] == "Leakage is ... [1]"      # marker built from the citation
    assert result["citations"] == [{"n": 1, "source": "data_leakage.md",
                                    "section": result["sources"][0]["section"],
                                    "quote": "Data leakage happens."}]
    assert "note" not in result
    (req,) = calls
    assert req["model"] == config.GEN_MODEL
    assert req["max_tokens"] == config.MAX_ANSWER_TOKENS
    assert req["system"] == generate.SYSTEM_PROMPT
    *docs, question = req["messages"][0]["content"]
    assert question == {"type": "text", "text": LEAKAGE_Q}
    assert len(docs) == len(result["sources"])
    first = docs[0]
    assert first["citations"] == {"enabled": True} and first["source"]["type"] == "text"
    assert first["title"].startswith("data_leakage.md: ")
    # the heading path is in the title, not repeated in the citable text
    assert not first["source"]["data"].startswith(result["sources"][0]["section"])


def test_citations_are_deduplicated_and_bounded(tfidf_store):
    hits = tfidf_store.search(LEAKAGE_Q, 2)
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="A.", citations=[_cite(0, "x"), _cite(0, "x")]),
            SimpleNamespace(type="text", text=" B.", citations=[_cite(1, "y"), _cite(7, "bad")]),
            SimpleNamespace(type="text", text=" C.", citations=None),
        ],
        stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    result = generate.build_result(message, hits)
    assert result["answer"] == "A. [1] B. [2] C."
    assert [(c["n"], c["quote"]) for c in result["citations"]] == [(1, "x"), (2, "y")]


def test_streaming_yields_text_then_the_result(tfidf_store, fake_key, monkeypatch):
    from docqa.pipeline import answer_stream

    _fake_claude(monkeypatch, "Leakage is bad", citations=[_cite(0, "q")])
    events = list(answer_stream(LEAKAGE_Q, store=tfidf_store))
    assert "".join(e["text"] for e in events if e["type"] == "text") == "Leakage is bad "
    assert [e["n"] for e in events if e["type"] == "citation"] == [1]
    done = events[-1]
    assert done["type"] == "done" and done["result"]["answer"] == "Leakage is bad [1]"
    assert done["result"]["question"] == LEAKAGE_Q


def test_stream_error_ends_with_the_fallback(tfidf_store, fake_key, monkeypatch):
    from docqa.pipeline import answer_stream

    _fake_claude(monkeypatch, "Leakage is bad", fail_after=1)
    events = list(answer_stream(LEAKAGE_Q, store=tfidf_store))
    assert events[0] == {"type": "text", "text": "Leakage "}
    result = events[-1]["result"]
    assert result["mode"] == "extractive" and "connection dropped" in result["note"]


def test_ask_cli_streams_and_prints_citations(tfidf_store, fake_key, monkeypatch, capsys):
    _fake_claude(monkeypatch, "Leakage is bad", citations=[_cite(0, "Data leakage happens.")])
    monkeypatch.setattr(VectorStore, "load", staticmethod(lambda path=None: tfidf_store))
    monkeypatch.setattr("sys.argv", ["ask", LEAKAGE_Q])
    ask.main()
    out = capsys.readouterr().out
    assert "A: Leakage is bad" in out and "[mode: claude]" in out
    assert '[1] "Data leakage happens."' in out and "data_leakage.md >" in out


def test_truncated_claude_answer_is_noted(tfidf_store, fake_key, monkeypatch):
    _fake_claude(monkeypatch, "Leakage is", stop_reason="max_tokens")
    result = answer(LEAKAGE_Q, store=tfidf_store)
    assert result["mode"] == "claude"
    assert "RAG_MAX_TOKENS" in result["note"]


def test_search_encodes_query_as_a_query():
    # Some embedding models use different prompts for queries and documents.
    class Embedder:
        name = "fake"

        def fit(self, texts):
            return self

        def encode(self, texts):          # documents
            return np.array([[0.0, 1.0], [1.0, 0.0]])

        def encode_queries(self, texts):
            return np.array([[1.0, 0.0]])

    chunks = load_and_chunk(NOTES)[:2]
    scorer = CosineScorer(Embedder()).fit([c.text for c in chunks])
    store = VectorStore({"fake": scorer}, chunks, backend="fake")
    assert store.search("q", k=1)[0].chunk is chunks[1]


# --- BM25, rank fusion, reranking --------------------------------------------------------

def test_bm25_ranks_term_matches_and_ignores_stop_words():
    texts = ["the cat sat on the mat", "dogs chase cats", "stock market prices fell"]
    bm25 = BM25Scorer().fit(texts)
    s = bm25.scores("the market")
    assert s.argmax() == 2 and s[0] == 0 and s[1] == 0     # "the" is a stop word
    assert not bm25.scores("quantum chromodynamics").any()  # no overlap -> all zero


def test_rrf_fuses_ranks():
    a = np.array([3.0, 2.0, 1.0])       # ranks 1, 2, 3
    b = np.array([0.1, 0.9, 0.5])       # ranks 3, 1, 2
    fused = rrf([a, b], k=60)
    assert fused == pytest.approx([1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])
    assert fused.argmax() == 1          # 2nd + 1st beats 1st + 3rd


def test_rerank_reorders_candidates(monkeypatch, tfidf_store):
    class ReverseReranker:
        def score(self, query, texts):
            return -np.arange(len(texts), dtype=float)[::-1]   # last candidate best

    monkeypatch.setattr("docqa.store.get_reranker", lambda: ReverseReranker())
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 5)
    plain = tfidf_store.search(LEAKAGE_Q, k=5, rerank=False)
    reranked = tfidf_store.search(LEAKAGE_Q, k=5, rerank=True)
    assert [h.chunk for h in reranked] == [h.chunk for h in plain][::-1]
    assert {h.kind for h in reranked} == {"rerank"} and plain[0].kind == "tfidf"


def test_resolve_backend(monkeypatch):
    monkeypatch.setattr(config, "dense_available", lambda: False)
    assert resolve_backend("auto") == "bm25"
    monkeypatch.setattr(config, "dense_available", lambda: True)
    assert resolve_backend("auto") == "hybrid"
    assert resolve_backend("dense") == "transformer"
    with pytest.raises(ValueError):
        resolve_backend("nope")


def test_old_index_format_is_reported(tmp_path):
    path = tmp_path / "index.joblib"
    joblib.dump(SimpleNamespace(embedder=None, matrix=None, chunks=[]), path)
    with pytest.raises(RuntimeError, match="older version"):
        VectorStore.load(path)


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


@pytest.fixture(scope="module")
def hybrid_store(transformer_store):
    # (depends on transformer_store only for its skip-without-the-extra logic)
    return build_index(NOTES, backend="hybrid", save=False)


def test_hybrid_search_fuses_and_can_be_restricted(hybrid_store):
    hits = hybrid_store.search(LEAKAGE_Q, k=3, rerank=False)
    assert hits[0].kind == "rrf" and hits[0].chunk.source == "data_leakage.md"
    only_bm25 = hybrid_store.search(LEAKAGE_Q, k=1, rerank=False, scorers=["bm25"])
    assert only_bm25[0].kind == "bm25"


def test_real_reranker(hybrid_store):
    hits = hybrid_store.search("my fraud model is 99% right but catches nothing", k=3, rerank=True)
    assert [h.kind for h in hits] == ["rerank"] * 3
    assert hits[0].chunk.source == "evaluation_metrics.md"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_reranker_filters_off_topic_questions(hybrid_store, monkeypatch):
    monkeypatch.setattr(config, "RERANK", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    off_topic = answer("Who won the 2022 World Cup?", store=hybrid_store)
    assert off_topic["sources"] == [] and "couldn't find" in off_topic["answer"]
    on_topic = answer(LEAKAGE_Q, store=hybrid_store)
    assert on_topic["sources"][0]["source"] == "data_leakage.md"


# --- Claude as the judge (fake client) ----------------------------------------------------

def test_judge_sends_schema_and_parses_verdict(tfidf_store):
    from docqa import judge

    verdict = {"support": "most", "unsupported_claims": ["x"], "answers_question": True,
               "matches_reference": True, "says_not_found": False, "citations_valid": True,
               "reason": "ok"}
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(verdict))],
                               stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=500, output_tokens=80))

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    hits = tfidf_store.search(LEAKAGE_Q, 2)
    got, usage = judge.judge_answer(client, LEAKAGE_Q, "leakage is ...", hits,
                                    "Leakage is [1].", "claude-opus-5")
    assert got == verdict and usage.input_tokens == 500
    (req,) = calls
    assert req["output_config"]["format"]["schema"] is judge.VERDICT_SCHEMA
    assert "[1] (source: data_leakage.md)" in req["messages"][0]["content"]
    assert "leakage is ..." in req["messages"][0]["content"]


def test_judge_rejects_truncated_verdicts(tfidf_store):
    from docqa import judge

    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="{")], stop_reason="max_tokens",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1))))
    with pytest.raises(RuntimeError, match="max_tokens"):
        judge.judge_answer(client, "q", "", [], "a", "claude-opus-5")


def test_judge_summary():
    from docqa.judge import summarize

    def row(support, match):
        return {"verdict": {"support": support, "answers_question": True,
                            "matches_reference": match, "citations_valid": True,
                            "says_not_found": False}}

    s = summarize([row("all", True), row("most", True), row("none", False), {"error": "x"}])
    assert s["judged"] == 3 and s["errors"] == 1
    assert s["fully_supported"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["mostly_or_fully_supported"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["matches_reference"] == pytest.approx(2 / 3, abs=1e-3)


# --- files, follow-ups and the chat UI logic ----------------------------------------------

PDF = config.REPO_ROOT / "tests" / "data" / "model_cards.pdf"


def test_pdf_is_chunked_by_page():
    from docqa.chunk import load_file

    chunks = load_file(PDF)
    assert [c.section for c in chunks] == ["model_cards > Page 1", "model_cards > Page 2"]
    assert "demographic group" in chunks[1].text


def test_unsupported_files_are_rejected(tmp_path):
    from docqa.chunk import load_file

    bad = tmp_path / "slides.pptx"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(bad)


def test_uploaded_files_are_indexed_and_searchable(monkeypatch):
    from docqa.chat import index_files

    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "bm25")
    store = index_files([PDF, NOTES / "data_leakage.md"])
    hit = store.search("metrics for each demographic group", 1)[0]
    assert hit.chunk.source == "model_cards.pdf" and hit.chunk.section.endswith("Page 2")


def test_follow_up_rewriting(monkeypatch):
    from docqa.conversation import standalone_question

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert standalone_question("What is PCA?", []) == ("What is PCA?", "as-is")
    history = [{"role": "user", "content": "How does gradient boosting work?"},
               {"role": "assistant", "content": "It adds trees."}]
    q, how = standalone_question("does it handle NaNs?", history)
    assert how == "prefixed" and q == "How does gradient boosting work? does it handle NaNs?"


def test_follow_up_rewriting_with_claude(fake_key, monkeypatch):
    from docqa.conversation import standalone_question

    calls = _fake_claude(monkeypatch, "Does gradient boosting handle missing values?")
    history = [{"role": "user", "content": "How does gradient boosting work?"},
               {"role": "assistant", "content": "It adds trees."}]
    q, how = standalone_question("does it handle NaNs?", history)
    assert (q, how) == ("Does gradient boosting handle missing values?", "claude")
    (req,) = calls
    assert req["model"] == config.REWRITE_MODEL and req["output_config"] == {"effort": "low"}
    assert "does it handle NaNs?" in req["messages"][0]["content"]


def test_chat_streams_then_shows_cited_sentences(tfidf_store, fake_key, monkeypatch):
    from docqa.chat import respond

    _fake_claude(monkeypatch, "Leakage is bad", citations=[_cite(0, "Data leakage happens.")])
    snapshots = list(respond(LEAKAGE_Q, [], tfidf_store))
    assert snapshots[0].endswith(" ▌")                       # streaming cursor
    final = snapshots[-1]
    assert final.startswith("Leakage is bad [1]")
    assert "> **[1]** Data leakage happens." in final and "data_leakage.md ›" in final


def test_chat_reports_the_rewritten_question(tfidf_store, monkeypatch):
    from docqa.chat import respond

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    history = [{"role": "user", "content": "What is data leakage?"},
               {"role": "assistant", "content": "..."}]
    final = list(respond("how do I detect it?", history, tfidf_store))[-1]
    assert "searched for: *What is data leakage? how do I detect it?*" in final
    assert final.startswith("**")                      # the passage's section title
    assert "**Also relevant**" in final and "retrieval only" in final


def test_chat_session_limit(tfidf_store, monkeypatch):
    from docqa.chat import respond

    monkeypatch.setattr(config, "MAX_QUESTIONS_PER_SESSION", 1)
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    (reply,) = list(respond("q2", history, tfidf_store))
    assert "1 questions per session" in reply
