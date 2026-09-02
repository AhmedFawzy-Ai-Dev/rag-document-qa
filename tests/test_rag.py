"""Tests for the RAG pipeline that run fully offline (no API key, no downloads)."""
from __future__ import annotations

from docqa.chunk import chunk_text, load_and_chunk
from docqa.generate import extractive_answer, has_credentials
from docqa.store import VectorStore, build_index


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
