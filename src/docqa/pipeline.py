"""The RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
from __future__ import annotations

from . import config
from .generate import generate
from .store import VectorStore


def answer(question: str, store: VectorStore | None = None, k: int | None = None,
           model: str | None = None, min_score: float | None = None) -> dict:
    """Answer a question over the indexed documents.

    Returns ``{question, answer, sources, mode}`` where ``mode`` is "claude" when
    an answer was synthesized by the model, or "extractive" for the offline
    fallback. A ``note`` key is added when Claude was tried but failed.
    """
    store = store or VectorStore.load()
    hits = store.search(question, k or config.TOP_K)
    threshold = config.min_score(store.embedder.name) if min_score is None else min_score
    hits = [h for h in hits if h.score >= threshold]
    result = generate(question, hits, model)
    return {"question": question, **result}
