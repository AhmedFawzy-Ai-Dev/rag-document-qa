"""The RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
from __future__ import annotations

from . import config
from .generate import generate
from .store import Hit, VectorStore


def retrieve(question: str, store: VectorStore, k: int | None = None,
             min_score: float | None = None) -> list[Hit]:
    """The passages an answer is grounded in: top-k hits above the relevance floor."""
    hits = store.search(question, k or config.TOP_K)
    return [h for h in hits
            if h.score >= (config.min_score(h.kind) if min_score is None else min_score)]


def answer(question: str, store: VectorStore | None = None, k: int | None = None,
           model: str | None = None, min_score: float | None = None) -> dict:
    """Answer a question over the indexed documents.

    Returns ``{question, answer, sources, mode}`` where ``mode`` is "claude" when
    an answer was synthesized by the model, or "extractive" for the offline
    fallback. A ``note`` key is added when Claude was tried but failed.
    """
    store = store or VectorStore.load()
    result = generate(question, retrieve(question, store, k, min_score), model)
    return {"question": question, **result}
