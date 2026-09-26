"""An in-memory index: build, persist, and search chunks.

A store holds one scorer per retrieval method. With one scorer, search ranks by
its scores; with several (the ``hybrid`` backend: BM25 + dense embeddings), their
rankings are fused with reciprocal rank fusion. Optionally, the top candidates
are then reranked by a cross-encoder. For a knowledge base of this size this is
all you need; ``search`` is the one place to swap in FAISS or a vector DB.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunk import Chunk, load_and_chunk
from .embed import get_embedder
from .retrieve import BM25Scorer, CosineScorer, get_reranker, rrf

# backend -> the scorers its index holds
BACKENDS = {
    "tfidf": ["tfidf"],
    "bm25": ["bm25"],
    "transformer": ["transformer"],
    "hybrid": ["bm25", "transformer"],
}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    kind: str = ""   # what the score is: a scorer name, "rrf" or "rerank"


def resolve_backend(backend: str | None = None) -> str:
    """Map "auto" to the best backend installed: hybrid with the extra, else BM25."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "dense":
        backend = "transformer"
    if backend == "auto":
        return "hybrid" if config.dense_available() else "bm25"
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; use one of {sorted(BACKENDS)} or 'auto'.")
    return backend


def _make_scorer(name: str):
    return BM25Scorer() if name == "bm25" else CosineScorer(get_embedder(name))


class VectorStore:
    def __init__(self, scorers: dict, chunks: list[Chunk], backend: str):
        self.scorers = scorers        # name -> fitted scorer
        self.chunks = chunks
        self.backend = backend

    @classmethod
    def build(cls, chunks: list[Chunk], backend: str | None = None) -> VectorStore:
        backend = resolve_backend(backend)
        texts = [c.text for c in chunks]
        scorers = {name: _make_scorer(name).fit(texts) for name in BACKENDS[backend]}
        return cls(scorers, chunks, backend)

    def search(self, query: str, k: int | None = None, *, rerank: bool | None = None,
               scorers: list[str] | None = None) -> list[Hit]:
        """Top-k chunks for a query.

        ``scorers`` restricts the search to some of the index's scorers (the eval
        uses it to compare them on one index); ``rerank`` defaults to the
        RAG_RERANK setting.
        """
        k = k or config.TOP_K
        names = scorers or list(self.scorers)
        lists = [self.scorers[n].scores(query) for n in names]
        scores, kind = (lists[0], names[0]) if len(lists) == 1 else (rrf(lists), "rrf")
        if config.rerank_enabled() if rerank is None else rerank:
            candidates = np.argsort(-scores, kind="stable")[: config.RERANK_CANDIDATES]
            rerank_scores = get_reranker().score(query, [self.chunks[i].text for i in candidates])
            order = np.argsort(-rerank_scores, kind="stable")[:k]
            return [Hit(self.chunks[candidates[j]], float(rerank_scores[j]), "rerank")
                    for j in order]
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i]), kind) for i in order]

    def save(self, path: Path | str | None = None) -> Path:
        import joblib

        path = Path(path) if path is not None else config.INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str | None = None) -> VectorStore:
        import joblib

        path = Path(path) if path is not None else config.INDEX_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build it with `python -m docqa.ingest`."
            )
        store = joblib.load(path)
        if not hasattr(store, "scorers"):
            raise RuntimeError(
                f"{path} was built by an older version; rebuild it with `python -m docqa.ingest`."
            )
        return store


def build_index(docs_dir=None, backend: str | None = None, save: bool = True) -> VectorStore:
    """Load documents, chunk, embed, and (optionally) persist the index."""
    chunks = load_and_chunk(docs_dir)
    store = VectorStore.build(chunks, backend)
    if save:
        store.save()
    return store
