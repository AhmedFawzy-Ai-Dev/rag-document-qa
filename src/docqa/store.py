"""A tiny in-memory vector store: build, persist, and cosine-search chunks.

For a small knowledge base this is all you need; the same interface scales to a
proper index (FAISS, a vector DB) by swapping ``search`` without touching the
rest of the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .chunk import Chunk, load_and_chunk
from .embed import get_embedder


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder, matrix, chunks: list[Chunk]):
        self.embedder = embedder
        self.matrix = matrix          # (n_chunks, dim), sparse or dense
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk], backend: str | None = None) -> VectorStore:
        embedder = get_embedder(backend)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        matrix = embedder.encode(texts)
        return cls(embedder, matrix, chunks)

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or config.TOP_K
        qv = self.embedder.encode_queries([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [Hit(chunk=self.chunks[i], score=float(sims[i])) for i in order]

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
        return joblib.load(path)


def build_index(docs_dir=None, backend: str | None = None, save: bool = True) -> VectorStore:
    """Load documents, chunk, embed, and (optionally) persist the index."""
    chunks = load_and_chunk(docs_dir)
    store = VectorStore.build(chunks, backend)
    if save:
        store.save()
    return store
