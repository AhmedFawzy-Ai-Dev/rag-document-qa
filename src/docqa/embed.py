"""Embedding backends: sparse TF-IDF (default) and dense transformer (optional).

Both expose the same tiny interface so the rest of the app doesn't care which
one is in use:

    embedder.fit(texts)                     # learn any vocabulary/state from the corpus
    matrix = embedder.encode(texts)         # documents -> (n, d) matrix (sparse or dense)
    qv = embedder.encode_queries([query])   # queries, embedded for search against them

Queries and documents are encoded separately because some embedding models
expect a different prompt for each. Cosine similarity in ``store.py`` works on
either representation.
"""
from __future__ import annotations

import numpy as np

from . import config


class TfidfEmbedder:
    """scikit-learn TF-IDF — no downloads, fully local, a strong baseline."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            # Without stop words, "who"/"the"/"is" alone give off-topic questions
            # scores as high as genuinely relevant ones.
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]):
        if not self._fitted:
            raise RuntimeError("Call fit() before encode() for the TF-IDF backend.")
        return self.vectorizer.transform(texts)  # sparse (n, vocab)

    def encode_queries(self, texts: list[str]):
        return self.encode(texts)


class TransformerEmbedder:
    """Dense sentence embeddings via sentence-transformers (needs the `transformer` extra).

    Any sentence-transformers model works (RAG_EMBED_MODEL); its own pooling,
    normalization and query/document prompts are applied.
    """

    name = "transformer"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    # The index is pickled with joblib: persist only the model name and reload the
    # weights lazily, rather than embedding the whole model (and failing on the
    # unpicklable torch objects) in index.joblib.
    def __getstate__(self) -> dict:
        return {"model_name": self.model_name}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["model_name"])

    def fit(self, texts: list[str]) -> TransformerEmbedder:
        return self  # pretrained; nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode_document(texts, normalize_embeddings=True)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode_query(texts, normalize_embeddings=True)


def get_embedder(backend: str | None = None):
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "transformer":
        return TransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r} (use 'tfidf' or 'transformer').")
