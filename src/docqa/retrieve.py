"""Retrieval building blocks: scorers, rank fusion and a cross-encoder reranker.

A *scorer* is fitted on the chunk texts and returns one relevance score per
chunk for a query:

    scorer.fit(texts)
    scores = scorer.scores(query)      # np.ndarray, shape (n_chunks,)

- ``BM25Scorer``: lexical matching (Okapi BM25), no downloads.
- ``CosineScorer``: cosine similarity between embeddings from an embedder in
  ``embed.py`` (TF-IDF vectors, or dense sentence-transformers vectors).

``rrf`` fuses several rankings (reciprocal rank fusion), and ``Reranker``
rescores the top candidates with a cross-encoder, which reads the question and
the passage together instead of comparing two independently computed vectors.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config


class BM25Scorer:
    """Okapi BM25 over unigrams with English stop words removed.

    Beat TF-IDF cosine on the scikit-learn eval (MRR 0.745 vs 0.696) with the
    textbook parameters k1=1.5, b=0.75, which are kept rather than tuned on the
    eval questions.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]) -> BM25Scorer:
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english")
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        doc_freq = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(doc_len.mean(), 1e-9))
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        # Precompute each (doc, term) weight so a query is one sparse product.
        tf.data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = tf.T.tocsr()          # (vocab, n_docs)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])   # term counts; repeated terms count twice
        return np.asarray((q @ self.weights).todense()).ravel()


class CosineScorer:
    """Cosine similarity between a query's and each chunk's embedding."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.name = embedder.name

    def fit(self, texts: list[str]) -> CosineScorer:
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)
        return self

    def scores(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self.embedder.encode_queries([query])
        return cosine_similarity(qv, self.matrix).ravel()


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: sum of 1 / (k + rank) over the rankings.

    Uses only ranks, so scorers on different scales (BM25, cosine) combine
    without calibration. k=60 is the value from the original paper.
    """
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


class Reranker:
    """A cross-encoder that scores (question, passage) pairs jointly."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANK_MODEL
        self._model = None

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512)
        return np.asarray(self._model.predict([(query, t) for t in texts], batch_size=16))


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None) -> Reranker:
    """One reranker per model and process, so the model loads only once."""
    return Reranker(model_name)
