"""Embedding backends: sparse TF-IDF (default) and dense transformer (optional).

Both expose the same tiny interface so the rest of the app doesn't care which
one is in use:

    embedder.fit(texts)          # learn any vocabulary/state from the corpus
    matrix = embedder.encode(texts)   # -> a (n, d) matrix (sparse or dense)

Cosine similarity in ``store.py`` works on either representation.
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


class TransformerEmbedder:
    """Mean-pooled embeddings from a small BERT (needs the `transformer` extra)."""

    name = "transformer"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self._tok = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        # bert-mini ships no fast tokenizer; the bert-base vocab is compatible.
        self._tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._model = AutoModel.from_pretrained(self.model_name).eval()

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
        import torch

        self._ensure_loaded()
        vectors = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                enc = self._tok(
                    batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
                )
                out = self._model(**enc).last_hidden_state          # (b, t, h)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (b, t, 1)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vectors.append(pooled.cpu().numpy())
        arr = np.concatenate(vectors, axis=0)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


def get_embedder(backend: str | None = None):
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "transformer":
        return TransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r} (use 'tfidf' or 'transformer').")
