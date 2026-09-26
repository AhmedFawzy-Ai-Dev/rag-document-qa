"""Configuration for the RAG document-Q&A app."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"          # source documents (.md / .txt)
INDEX_PATH = DATA_DIR / "index.joblib"  # built retrieval index

# --- chunking ---
CHUNK_WORDS = 120        # target words per chunk
CHUNK_OVERLAP = 30       # words shared between consecutive chunks

# --- retrieval ---
# Backend for turning text into vectors:
#   "tfidf"       -> scikit-learn TF-IDF (default; no downloads, fully local)
#   "transformer" -> mean-pooled embeddings from a small BERT (optional extra)
EMBEDDING_BACKEND = os.environ.get("RAG_BACKEND", "tfidf")
EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "google/bert_uncased_L-4_H-256_A-4")
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

# Retrieved passages scoring below this are dropped as irrelevant; if none remain,
# the app says it found nothing instead of answering from unrelated text. The
# TF-IDF default is calibrated on the bundled docs (on-topic questions score
# >= 0.16; off-topic ones mostly 0). Dense embeddings score on a different scale,
# so the transformer backend doesn't filter unless RAG_MIN_SCORE is set.
MIN_SCORE_DEFAULTS = {"tfidf": 0.1, "transformer": 0.0}
MIN_SCORE_OVERRIDE = os.environ.get("RAG_MIN_SCORE")


def min_score(backend: str) -> float:
    """The relevance threshold for a retrieval backend."""
    if MIN_SCORE_OVERRIDE is not None:
        return float(MIN_SCORE_OVERRIDE)
    return MIN_SCORE_DEFAULTS.get(backend, 0.0)

# --- generation (Claude) ---
# Default to Claude Opus 5; override with RAG_MODEL (e.g. claude-haiku-4-5 for
# cheaper/faster answers). The app reads ANTHROPIC_API_KEY from the environment
# and never stores it; with no key it answers in extractive mode.
GEN_MODEL = os.environ.get("RAG_MODEL", "claude-opus-5")
MAX_ANSWER_TOKENS = 1024
