"""Configuration for the RAG document-Q&A app."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

# REPO_ROOT/data only exists when running from a checkout (or an editable
# install); a regular `pip install .` puts the package in site-packages, so point
# RAG_DATA_DIR at the folder holding the corpus and index.joblib.
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", REPO_ROOT / "data"))
# Source documents (.md / .txt). Default: the scikit-learn user guide (see
# data/README.md); data/ml_notes is a 4-note corpus the fast tests use.
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", DATA_DIR / "sklearn"))
INDEX_PATH = DATA_DIR / "index.joblib"  # built retrieval index
# Labeled questions for `python -m docqa.evaluate` (kept outside DOCS_DIR so
# they are never indexed).
EVAL_PATH = DATA_DIR / "sklearn_eval.jsonl"

# --- chunking ---
# Chosen on the eval set (see README, "Evaluation"): packing sections into
# ~200-word chunks beat 80/120/250/300 and one-paragraph-per-chunk on MRR for
# both backends, and keeps 94% of chunks within bge-small's 512-token window.
CHUNK_WORDS = 200        # max words per chunk
CHUNK_OVERLAP = 50       # words shared between windows of an over-long block

# --- retrieval ---
# Retrieval backend (scores on the eval set are in the README):
#   "bm25"        -> Okapi BM25 keyword matching (no downloads, fully local)
#   "tfidf"       -> TF-IDF cosine (the original baseline)
#   "transformer" -> dense sentence-transformers embeddings (`transformer` extra)
#   "hybrid"      -> BM25 + dense, fused with reciprocal rank fusion (extra)
#   "auto"        -> hybrid when the extra is installed, else bm25 (default)
EMBEDDING_BACKEND = os.environ.get("RAG_BACKEND", "auto")
# bge-small (33M params, 384-d; a BERT fine-tuned for retrieval) beat
# all-MiniLM-L6-v2, multi-qa-MiniLM and a raw mean-pooled BERT-mini in a
# comparison on the ML notes (README, "Design notes").
EMBED_MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

# Reranking: a cross-encoder rescores the top RERANK_CANDIDATES chunks. "auto"
# turns it on when the `transformer` extra is installed. On the eval set,
# MiniLM-L6 (22M params) lifted hybrid MRR from 0.74 to 0.85. With 20 candidates
# (the right section is among them for 97% of questions) it is 30% faster than
# with 30 for an MRR 0.003 lower.
RERANK = os.environ.get("RAG_RERANK", "auto").lower()
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "20"))


def dense_available() -> bool:
    """Whether the `transformer` extra (sentence-transformers) is installed."""
    from importlib.util import find_spec

    return find_spec("sentence_transformers") is not None


def rerank_enabled() -> bool:
    if RERANK == "auto":
        return dense_available()
    return RERANK in {"1", "true", "yes", "on"}

# Retrieved passages scoring below this are dropped as irrelevant; if none remain,
# the app says it found nothing instead of answering from unrelated text.
# Neither backend's similarity scores really separate on- from off-topic
# questions on the scikit-learn corpus: with TF-IDF, 10% of eval questions have
# a top score below 0.1 while off-topic questions reach 0.27; bge-small's cosine
# scores overlap the same way. So the TF-IDF floor is a low 0.05 (drops 4% of
# correct hits, suppresses no eval question) that only catches questions sharing
# no vocabulary with the docs, and the transformer backend doesn't filter unless
# RAG_MIN_SCORE is set. Claude is told to say when the passages don't answer.
# BM25 only requires some term overlap; fused (rrf) scores are rank-based and
# carry no relevance signal, so they aren't filtered.
# The reranker's scores *do* separate the two: every eval question's best
# passage scores above -5.9, while 10 of 12 off-topic questions ("Who won the
# World Cup?") score below -8.4. -7 leaves a margin below the lowest on-topic
# score because it was calibrated on the eval questions themselves.
MIN_SCORE_DEFAULTS = {"tfidf": 0.05, "bm25": 1e-9, "transformer": 0.0, "rrf": 0.0,
                      "rerank": -7.0}
MIN_SCORE_OVERRIDE = os.environ.get("RAG_MIN_SCORE")


def min_score(kind: str) -> float:
    """The relevance threshold for a kind of score (a scorer, "rrf" or "rerank")."""
    if MIN_SCORE_OVERRIDE is not None:
        return float(MIN_SCORE_OVERRIDE)
    return MIN_SCORE_DEFAULTS.get(kind, 0.0)

# --- generation (Claude) ---
# Default to Claude Opus 5; override with RAG_MODEL (e.g. claude-haiku-4-5 for
# cheaper/faster answers). The app reads ANTHROPIC_API_KEY from the environment
# and never stores it; with no key it answers in extractive mode.
GEN_MODEL = os.environ.get("RAG_MODEL", "claude-opus-5")
# Claude Opus 5 thinks by default and thinking counts toward max_tokens, so a
# tight cap can cut the answer off (or leave none); answers stay short anyway.
MAX_ANSWER_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "16000"))
# Model that grades answers in `python -m docqa.judge`.
JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", "claude-opus-5")
# Model that rewrites follow-up questions into standalone ones (chat UI).
REWRITE_MODEL = os.environ.get("RAG_REWRITE_MODEL", GEN_MODEL)
# Questions allowed per chat session in the web UI (protects a public demo's
# API credits); 0 for no limit.
MAX_QUESTIONS_PER_SESSION = int(os.environ.get("RAG_MAX_QUESTIONS", "20"))
