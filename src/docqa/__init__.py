"""docqa -- a small Retrieval-Augmented Generation (RAG) Q&A app.

Retrieve relevant passages from your documents (TF-IDF or transformer
embeddings) and answer questions with Claude, grounded in the sources.
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import config  # noqa: F401
from .pipeline import answer  # noqa: F401
