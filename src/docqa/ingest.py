"""Build the retrieval index from the documents in ``data/docs/``.

    python -m docqa.ingest                 # default TF-IDF backend
    python -m docqa.ingest --backend transformer
"""
from __future__ import annotations

import argparse

from . import config
from .store import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--backend", choices=["tfidf", "transformer"],
                        default=config.EMBEDDING_BACKEND)
    parser.add_argument("--docs", default=None, help="Path to a documents directory.")
    args = parser.parse_args()

    store = build_index(docs_dir=args.docs, backend=args.backend, save=True)
    print(f"Indexed {len(store.chunks)} chunks from {args.docs or config.DOCS_DIR} "
          f"using the '{args.backend}' backend.")
    print(f"Saved index -> {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
