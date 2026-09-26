"""Build the retrieval index from the documents in ``RAG_DOCS_DIR`` (default: data/sklearn).

    python -m docqa.ingest                 # auto: hybrid if the extra is installed, else BM25
    python -m docqa.ingest --backend bm25
"""
from __future__ import annotations

import argparse

from . import config
from .store import BACKENDS, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--backend", choices=["auto", *BACKENDS, "dense"],
                        default=config.EMBEDDING_BACKEND,
                        help="Retrieval backend (default: RAG_BACKEND, else auto).")
    parser.add_argument("--docs", default=None, help="Path to a documents directory.")
    args = parser.parse_args()

    store = build_index(docs_dir=args.docs, backend=args.backend, save=True)
    print(f"Indexed {len(store.chunks)} chunks from {args.docs or config.DOCS_DIR} "
          f"using the '{store.backend}' backend.")
    print(f"Saved index -> {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
