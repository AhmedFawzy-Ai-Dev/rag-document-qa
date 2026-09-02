"""Split documents into overlapping, self-contained chunks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Chunk:
    text: str
    source: str   # file name the chunk came from
    index: int    # position of the chunk within its source


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Chunk one document, respecting paragraph boundaries where possible.

    Markdown headings are merged onto the following paragraph so every chunk
    carries context (a bare heading on its own is a poor retrieval unit).
    """
    chunks: list[Chunk] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pending_heading = ""
    for para in paragraphs:
        if para.lstrip().startswith("#"):
            heading = para.lstrip("#").strip()
            pending_heading = f"{pending_heading} — {heading}" if pending_heading else heading
            continue
        body = f"{pending_heading}. {para}" if pending_heading else para
        pending_heading = ""
        # A short paragraph is its own chunk; a long one is windowed with overlap.
        pieces = (
            [body]
            if len(body.split()) <= config.CHUNK_WORDS
            else _split_words(body, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
        )
        for piece in pieces:
            chunks.append(Chunk(text=piece, source=source, index=len(chunks)))
    if pending_heading:  # a trailing heading with no body
        chunks.append(Chunk(text=pending_heading, source=source, index=len(chunks)))
    return chunks


def load_and_chunk(docs_dir: Path | str | None = None) -> list[Chunk]:
    """Read every .md/.txt file in a directory and chunk them all."""
    docs_dir = Path(docs_dir) if docs_dir is not None else config.DOCS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(f"No docs directory at {docs_dir}.")
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise ValueError(f"No .md/.txt documents found in {docs_dir}.")
    return chunks
