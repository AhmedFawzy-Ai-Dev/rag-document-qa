"""Split documents into overlapping, self-contained chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    text: str
    source: str        # file name the chunk came from
    index: int         # position of the chunk within its source
    section: str = ""  # heading path, e.g. "Cross-validation > K-fold"


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


def _blocks(text: str) -> list[str]:
    """Blank-line-separated blocks, keeping fenced code (which has blank lines) whole."""
    blocks, current, in_fence = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    # drop HTML comments such as the corpus's "converted from" header
    return [b for b in blocks if b and not (b.startswith("<!--") and b.endswith("-->"))]


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Chunk one document along its heading structure.

    Consecutive blocks of the same section are packed into chunks of up to
    ``CHUNK_WORDS`` words; a longer block is windowed with overlap. Every chunk
    starts with its heading path, so it carries its context wherever it lands
    (a passage from deep in a page otherwise loses what it is about).
    """
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []   # (level, title) of the enclosing headings
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        section = " > ".join(title for _, title in path)
        body = "\n\n".join(buf)
        pieces = (
            [body]
            if len(body.split()) <= config.CHUNK_WORDS
            else _split_words(body, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
        )
        for piece in pieces:
            text = f"{section}. {piece}" if section else piece
            chunks.append(Chunk(text=text, source=source, index=len(chunks), section=section))
        buf.clear()

    for block in _blocks(text):
        m = _HEADING.match(block) if "\n" not in block else None
        if m:
            flush()
            level = len(m.group(1))
            path = [p for p in path if p[0] < level] + [(level, m.group(2).strip())]
            continue
        if buf and len(" ".join(buf).split()) + len(block.split()) > config.CHUNK_WORDS:
            flush()
        buf.append(block)
    flush()
    return chunks


SUPPORTED = {".md", ".txt", ".pdf"}


def pdf_to_markdown(path: Path | str) -> str:
    """A PDF as Markdown with one "## Page n" section per page (needs pypdf).

    Page numbers become part of each chunk's section path, so an answer can cite
    "Page 3". Pages without extractable text (scans) are skipped.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata else None) or Path(path).stem
    parts = [f"# {title.strip()}"]
    for n, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"## Page {n}\n\n{text}")
    if len(parts) == 1:
        raise ValueError(f"{Path(path).name} has no extractable text (is it a scanned PDF?).")
    return "\n\n".join(parts)


def load_file(path: Path | str, name: str | None = None) -> list[Chunk]:
    """Chunk one .md/.txt/.pdf file; ``name`` overrides the source name shown."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix!r}; use {', '.join(sorted(SUPPORTED))}.")
    text = pdf_to_markdown(path) if suffix == ".pdf" else path.read_text(encoding="utf-8")
    return chunk_text(text, name or path.name)


def load_and_chunk(docs_dir: Path | str | None = None) -> list[Chunk]:
    """Read every .md/.txt/.pdf file in a directory and chunk them all."""
    docs_dir = Path(docs_dir) if docs_dir is not None else config.DOCS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"No docs directory at {docs_dir}. Pass --docs, or set RAG_DOCS_DIR to it."
        )
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() in SUPPORTED:
            chunks.extend(load_file(path))
    if not chunks:
        raise ValueError(f"No .md/.txt/.pdf documents found in {docs_dir}.")
    return chunks
