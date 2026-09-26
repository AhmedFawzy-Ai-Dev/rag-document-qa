"""Chat logic behind the web UI, kept free of Gradio so it can be tested.

``respond`` turns one chat turn into a stream of Markdown snapshots: the answer
as it is written, then the final answer with the sentences it cites. Follow-ups
are rewritten into standalone questions before retrieval (``conversation.py``).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from . import config
from .chunk import load_file
from .conversation import standalone_question
from .generate import generate_stream
from .pipeline import retrieve
from .store import VectorStore

SOURCES_RULE = "\n\n---\n"   # separates an answer from its sources in the chat


def index_files(paths: list[str | Path]) -> VectorStore:
    """An index over uploaded .pdf/.md/.txt files (same backend as the main index)."""
    chunks = []
    for path in paths:
        chunks.extend(load_file(path, name=Path(path).name))
    if not chunks:
        raise ValueError("No text found in the uploaded files.")
    return VectorStore.build(chunks)


def format_answer(result: dict, searched_for: str | None = None) -> str:
    """The final chat message: answer, then what was cited (or retrieved)."""
    parts = [result["answer"]]
    if result.get("citations"):
        parts.append(SOURCES_RULE + "**Sources: the sentences this answer relies on**")
        for c in result["citations"]:
            quote = " ".join(c["quote"].split())
            parts.append(f"\n\n> **[{c['n']}]** {quote}  \n> <sub>{c['source']} › {c['section']}</sub>")
    elif result["sources"]:
        parts.append(SOURCES_RULE + "**Retrieved passages**\n")
        parts.extend(f"\n- [{s['n']}] {s['source']} › {s.get('section', '')}"
                     for s in result["sources"])
    notes = []
    if searched_for:
        notes.append(f"searched for: *{searched_for}*")
    if result["mode"] != "claude":
        notes.append("extractive mode (no Claude answer)")
    if result.get("note"):
        notes.append(result["note"])
    if notes:
        parts.append("\n\n<sub>" + " · ".join(notes) + "</sub>")
    return "".join(parts)


def _answer_only(history: list[dict]) -> list[dict]:
    """History with the sources section stripped from assistant messages."""
    out = []
    for m in history:
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, str):
            content = content.split(SOURCES_RULE)[0]
        out.append({"role": m.get("role"), "content": content})
    return out


def respond(message: str, history: list[dict], store: VectorStore) -> Iterator[str]:
    """Markdown snapshots for one chat turn (the last one is the final answer)."""
    message = message.strip()
    if not message:
        return
    limit = config.MAX_QUESTIONS_PER_SESSION
    if limit and sum(1 for m in history if m.get("role") == "user") >= limit:
        yield (f"This demo allows {limit} questions per session, to keep its API costs in "
               "check. Refresh the page to start a new session, or run it yourself from GitHub.")
        return
    question, how = standalone_question(message, _answer_only(history))
    hits = retrieve(question, store)
    text = ""
    for event in generate_stream(question, hits):
        if event["type"] == "text":
            text += event["text"]
            yield text + " ▌"
        elif event["type"] == "done":
            yield format_answer(event["result"], question if how != "as-is" else None)
