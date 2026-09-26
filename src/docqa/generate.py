"""Answer generation with verified citations.

With an ``ANTHROPIC_API_KEY`` in the environment, each retrieved passage is sent
to Claude as its own document with the Citations API enabled. Claude's answer
then comes back with citations the API itself attaches: for every cited claim,
the passage it comes from and the exact sentence quoted (``cited_text``), so a
citation can never point at a passage that doesn't say it. The [n] markers in
the answer are built from those citations, not written by the model.

``generate_stream`` yields the answer as it is written (text and citation
events), ending with the same result ``generate`` returns.

With no key, the app degrades gracefully to an **extractive** answer (the most
relevant passage) so the repo still runs and tests end to end offline. The API
key is read from the environment by the Anthropic SDK and is never stored,
logged, or printed by this code.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

from . import config
from .store import Hit

SYSTEM_PROMPT = (
    "You answer questions about the provided documents, which are passages from "
    "the scikit-learn user guide. Use ONLY those documents. If they do not contain "
    "the answer, say so plainly instead of guessing. Be concise and factual."
)


def _format_context(hits: list[Hit]) -> str:
    """Numbered passages as plain text (the judge's view of the evidence)."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit.chunk.source})\n{hit.chunk.text}")
    return "\n\n".join(lines)


def _sources(hits: list[Hit]) -> list[dict]:
    return [
        {"n": i, "source": h.chunk.source, "section": h.chunk.section, "score": round(h.score, 4)}
        for i, h in enumerate(hits, start=1)
    ]


def _body(hit: Hit) -> str:
    """Chunk text without its heading-path prefix (the title carries that)."""
    prefix = f"{hit.chunk.section}. "
    text = hit.chunk.text
    return text[len(prefix):] if hit.chunk.section and text.startswith(prefix) else text


def _documents(hits: list[Hit]) -> list[dict]:
    """One citable plain-text document per passage (cited sentence by sentence)."""
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": _body(h)},
            "title": f"{h.chunk.source}: {h.chunk.section}" if h.chunk.section else h.chunk.source,
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def _request(question: str, hits: list[Hit], model: str | None) -> dict:
    return {
        "model": model or config.GEN_MODEL,
        "max_tokens": config.MAX_ANSWER_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [
            *_documents(hits), {"type": "text", "text": question}]}],
    }


def _citation(cite, hits: list[Hit]) -> dict | None:
    """A citation from the API as {n, source, section, quote}; None if unusable."""
    i = getattr(cite, "document_index", None)
    if i is None or not 0 <= i < len(hits):
        return None
    return {"n": i + 1, "source": hits[i].chunk.source, "section": hits[i].chunk.section,
            "quote": getattr(cite, "cited_text", "").strip()}


class NoAnswerError(Exception):
    """Claude responded but produced no answer text (e.g. a refusal)."""


def has_credentials() -> bool:
    """True if an Anthropic API key is available in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def extractive_answer(question: str, hits: list[Hit]) -> dict:
    """No-LLM fallback: surface the top passages as the answer.

    ``passage`` carries the top passage's parts (for a UI to lay out); ``answer``
    is the same content as plain text.
    """
    result = {"sources": _sources(hits), "mode": "extractive", "citations": []}
    if not hits or hits[0].score <= 0:
        result["answer"] = "I couldn't find anything relevant to that in the documents."
        return result
    top = hits[0]
    hint = "" if has_credentials() else " Set ANTHROPIC_API_KEY for a synthesized answer."
    result["answer"] = (
        f"{top.chunk.text}\n\n"
        f"(Extractive mode — showing the most relevant passage [1] from "
        f"{top.chunk.source}.{hint})"
    )
    result["passage"] = {"source": top.chunk.source, "section": top.chunk.section,
                         "text": _body(top)}
    return result


def build_result(message, hits: list[Hit]) -> dict:
    """Turn a Messages API response into the app's result.

    Each text block's citations become [n] markers right after that block, and
    the distinct quotes are collected in ``citations``.
    """
    parts, citations, seen = [], [], set()
    for block in message.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        numbers = []
        for cite in getattr(block, "citations", None) or []:
            c = _citation(cite, hits)
            if c is None:
                continue
            if c["n"] not in numbers:
                numbers.append(c["n"])
            if (c["n"], c["quote"]) not in seen:
                seen.add((c["n"], c["quote"]))
                citations.append(c)
        if numbers:
            parts.append(" " + "".join(f"[{n}]" for n in numbers))
    text = "".join(parts).strip()
    if not text:
        raise NoAnswerError(f"no answer text (stop_reason={message.stop_reason})")
    result = {"answer": text, "sources": _sources(hits), "mode": "claude",
              "citations": citations,
              "usage": {"input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens}}
    if message.stop_reason == "max_tokens":
        result["note"] = "Answer cut off at the token limit; raise RAG_MAX_TOKENS."
    return result


def synthesize(question: str, hits: list[Hit], model: str | None = None) -> dict:
    """Ask Claude for an answer grounded in, and cited from, the retrieved passages."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return build_result(client.messages.create(**_request(question, hits, model)), hits)


def synthesize_stream(question: str, hits: list[Hit], model: str | None = None) -> Iterator[dict]:
    """Stream Claude's answer: {"type": "text"|"citation"|"done", ...} events."""
    import anthropic

    client = anthropic.Anthropic()
    with client.messages.stream(**_request(question, hits, model)) as stream:
        for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "text_delta":
                yield {"type": "text", "text": event.delta.text}
            elif event.delta.type == "citations_delta":
                c = _citation(event.delta.citation, hits)
                if c is not None:
                    yield {"type": "citation", **c}
        message = stream.get_final_message()
    yield {"type": "done", "result": build_result(message, hits)}


def _fallback(question: str, hits: list[Hit], exc: Exception) -> dict:
    result = extractive_answer(question, hits)
    result["note"] = f"Claude call failed ({type(exc).__name__}: {exc}); used extractive fallback."
    return result


def generate(question: str, hits: list[Hit], model: str | None = None) -> dict:
    """Synthesize with Claude when possible; otherwise fall back to extractive.

    With no relevant passages there is nothing to ground an answer in, so Claude
    isn't called. SDK errors (bad key, unknown model, network) still fall back,
    but the reason is returned in ``note`` so it isn't mistaken for "no key".
    Anything else is a bug and propagates.
    """
    if not hits or not has_credentials():
        return extractive_answer(question, hits)
    import anthropic

    try:
        return synthesize(question, hits, model)
    except (anthropic.AnthropicError, NoAnswerError) as exc:
        return _fallback(question, hits, exc)


def generate_stream(question: str, hits: list[Hit], model: str | None = None) -> Iterator[dict]:
    """Like ``generate``, but yields text/citation events before the final result.

    On an SDK error mid-stream, the final "done" event carries the extractive
    fallback (with a note), and consumers should replace any partial text.
    """
    if not hits or not has_credentials():
        yield {"type": "done", "result": extractive_answer(question, hits)}
        return
    import anthropic

    try:
        yield from synthesize_stream(question, hits, model)
    except (anthropic.AnthropicError, NoAnswerError) as exc:
        yield {"type": "done", "result": _fallback(question, hits, exc)}
