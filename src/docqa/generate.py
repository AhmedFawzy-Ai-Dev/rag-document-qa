"""Answer generation.

With an ``ANTHROPIC_API_KEY`` in the environment, the retrieved context is sent
to Claude, which writes a grounded answer that cites its sources. With no key,
the app degrades gracefully to an **extractive** answer (the most relevant
passages, with citations) so the repo still runs and tests end to end offline.

The API key is read from the environment by the Anthropic SDK and is never
stored, logged, or printed by this code.
"""
from __future__ import annotations

import os

from . import config
from .store import Hit

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you use with their bracketed numbers, e.g. [1] or [2]. "
    "If the context does not contain the answer, say so plainly instead of "
    "guessing. Be concise and factual."
)


def _format_context(hits: list[Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit.chunk.source})\n{hit.chunk.text}")
    return "\n\n".join(lines)


def _sources(hits: list[Hit]) -> list[dict]:
    return [
        {"n": i, "source": h.chunk.source, "score": round(h.score, 4)}
        for i, h in enumerate(hits, start=1)
    ]


class NoAnswerError(Exception):
    """Claude responded but produced no answer text (e.g. a refusal)."""


def has_credentials() -> bool:
    """True if an Anthropic API key is available in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def extractive_answer(question: str, hits: list[Hit]) -> dict:
    """No-LLM fallback: surface the top passages as the answer."""
    if not hits or hits[0].score <= 0:
        answer = "I couldn't find anything relevant to that in the documents."
    else:
        top = hits[0]
        hint = "" if has_credentials() else " Set ANTHROPIC_API_KEY for a synthesized answer."
        answer = (
            f"{top.chunk.text}\n\n"
            f"(Extractive mode — showing the most relevant passage [1] from "
            f"{top.chunk.source}.{hint})"
        )
    return {"answer": answer, "sources": _sources(hits), "mode": "extractive"}


def synthesize(question: str, hits: list[Hit], model: str | None = None) -> dict:
    """Ask Claude to answer grounded in the retrieved context."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    context = _format_context(hits)
    user_content = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the passages above, and cite them by number."
    )
    response = client.messages.create(
        model=model or config.GEN_MODEL,
        max_tokens=config.MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise NoAnswerError(f"no answer text (stop_reason={response.stop_reason})")
    return {"answer": text, "sources": _sources(hits), "mode": "claude"}


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
        result = extractive_answer(question, hits)
        result["note"] = (
            f"Claude call failed ({type(exc).__name__}: {exc}); used extractive fallback."
        )
        return result
