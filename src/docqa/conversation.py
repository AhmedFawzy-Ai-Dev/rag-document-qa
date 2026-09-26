"""Follow-up questions: rewrite them into standalone ones before retrieval.

Retrieval sees one question at a time, so "does it handle missing values?"
after a question about gradient boosting would search for "it". With a key,
Claude rewrites the follow-up using the recent conversation (at low effort, as
it is a small task). Without one, the previous question is prepended, which
gives the search the missing context well enough in practice.
"""
from __future__ import annotations

from . import config
from .generate import has_credentials

REWRITE_SYSTEM = (
    "You rewrite the user's latest message into a standalone question for searching "
    "documentation. Resolve pronouns and references using the conversation. If it is "
    "already standalone, return it unchanged. Output only the question, nothing else."
)
MAX_TURNS = 3   # recent user/assistant exchanges given to the rewriter


def _transcript(history: list[dict]) -> str:
    lines = []
    for msg in history[-2 * MAX_TURNS:]:
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(f"{msg['role'].upper()}: {content.strip()[:1500]}")
    return "\n".join(lines)


def standalone_question(question: str, history: list[dict]) -> tuple[str, str]:
    """(question to retrieve with, how it was produced: "as-is"/"claude"/"prefixed").

    ``history`` is the chat so far as ``{"role", "content"}`` messages, oldest first.
    """
    previous = [m["content"] for m in history
                if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if not previous:
        return question, "as-is"
    if has_credentials():
        import anthropic

        try:
            response = anthropic.Anthropic().messages.create(
                model=config.REWRITE_MODEL,
                max_tokens=2000,
                system=REWRITE_SYSTEM,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content":
                           f"Conversation:\n{_transcript(history)}\n\nLatest message: {question}"}],
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if text and response.stop_reason == "end_turn":
                return text, "claude"
        except anthropic.AnthropicError:
            pass    # fall through to the offline rewrite
    return f"{previous[-1]} {question}", "prefixed"
