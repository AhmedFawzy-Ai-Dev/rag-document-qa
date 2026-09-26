"""Answer-quality evaluation with Claude as the judge (needs ANTHROPIC_API_KEY).

    python -m docqa.judge --limit 10              # try it on 10 questions first
    python -m docqa.judge --json judged.json      # all 100 eval questions

For each eval question the full pipeline answers it (retrieval + Claude), then
a separate Claude call grades the answer against the retrieved passages and the
question's reference answer, returning a fixed JSON schema:

- ``support``: are the answer's claims backed by the passages ("all" / "most" /
  "some" / "none")? This is faithfulness: an answer can be right yet unsupported.
- ``answers_question`` and ``matches_reference``: does it answer what was asked,
  and agree with the reference answer written with the question?
- ``citations_valid``: do its [n] markers point at passages that say what's cited?

Per-question verdicts (with the judge's reasons) go to --json; the summary and
the token cost are printed. The judge is the same model family as the answerer,
which can flatter it; the reasons are there to spot-check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .evaluate import load_questions
from .generate import _format_context, generate, has_credentials
from .pipeline import retrieve
from .store import Hit, VectorStore

JUDGE_SYSTEM = (
    "You grade answers produced by a retrieval-augmented QA system over the "
    "scikit-learn user guide. Judge only what is written: the numbered passages "
    "are the sole evidence the system had. Be strict and specific."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "support": {
            "type": "string",
            "enum": ["all", "most", "some", "none"],
            "description": "How many of the answer's factual claims the passages support.",
        },
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "answers_question": {"type": "boolean"},
        "matches_reference": {"type": "boolean"},
        "says_not_found": {
            "type": "boolean",
            "description": "The answer says the passages don't contain the answer.",
        },
        "citations_valid": {
            "type": "boolean",
            "description": "Every [n] points at a passage that says what it is cited for.",
        },
        "reason": {"type": "string"},
    },
    "required": ["support", "unsupported_claims", "answers_question", "matches_reference",
                 "says_not_found", "citations_valid", "reason"],
    "additionalProperties": False,
}

# $ per million tokens (input, output), for the printed cost estimate only.
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def judge_prompt(question: str, reference: str, hits: list[Hit], answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Reference answer (written with the question, from the relevant section):\n"
        f"{reference}\n\n"
        f"Passages the system retrieved:\n\n{_format_context(hits) or '(none)'}\n\n"
        f"The system's answer:\n{answer}\n\n"
        "Grade the answer. A claim counts as supported only if a passage states it; "
        "general knowledge that happens to be true does not count."
    )


def judge_answer(client, question: str, reference: str, hits: list[Hit], answer: str,
                 model: str) -> tuple[dict, object]:
    """One verdict (the schema above) and the response's usage."""
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt(question, reference, hits, answer)}],
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"judge stopped early (stop_reason={response.stop_reason})")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def summarize(rows: list[dict]) -> dict:
    """Rates over the judged questions (errors excluded)."""
    ok = [r for r in rows if "verdict" in r]
    n = len(ok) or 1

    def rate(test) -> float:
        return round(sum(1 for r in ok if test(r["verdict"])) / n, 3)

    return {
        "judged": len(ok),
        "errors": len(rows) - len(ok),
        "fully_supported": rate(lambda v: v["support"] == "all"),
        "mostly_or_fully_supported": rate(lambda v: v["support"] in ("all", "most")),
        "answers_question": rate(lambda v: v["answers_question"]),
        "matches_reference": rate(lambda v: v["matches_reference"]),
        "citations_valid": rate(lambda v: v["citations_valid"]),
        "says_not_found": rate(lambda v: v["says_not_found"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade RAG answers with Claude as the judge.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    parser.add_argument("--questions", default=None, help="Path to the eval set (.jsonl).")
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument("--json", default=None, help="Write per-question verdicts here.")
    args = parser.parse_args()

    if not has_credentials():
        sys.exit("Set ANTHROPIC_API_KEY: the judge needs Claude for both answering and grading.")
    import anthropic

    client = anthropic.Anthropic()
    store = VectorStore.load()
    questions = load_questions(args.questions)[: args.limit]
    rows = []
    tokens = {"answer": [0, 0], "judge": [0, 0]}     # [input, output]
    for i, q in enumerate(questions, start=1):
        row = {"question": q.question, "source": q.source, "section": q.section}
        try:
            hits = retrieve(q.question, store)
            result = generate(q.question, hits)
            row.update(answer=result["answer"], mode=result["mode"],
                       sources=[h.chunk.section for h in hits])
            if result["mode"] != "claude":
                raise RuntimeError(result.get("note", "no Claude answer"))
            tokens["answer"][0] += result["usage"]["input_tokens"]
            tokens["answer"][1] += result["usage"]["output_tokens"]
            verdict, usage = judge_answer(client, q.question, q.answer_hint, hits,
                                          result["answer"], args.judge_model)
            row["verdict"] = verdict
            tokens["judge"][0] += usage.input_tokens
            tokens["judge"][1] += usage.output_tokens
        except Exception as exc:  # noqa: BLE001 - record and continue; summarized as errors
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        status = row.get("verdict", {}).get("support", "ERROR")
        print(f"[{i}/{len(questions)}] {status:5s} {q.question[:70]}", file=sys.stderr, flush=True)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    for role, model in (("answer", config.GEN_MODEL), ("judge", args.judge_model)):
        t_in, t_out = tokens[role]
        price = PRICES.get(model)
        cost = f" ~${t_in / 1e6 * price[0] + t_out / 1e6 * price[1]:.2f}" if price else ""
        print(f"{role} calls ({model}): {t_in:,} input + {t_out:,} output tokens{cost}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
