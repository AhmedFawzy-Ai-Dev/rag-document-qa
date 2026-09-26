"""Ask a question against the indexed documents.

    python -m docqa.ask "How does gradient boosting handle missing values?"
    python -m docqa.ask --show-sources "When should I use RobustScaler?"

With ANTHROPIC_API_KEY set, Claude's answer streams in as it is written and is
followed by the sentences it cites; otherwise an extractive fallback is shown.
"""
from __future__ import annotations

import argparse
import sys

from .pipeline import answer_stream

EXAMPLE = "How does gradient boosting handle missing values?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over your documents.")
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("--show-sources", action="store_true", help="Print the retrieved sources.")
    parser.add_argument("--k", type=int, default=None, help="How many passages to retrieve.")
    args = parser.parse_args()

    question = " ".join(args.question) or EXAMPLE
    print(f"Q: {question}\n")
    streamed = False
    for event in answer_stream(question, k=args.k):
        if event["type"] == "text":
            if not streamed:
                print("A: ", end="")
                streamed = True
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            result = event["result"]

    if not streamed:
        print(f"A: {result['answer']}\n")
    elif result["mode"] != "claude":   # the stream broke off; show the fallback instead
        print(f"\n\n(Answer interrupted.)\n\nA: {result['answer']}\n")
    else:
        print("\n")
    print(f"[mode: {result['mode']}]")
    if "note" in result:
        print(f"Note: {result['note']}", file=sys.stderr)
    if result.get("citations"):
        print("\nCitations (the exact sentences the answer relies on):")
        for c in result["citations"]:
            print(f'  [{c["n"]}] "{c["quote"]}"\n      — {c["source"]} > {c["section"]}')
    if args.show_sources:
        print("\nSources (most relevant first):")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['source']}  (score {s['score']})")


if __name__ == "__main__":
    main()
