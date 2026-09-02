"""Ask a question against the indexed documents.

    python -m docqa.ask "What is data leakage?"
    python -m docqa.ask --show-sources "How does RAG reduce hallucination?"

Uses Claude when ANTHROPIC_API_KEY is set, otherwise an extractive fallback.
"""
from __future__ import annotations

import argparse

from .pipeline import answer

EXAMPLE = "What is data leakage and how do you detect it?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over your documents.")
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("--show-sources", action="store_true", help="Print the retrieved sources.")
    parser.add_argument("--k", type=int, default=None, help="How many passages to retrieve.")
    args = parser.parse_args()

    question = " ".join(args.question) or EXAMPLE
    result = answer(question, k=args.k)

    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(f"[mode: {result['mode']}]")
    if args.show_sources:
        print("\nSources (most relevant first):")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['source']}  (score {s['score']})")


if __name__ == "__main__":
    main()
