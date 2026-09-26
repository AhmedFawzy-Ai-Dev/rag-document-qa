"""Retrieval evaluation against a labeled question set.

    python -m docqa.evaluate                          # every method installed
    python -m docqa.evaluate --method bm25 --method hybrid+rerank --json results.json

Each question in the eval set names the page (``source``) and the section
(``section``, a heading path) that answers it. A retrieved chunk is a
*page hit* if it comes from that page, and a *section hit* if it also sits in
that section (or a subsection of it). For each backend this reports hit@k at
both levels and the mean reciprocal rank of the first section hit, overall and
per question style ("keyword" vs "paraphrase").

Methods: ``tfidf``, ``bm25``, ``dense`` (bge-small), ``hybrid`` (BM25 + dense
fused with RRF), and each of those with ``+rerank`` (cross-encoder on the top
candidates). The dense ones need the `transformer` extra. They share one
hybrid index, so the corpus is embedded only once.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .chunk import Chunk
from .store import VectorStore, build_index

KS = (1, 3, 5)


@dataclass
class Question:
    question: str
    source: str
    section: str
    style: str = "keyword"


def load_questions(path: Path | str | None = None) -> list[Question]:
    path = Path(path) if path is not None else config.EVAL_PATH
    fields = Question.__dataclass_fields__
    with open(path, encoding="utf-8") as f:
        return [Question(**{k: v for k, v in json.loads(line).items() if k in fields})
                for line in f if line.strip()]


def page_hit(chunk: Chunk, q: Question) -> bool:
    return chunk.source == q.source


def section_hit(chunk: Chunk, q: Question) -> bool:
    return page_hit(chunk, q) and (
        chunk.section == q.section or chunk.section.startswith(q.section + " > ")
    )


def score(ranked: list[list[Chunk]], questions: list[Question]) -> dict:
    """Metrics for ranked results (one list of chunks per question)."""
    n = len(questions)
    out: dict[str, float] = {"n": n}
    for k in KS:
        out[f"page@{k}"] = sum(any(page_hit(c, q) for c in r[:k])
                               for r, q in zip(ranked, questions)) / n
        out[f"section@{k}"] = sum(any(section_hit(c, q) for c in r[:k])
                                  for r, q in zip(ranked, questions)) / n
    rr = []
    for r, q in zip(ranked, questions):
        rank = next((i for i, c in enumerate(r, start=1) if section_hit(c, q)), None)
        rr.append(1 / rank if rank else 0.0)
    out["mrr"] = sum(rr) / n
    return out


def evaluate(store: VectorStore, questions: list[Question], k: int = max(KS), *,
             scorers: list[str] | None = None, rerank: bool = False) -> dict:
    """Overall and per-style metrics for one retrieval method, plus mean latency."""
    t = time.perf_counter()
    ranked = [[h.chunk for h in store.search(q.question, k, rerank=rerank, scorers=scorers)]
              for q in questions]
    latency_ms = (time.perf_counter() - t) / len(questions) * 1000
    result = {"all": score(ranked, questions), "latency_ms": round(latency_ms, 1)}
    for style in sorted({q.style for q in questions}):
        idx = [i for i, q in enumerate(questions) if q.style == style]
        result[style] = score([ranked[i] for i in idx], [questions[i] for i in idx])
    return result


def format_table(results: dict[str, dict]) -> str:
    """A Markdown table: one row per backend."""
    cols = ["page@1", "page@5", "section@1", "section@3", "section@5", "mrr"]
    lines = [
        "| method | " + " | ".join(cols) + " | paraphrase section@5 | ms/query |",
        "|---" * (len(cols) + 3) + "|",
    ]
    for name, r in results.items():
        cells = [f"{r['all'][c]:.2f}" for c in cols]
        para = r.get("paraphrase", {}).get("section@5")
        cells.append(f"{para:.2f}" if para is not None else "–")
        cells.append(f"{r['latency_ms']:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# method -> (index backend, scorers to use, rerank)
METHODS = {
    "tfidf": ("tfidf", ["tfidf"], False),
    "bm25": ("hybrid", ["bm25"], False),
    "dense": ("hybrid", ["transformer"], False),
    "hybrid": ("hybrid", None, False),
    "bm25+rerank": ("hybrid", ["bm25"], True),
    "dense+rerank": ("hybrid", ["transformer"], True),
    "hybrid+rerank": ("hybrid", None, True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a labeled question set.")
    parser.add_argument("--method", action="append", choices=list(METHODS),
                        help="Method(s) to evaluate (default: every one installed).")
    parser.add_argument("--questions", default=None, help="Path to the eval set (.jsonl).")
    parser.add_argument("--docs", default=None, help="Documents directory.")
    parser.add_argument("--json", default=None, help="Also write the full results here.")
    args = parser.parse_args()

    methods = args.method or (list(METHODS) if config.dense_available() else ["tfidf", "bm25"])
    # Without the extra, BM25 gets its own BM25-only index instead of the hybrid one.
    backend_for = {m: ("bm25" if METHODS[m][0] == "hybrid" and not config.dense_available()
                       else METHODS[m][0]) for m in methods}
    stores = {b: build_index(docs_dir=args.docs, backend=b, save=False)
              for b in dict.fromkeys(backend_for.values())}
    questions = load_questions(args.questions)
    results = {}
    for m in methods:
        _, scorers, rerank = METHODS[m]
        results[m] = evaluate(stores[backend_for[m]], questions, scorers=scorers, rerank=rerank)
        print(f"evaluated {m}", file=sys.stderr, flush=True)
    n_chunks = len(next(iter(stores.values())).chunks)
    print(f"{len(questions)} questions, {n_chunks} chunks\n")
    print(format_table(results))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
