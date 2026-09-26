"""Charts of the retrieval evaluation, for the README (needs the `report` extra).

    python -m docqa.evaluate --json results.json
    python -m docqa.report results.json --out docs/images

Writes ``retrieval_quality.png`` (MRR and paraphrase section@5 per method, as
two panels on their own axes) and ``quality_vs_latency.png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display names, in the order the methods were added (top to bottom).
LABELS = {
    "tfidf": "TF-IDF (baseline)",
    "bm25": "BM25",
    "dense": "Dense (bge-small)",
    "hybrid": "Hybrid (BM25 + dense)",
    "bm25+rerank": "BM25 + rerank",
    "dense+rerank": "Dense + rerank",
    "hybrid+rerank": "Hybrid + rerank",
}
HIGHLIGHT = "hybrid+rerank"      # the default pipeline

# One blue hue: light for the alternatives, dark for the default (validated as
# an ordinal pair); values are printed on every mark, so color is never alone.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BAR, BAR_HI = "#86b6ef", "#256abf"
# Label placement on the latency chart (points offset, alignment): the reranked
# methods and TF-IDF/dense sit close together.
LABEL_OFFSETS = {
    "tfidf": (0, -17, "center"),
    "dense": (8, -15, "left"),
    "bm25+rerank": (-10, 7, "right"),
    "dense+rerank": (-10, -15, "right"),
    "hybrid+rerank": (-10, 7, "right"),
}


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK_2, "font.size": 11,
    })


def quality_chart(results: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    methods = [m for m in LABELS if m in results]
    panels = [
        ("MRR, all 100 questions", lambda r: r["all"]["mrr"]),
        ("Right section in top 5, 51 paraphrased questions",
         lambda r: r["paraphrase"]["section@5"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    y = list(range(len(methods)))[::-1]
    for ax, (title, value) in zip(axes, panels):
        vals = [value(results[m]) for m in methods]
        colors = [BAR_HI if m == HIGHLIGHT else BAR for m in methods]
        ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        for yi, v, m in zip(y, vals, methods):
            ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=11,
                    color=INK if m == HIGHLIGHT else INK_2,
                    fontweight="bold" if m == HIGHLIGHT else "normal")
        ax.set_xlim(0, 1.0)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [LABELS[m] for m in methods])
    for label, m in zip(axes[0].get_yticklabels(), methods):
        if m == HIGHLIGHT:
            label.set_fontweight("bold")
            label.set_color(INK)
    fig.suptitle("Retrieval on 100 questions about the scikit-learn user guide",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, "MRR: mean of 1/rank of the first passage from the section that "
             "answers the question. Paraphrased questions avoid the docs' own terms.",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def latency_chart(results: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    methods = [m for m in LABELS if m in results]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in methods:
        x = max(results[m]["latency_ms"], 0.5)     # log axis: keep sub-ms methods visible
        yv = results[m]["all"]["mrr"]
        hi = m == HIGHLIGHT
        ax.scatter([x], [yv], s=110 if hi else 70, color=BAR_HI if hi else BAR,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy, ha = LABEL_OFFSETS.get(m, (8, 6, "left"))
        ax.annotate(LABELS[m], (x, yv), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=10, color=INK if hi else INK_2,
                    fontweight="bold" if hi else "normal")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"))
    ax.set_xlabel("Time per question (log scale, 4-core laptop CPU)")
    ax.set_ylabel("MRR")
    ax.set_ylim(0.6, 0.9)
    ax.set_xlim(0.3, 30000)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    title = "Quality vs. speed"
    if {"hybrid", HIGHLIGHT} <= results.keys():
        gain = results[HIGHLIGHT]["all"]["mrr"] - results["hybrid"]["all"]["mrr"]
        secs = results[HIGHLIGHT]["latency_ms"] / 1000
        title += f": the reranker buys {gain:+.2f} MRR for ~{secs:.0f} s per question"
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart the retrieval evaluation.")
    parser.add_argument("results", help="JSON written by `python -m docqa.evaluate --json`.")
    parser.add_argument("--out", default="docs/images", help="Output directory.")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quality_chart(results, out / "retrieval_quality.png")
    latency_chart(results, out / "quality_vs_latency.png")
    print(f"Wrote {out / 'retrieval_quality.png'} and {out / 'quality_vs_latency.png'}")


if __name__ == "__main__":
    main()
