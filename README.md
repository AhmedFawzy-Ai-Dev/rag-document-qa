# RAG Document Q&A — Ask Your Docs

> A small but complete **Retrieval-Augmented Generation** app: it retrieves the
> most relevant passages from your documents and answers questions with
> **Claude**, grounded in the sources and cited by number. Runs offline with an
> extractive fallback when no API key is set. Ships with the full
> **scikit-learn user guide** (44 pages, ~140k words) as its knowledge base and a
> 100-question retrieval eval.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Claude](https://img.shields.io/badge/LLM-Claude-6b57ff)
![scikit-learn](https://img.shields.io/badge/retrieval-TF--IDF%20%2F%20embeddings-orange)
![CI](https://github.com/AhmedFawzy-Ai-Dev/rag-document-qa/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)

---

## What it does

Ask a question over a folder of documents and get a grounded, cited answer:

```bash
python -m docqa.ask "How do I keep class proportions equal across CV folds?" --show-sources
```

With `ANTHROPIC_API_KEY` set, Claude synthesizes a grounded answer and cites the
passages (wording will vary — this is an illustrative example of the format):

```
A: If class proportions must be balanced across folds while keeping groups
together, use StratifiedGroupKFold instead of GroupKFold [1].

[mode: claude]
```

With **no key**, the app answers in extractive mode (verbatim from the docs) —
this is the exact, reproducible output you get offline with the default TF-IDF
retrieval:

```
Q: How do I keep class proportions equal across CV folds?

A: Cross-validation: evaluating estimator performance > Cross validation iterators > Cross-validation iterators for grouped data > Group K-fold. Each subject is in a different testing fold, and the same subject is never in
both testing and training. Notice that the folds do not have exactly the same
size due to the imbalance in the data. If class proportions must be balanced
across folds, `StratifiedGroupKFold` is a better option.

Here is a visualization of the cross-validation behavior.

Similar to `KFold`, the test sets from `GroupKFold` will form a
complete partition of all the data.

While `GroupKFold` attempts to place the same number of samples in each
fold when `shuffle=False`, when `shuffle=True` it attempts to place an equal
number of distinct groups in each fold (but does not account for group sizes).

(Extractive mode — showing the most relevant passage [1] from cross_validation.md. Set ANTHROPIC_API_KEY for a synthesized answer.)

[mode: extractive]

Sources (most relevant first):
  [1] cross_validation.md  (score 0.1832)
  [2] cross_validation.md  (score 0.0923)
  [3] cross_validation.md  (score 0.0695)
  [4] preprocessing.md  (score 0.0642)
```

Note what TF-IDF did: it ranked the *Group K-fold* section first because that
section happens to say "class proportions", while the real answer is
*Stratified K-fold*. Word overlap alone misses what the question means; closing
that gap is what the retrieval upgrades below are measured on.

The knowledge base is the scikit-learn user guide, committed in
[`data/sklearn/`](data/sklearn) so the app works the moment you clone it. Point
`RAG_DOCS_DIR` (or `ingest --docs`) at your own `.md` / `.txt` files to use
those instead; see [`data/README.md`](data/README.md).

---

## How it works

A textbook RAG pipeline, kept small enough to read in one sitting:

```
documents ──► chunk ──► embed ──► vector store
                                       │
question ─────────────► embed ─► cosine search ─► top-k passages
                                                        │
                                          Claude (grounded + cited)
                                                        │
                                                    answer [n]
```

1. **Chunk** (`chunk.py`) — split docs along their headings, packing short
   paragraphs of the same section together (~120 words per chunk, long ones
   windowed with overlap). Every chunk starts with its heading path, e.g.
   *Cross-validation > Cross validation iterators > K-fold*, so it keeps its
   context wherever it lands.
2. **Embed** (`embed.py`) — two interchangeable backends:
   - **`tfidf`** (default) — scikit-learn TF-IDF. No downloads, fully local, and
     a genuinely strong retrieval baseline.
   - **`transformer`** (optional) — dense [sentence-transformers](https://sbert.net)
     embeddings (default `BAAI/bge-small-en-v1.5`) for semantic retrieval beyond
     exact word matches.
3. **Retrieve** (`store.py`) — cosine similarity over the chunk vectors.
4. **Generate** (`generate.py`) — the retrieved passages are put in the prompt
   and **Claude** answers using only them, citing sources. No key? It falls back
   to an **extractive** answer (the top passages) so the app still runs.

**Why grounding matters:** answers are tied to retrieved evidence (less
hallucination), stay current without retraining, and are auditable through
citations.

---

## Quickstart

```bash
pip install -e ".[dev]"
python -m docqa.ingest                       # build the index (TF-IDF, seconds)
python -m docqa.ask "How do I keep class proportions equal across CV folds?" --show-sources
python -m docqa.evaluate                     # retrieval metrics on the eval set
pytest -q
```

**Enable Claude answers** — set your key (the app reads it from the environment
and never stores it):

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # Windows: setx ANTHROPIC_API_KEY ...
python -m docqa.ask "Why can accuracy be misleading on imbalanced data?"
```

**Semantic (embedding) retrieval** and the **web UI** are optional extras:

```bash
pip install -e ".[transformer]" && python -m docqa.ingest --backend transformer
pip install -e ".[ui]" && python app.py      # Gradio chat demo
```

The generation model defaults to **`claude-opus-5`**; override with `RAG_MODEL`
(e.g. `RAG_MODEL=claude-haiku-4-5` for a cheaper, faster answer). If a Claude
call fails (bad key, unknown model, network), the app still answers extractively
and prints the reason as a `Note:`.

Passages scoring below a relevance threshold are dropped, so a question that
shares no vocabulary with the docs gets "I couldn't find anything relevant"
(and no Claude call is made). The threshold is deliberately low (TF-IDF `0.05`,
none for the transformer backend): on the scikit-learn corpus, similarity scores
of on- and off-topic questions overlap, so a stricter cut would also suppress
genuine questions. Claude's instruction to say when the passages don't answer
the question handles the rest. Tune it with `RAG_MIN_SCORE`.

**All settings** (environment variables, see `src/docqa/config.py`):

| variable | default | what it does |
|---|---|---|
| `RAG_MODEL` | `claude-opus-5` | Claude model for answers |
| `RAG_MAX_TOKENS` | `16000` | answer token cap (thinking counts toward it) |
| `RAG_BACKEND` | `tfidf` | `tfidf` or `transformer` |
| `RAG_TOP_K` | `4` | passages retrieved per question |
| `RAG_MIN_SCORE` | `0.05` (TF-IDF) | relevance threshold |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | any sentence-transformers model, for the transformer backend |
| `RAG_DOCS_DIR` | `<data dir>/sklearn` | the documents to index |
| `RAG_DATA_DIR` | `<repo>/data` | folder holding the corpus, eval set and `index.joblib`; set it when installed with a regular `pip install .` |

---

## Evaluation

`data/sklearn_eval.jsonl` holds 100 questions about the scikit-learn user guide,
each labeled with the page and section that answers it: 49 phrased with the
guide's own terms ("keyword") and 51 describing the need in other words
("paraphrase"). They were written by Claude from the section text and
spot-checked against it.

```bash
python -m docqa.evaluate            # every installed backend
```

A retrieved chunk is a **page hit** if it comes from the right page and a
**section hit** if it also sits in the right section; **MRR** is the mean of
1/rank of the first section hit. Current results (1,003 chunks):

| backend | page@1 | page@5 | section@1 | section@5 | MRR | paraphrase section@5 | ms/query |
|---|---|---|---|---|---|---|---|
| TF-IDF | 0.76 | 0.94 | 0.64 | 0.78 | 0.70 | 0.57 | 4 |
| bge-small (dense) | 0.84 | 0.89 | 0.65 | 0.77 | 0.70 | 0.55 | 33 |

Dense retrieval finds the right *page* more often; TF-IDF is as good at the
right *section* and slightly better on paraphrases. Neither is better
overall, which is the case for combining them.

**Chunking was chosen on this set too** (MRR, TF-IDF / bge-small):

| chunking | chunks | MRR |
|---|---|---|
| one paragraph per chunk | 4,358 | 0.59 / 0.67 |
| sections packed to 80 words | 2,504 | 0.63 / 0.68 |
| sections packed to 120 words | 1,642 | 0.63 / 0.66 |
| **sections packed to 200 words** | **1,003** | **0.70 / 0.70** |
| sections packed to 250 / 300 words | 854 / 755 | 0.67 / 0.69 (TF-IDF) |

## Project structure

```
rag-document-qa/
├── data/
│   ├── sklearn/*.md             # the knowledge base: scikit-learn user guide
│   ├── sklearn_eval.jsonl       # 100 labeled questions (page + section)
│   └── ml_notes/*.md            # tiny corpus for fast tests
├── scripts/fetch_sklearn_docs.py  # rebuilds data/sklearn from a pinned release
├── src/docqa/
│   ├── config.py                # paths, backend, model, chunk params
│   ├── chunk.py                 # heading-aware chunking
│   ├── embed.py                 # TF-IDF and transformer backends
│   ├── store.py                 # build / persist / cosine-search index
│   ├── generate.py              # Claude answer + extractive fallback
│   ├── pipeline.py              # retrieve -> generate
│   ├── evaluate.py              # retrieval metrics on the eval set
│   ├── ingest.py  ask.py        # CLIs
├── app.py                       # optional Gradio UI
├── tests/  .github/workflows/  pyproject.toml
```

## Design notes & honesty

- **TF-IDF is the default on purpose.** Sparse retrieval is dependency-free,
  instant, and hard to beat on small corpora; dense embeddings are one flag away
  when semantics matter. Real systems often use both (hybrid).
- **Why bge-small.** On the 24 labeled questions in `tests/test_rag.py` (12
  keyword, 12 paraphrased), recall@1 was: TF-IDF 12 + 9, `all-MiniLM-L6-v2`
  12 + 11, `bge-small-en-v1.5` 12 + 12. The mean-pooled BERT-mini that this
  backend used before scored 9 + 8, below TF-IDF.
- **In-memory store.** Fine for a small knowledge base; the `store.search`
  interface swaps cleanly to FAISS or a vector DB for scale.
- **Extractive fallback is a feature, not a stub** — it keeps the app runnable
  and CI green without an API key, and gives a sensible answer offline.
- **Keys are never persisted** — the Anthropic SDK reads `ANTHROPIC_API_KEY`
  from the environment; this code doesn't log or store it.

## License

[MIT](LICENSE) © Ahmed Fawzy Yahya
