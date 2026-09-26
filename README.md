# RAG Document Q&A — Ask Your Docs

> A small but complete **Retrieval-Augmented Generation** app: it retrieves the
> most relevant passages from your documents and answers questions with
> **Claude**, grounded in the sources and cited by number. Runs offline with an
> extractive fallback when no API key is set.

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
python -m docqa.ask "What is data leakage and how do you detect it?" --show-sources
```

With `ANTHROPIC_API_KEY` set, Claude synthesizes a grounded answer and cites the
passages (wording will vary — this is an illustrative example of the format):

```
A: Data leakage is when information unavailable at prediction time leaks into
training, so a model looks great in evaluation but fails in production [1].
Detect it with a label-permutation test and a near-duplicate audit across the
split [1].

[mode: claude]
```

With **no key**, the app answers in extractive mode (verbatim from the docs) —
this is the exact, reproducible output you get offline:

```
Q: What is data leakage and how do you detect it?

A: Data Leakage in Machine Learning. Data leakage happens when information that would not be available at prediction
time leaks into the training process, producing models that look excellent in
evaluation but fail in production.

(Extractive mode — showing the most relevant passage [1] from data_leakage.md. Set ANTHROPIC_API_KEY for a synthesized answer.)

[mode: extractive]

Sources (most relevant first):
  [1] data_leakage.md  (score 0.4002)
```

The repo ships a tiny ML knowledge base in [`data/docs/`](data/docs) so it works
the moment you clone it — drop your own `.md` / `.txt` files there and re-ingest.

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

1. **Chunk** (`chunk.py`) — split docs into overlapping passages; markdown
   headings are merged into their content so every chunk carries context.
2. **Embed** (`embed.py`) — two interchangeable backends:
   - **`tfidf`** (default) — scikit-learn TF-IDF. No downloads, fully local, and
     a genuinely strong retrieval baseline.
   - **`transformer`** (optional) — mean-pooled BERT embeddings for semantic
     retrieval beyond exact word matches.
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
python -m docqa.ingest                       # build the index (TF-IDF, instant)
python -m docqa.ask "When does transfer learning help?" --show-sources
pytest -q
```

**Enable Claude answers** — set your key (the app reads it from the environment
and never stores it):

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # Windows: setx ANTHROPIC_API_KEY ...
python -m docqa.ask "Why is accuracy alone misleading?"
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

Passages scoring below a relevance threshold are dropped, so an off-topic
question gets "I couldn't find anything relevant" instead of an answer built
from unrelated text (and no Claude call is made). The TF-IDF default is `0.1`;
the transformer backend doesn't filter by default since its scores use a
different scale. Tune it with `RAG_MIN_SCORE`.

**All settings** (environment variables, see `src/docqa/config.py`):

| variable | default | what it does |
|---|---|---|
| `RAG_MODEL` | `claude-opus-5` | Claude model for answers |
| `RAG_MAX_TOKENS` | `16000` | answer token cap (thinking counts toward it) |
| `RAG_BACKEND` | `tfidf` | `tfidf` or `transformer` |
| `RAG_TOP_K` | `4` | passages retrieved per question |
| `RAG_MIN_SCORE` | `0.1` (TF-IDF) | relevance threshold |
| `RAG_EMBED_MODEL` | `google/bert_uncased_L-4_H-256_A-4` | transformer embedding model |
| `RAG_EMBED_TOKENIZER` | the model's own | tokenizer override for the transformer backend |
| `RAG_DATA_DIR` | `<repo>/data` | folder holding `docs/` and `index.joblib`; set it when installed with a regular `pip install .` |

---

## Project structure

```
rag-document-qa/
├── data/docs/*.md               # the knowledge base (swap in your own)
├── src/docqa/
│   ├── config.py                # paths, backend, model, chunk params
│   ├── chunk.py                 # document -> overlapping chunks
│   ├── embed.py                 # TF-IDF and transformer backends
│   ├── store.py                 # build / persist / cosine-search index
│   ├── generate.py              # Claude answer + extractive fallback
│   ├── pipeline.py              # retrieve -> generate
│   ├── ingest.py  ask.py        # CLIs
├── app.py                       # optional Gradio UI
├── tests/  .github/workflows/  pyproject.toml
```

## Design notes & honesty

- **TF-IDF is the default on purpose.** Sparse retrieval is dependency-free,
  instant, and hard to beat on small corpora; dense embeddings are one flag away
  when semantics matter. Real systems often use both (hybrid).
- **In-memory store.** Fine for a small knowledge base; the `store.search`
  interface swaps cleanly to FAISS or a vector DB for scale.
- **Extractive fallback is a feature, not a stub** — it keeps the app runnable
  and CI green without an API key, and gives a sensible answer offline.
- **Keys are never persisted** — the Anthropic SDK reads `ANTHROPIC_API_KEY`
  from the environment; this code doesn't log or store it.

## License

[MIT](LICENSE) © Ahmed Fawzy Yahya
