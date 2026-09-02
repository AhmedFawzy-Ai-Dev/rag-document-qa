# Data

**`docs/`** holds the knowledge base the app answers from — a small, committed
set of Markdown notes on core ML topics, so the repo works straight from a clone:

| file | topic |
|---|---|
| `data_leakage.md` | what data leakage is and how to detect it |
| `transfer_learning.md` | feature extraction vs. fine-tuning |
| `evaluation_metrics.md` | why accuracy misleads; precision / recall / F1 |
| `rag_systems.md` | how Retrieval-Augmented Generation works |

**Use your own documents:** drop `.md` or `.txt` files into `docs/` (delete the
samples if you like) and rebuild the index:

```bash
python -m docqa.ingest
```

**`index.joblib`** (git-ignored) is the built retrieval index — regenerate it any
time with the command above.
