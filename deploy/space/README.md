---
title: Ask the scikit-learn Docs
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: RAG over the scikit-learn guide with verified citations
---

# Ask the scikit-learn Docs

Ask anything about the full **scikit-learn user guide** (44 pages), or upload
your own PDFs, and get an answer from Claude that quotes the exact sentences it
relies on.

Under the hood: heading-aware chunking, **hybrid retrieval** (BM25 + dense
`bge-small` embeddings fused with reciprocal rank fusion), a **cross-encoder
reranker**, and Claude's **Citations API**, so every citation is guaranteed to
quote a retrieved passage. On a 100-question eval, this raised retrieval MRR from
0.70 (TF-IDF) to 0.85.

Code, evaluation and write-up:
[github.com/AhmedFawzy-Ai-Dev/rag-document-qa](https://github.com/AhmedFawzy-Ai-Dev/rag-document-qa)

*This demo runs on a free CPU, so answers take a few seconds, and each session
is limited to a few questions to keep API costs in check.*
