# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation grounds a language model's answers in an external
knowledge source. Instead of relying only on what the model memorised during
training, a RAG system retrieves relevant passages at query time and asks the
model to answer using them, with citations.

## The pipeline

1. **Ingest & chunk.** Documents are split into passages small enough to be
   specific but large enough to be self-contained (often with a little overlap).
2. **Index.** Each chunk is converted into a vector — either a sparse TF-IDF
   vector or a dense embedding from a neural encoder — and stored.
3. **Retrieve.** The query is embedded the same way, and the most similar chunks
   are found by cosine similarity (dense) or term overlap (sparse).
4. **Generate.** The retrieved chunks are placed in the prompt as context, and
   the language model writes an answer grounded in them, citing its sources.

## Why it matters

RAG reduces hallucination because answers are tied to retrieved evidence, keeps
knowledge up to date without retraining the model, and makes answers auditable
through citations. Sparse retrieval (TF-IDF / BM25) is a strong, dependency-free
baseline; dense embeddings capture meaning beyond exact word matches; hybrids
combine both.
