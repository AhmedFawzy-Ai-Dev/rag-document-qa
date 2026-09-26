"""Gradio web UI: chat with the scikit-learn user guide, or with your own files.

    pip install -e ".[ui,transformer]"
    python app.py

Answers stream in as they are written and end with the exact sentences they
cite. Upload PDFs / Markdown / text files to ask about those instead. Follow-up
questions are rewritten into standalone ones before searching. Uses Claude when
ANTHROPIC_API_KEY is set, otherwise the extractive fallback.
"""
from __future__ import annotations

from docqa import config
from docqa.chat import index_files, respond
from docqa.store import VectorStore, build_index

GUIDE, UPLOADS = "scikit-learn user guide", "My uploaded files"
EXAMPLES = [
    "How does gradient boosting handle missing values?",
    "When should I use RobustScaler instead of StandardScaler?",
    "How do I keep class proportions equal across CV folds?",
    "What's the difference between bagging and boosting?",
]


def _load_store() -> VectorStore:
    try:
        return VectorStore.load()
    except FileNotFoundError:
        return build_index(save=True)


def main() -> None:
    import gradio as gr

    guide = _load_store()
    pipeline = (f"{guide.backend} retrieval"
                + (" + cross-encoder reranking" if config.rerank_enabled() else "")
                + f" · answers by {config.GEN_MODEL} with verified citations")

    def upload(files, current):
        if not files:
            return current, "No files uploaded.", gr.update()
        paths = [getattr(f, "name", f) for f in files]
        try:
            store = index_files(paths)
        except ValueError as exc:
            return current, f"⚠️ {exc}", gr.update()
        names = ", ".join(sorted({c.source for c in store.chunks}))
        return (store, f"Indexed **{len(store.chunks)} passages** from {names}.",
                gr.update(value=UPLOADS))

    def chat(message, history, source, uploads):
        store = uploads if source == UPLOADS else guide
        if store is None:
            yield "Upload some files first (PDF, Markdown or text), or switch back to the guide."
            return
        yield from respond(message, history, store)

    with gr.Blocks(title="Ask the Docs · RAG") as demo:
        gr.Markdown(
            "# Ask the Docs\n"
            "Ask anything about the **scikit-learn user guide** (44 pages), or upload your "
            f"own PDFs. <sub>{pipeline}</sub>"
        )
        uploads = gr.State(None)
        with gr.Row():
            source = gr.Radio([GUIDE, UPLOADS], value=GUIDE, label="Search in", scale=1)
            files = gr.File(label="Upload PDF / .md / .txt", file_count="multiple",
                            file_types=[".pdf", ".md", ".txt"], scale=2)
        status = gr.Markdown()
        gr.ChatInterface(
            chat,
            additional_inputs=[source, uploads],
            examples=[[q, GUIDE, None] for q in EXAMPLES],
            chatbot=gr.Chatbot(height=520, render_markdown=True),
        )
        files.upload(upload, inputs=[files, uploads], outputs=[uploads, status, source])

    print(f"Pipeline: {pipeline}")
    demo.launch()


if __name__ == "__main__":
    main()
