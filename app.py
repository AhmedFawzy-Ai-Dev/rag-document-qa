"""Optional Gradio web UI for the RAG document-Q&A app.

    pip install ".[ui]"
    python app.py

Builds the index on first launch (if missing) and answers questions in a chat
box, showing which sources were retrieved. Uses Claude when ANTHROPIC_API_KEY is
set, otherwise the extractive fallback.
"""
from __future__ import annotations

from docqa import config
from docqa.pipeline import answer
from docqa.store import VectorStore, build_index


def _load_store() -> VectorStore:
    try:
        return VectorStore.load()
    except FileNotFoundError:
        return build_index(save=True)


def main() -> None:
    import gradio as gr

    store = _load_store()

    def respond(question: str) -> tuple[str, str]:
        if not question.strip():
            return "", ""
        result = answer(question, store=store)
        srcs = "\n".join(
            f"[{s['n']}] {s['source']} (score {s['score']})" for s in result["sources"]
        )
        note = f"\n\n**Note:** {result['note']}" if "note" in result else ""
        return f"{result['answer']}\n\n_mode: {result['mode']}_{note}", srcs

    with gr.Blocks(title="Ask Your Docs (RAG)") as demo:
        gr.Markdown("# Ask Your Docs\nRetrieval-Augmented Q&A over a small ML knowledge base.")
        q = gr.Textbox(label="Question", placeholder="What is data leakage?")
        ask = gr.Button("Ask", variant="primary")
        ans = gr.Markdown(label="Answer")
        src = gr.Textbox(label="Retrieved sources", interactive=False)
        ask.click(respond, inputs=q, outputs=[ans, src])
        q.submit(respond, inputs=q, outputs=[ans, src])
        gr.Examples(
            ["What is data leakage?", "When does transfer learning help?",
             "Why is accuracy alone misleading?", "How does RAG reduce hallucination?"],
            inputs=q,
        )

    print(f"Backend: {config.EMBEDDING_BACKEND} | Generation model: {config.GEN_MODEL}")
    demo.launch()


if __name__ == "__main__":
    main()
