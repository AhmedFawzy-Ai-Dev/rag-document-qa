"""Publish the web demo to a Hugging Face Space.

    python -c "from huggingface_hub import login; login()"   # once (needs HF PRO for Gradio)
    python -m docqa.ingest                       # the hybrid index (needs the transformer extra)
    python scripts/deploy_space.py --space YOUR_NAME/ask-sklearn-docs --set-secret

Uploads the app, the package, the scikit-learn corpus and the prebuilt index
(so the Space doesn't spend minutes embedding the guide on every restart), with
deploy/space/README.md as the Space card and deploy/space/requirements.txt as
its pinned requirements.

``--set-secret`` copies ANTHROPIC_API_KEY from *your* shell into the Space's
secrets (it is sent to Hugging Face and never printed). Without it, add the
secret yourself under the Space's Settings -> Variables and secrets; until then
the demo answers in extractive mode. Put a spending limit on that key: the
Space's visitors use it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def stage(target: Path) -> None:
    """Copy what the Space needs into ``target``."""
    shutil.copy(ROOT / "app.py", target / "app.py")
    shutil.copy(ROOT / "LICENSE", target / "LICENSE")
    shutil.copy(ROOT / "deploy" / "space" / "README.md", target / "README.md")
    shutil.copy(ROOT / "deploy" / "space" / "requirements.txt", target / "requirements.txt")
    shutil.copytree(ROOT / "src" / "docqa", target / "src" / "docqa",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "data" / "sklearn", target / "data" / "sklearn")
    shutil.copy(ROOT / "data" / "index.joblib", target / "data" / "index.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the demo to a Hugging Face Space.")
    parser.add_argument("--space", required=True, help="Space id, e.g. your-name/ask-sklearn-docs")
    parser.add_argument("--private", action="store_true", help="Create the Space as private.")
    parser.add_argument("--set-secret", action="store_true",
                        help="Store ANTHROPIC_API_KEY from this shell as the Space's secret.")
    parser.add_argument("--max-questions", type=int, default=20,
                        help="Questions per chat session on the Space (RAG_MAX_QUESTIONS).")
    args = parser.parse_args()

    from docqa import config
    from docqa.store import VectorStore

    try:
        store = VectorStore.load()
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"{exc}\nBuild the index first: python -m docqa.ingest")
    if store.backend != "hybrid" or config.DOCS_DIR.name != "sklearn":
        sys.exit(f"The index uses the '{store.backend}' backend over {config.DOCS_DIR}; the Space "
                 "needs the hybrid index of data/sklearn. Install the transformer extra and run "
                 "`python -m docqa.ingest --backend hybrid`.")
    from docqa.chunk import load_and_chunk

    if [c.text for c in load_and_chunk()] != [c.text for c in store.chunks]:
        sys.exit("The index is stale (data/sklearn or the chunker changed since it was built). "
                 "Rebuild it: python -m docqa.ingest")
    if args.set_secret and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--set-secret needs ANTHROPIC_API_KEY set in this shell.")

    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]        # fails early with a clear error if not logged in
    print(f"Logged in to Hugging Face as {user}.")
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True,
                    private=args.private)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp,
                          commit_message="Deploy from rag-document-qa")
    api.add_space_variable(args.space, "RAG_MAX_QUESTIONS", str(args.max_questions))
    if args.set_secret:
        api.add_space_secret(args.space, "ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY"],
                             description="Claude API key for the demo's answers")
        print("Stored ANTHROPIC_API_KEY as a Space secret.")
    else:
        print("No API key set on the Space yet: add ANTHROPIC_API_KEY under Settings -> "
              "Variables and secrets (until then it answers in extractive mode).")
    print(f"Done: https://huggingface.co/spaces/{args.space} (the first build takes a few minutes)")


if __name__ == "__main__":
    main()
