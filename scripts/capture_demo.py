"""Record the README's demo media from a running app (a dev tool; needs Playwright).

    python app.py                       # in another terminal; set ANTHROPIC_API_KEY there
    pip install playwright
    python scripts/capture_demo.py      # -> docs/images/demo.gif and demo_*.png
    python scripts/capture_demo.py --url http://localhost:7861   # another port

Drives the app in the installed Microsoft Edge (Playwright's "msedge" channel, so
no browser download), asks the questions below, and assembles the frames taken
while the first answer streams into a GIF.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How does gradient boosting handle missing values?"
FOLLOW_UP = "Does it treat categorical features the same way?"
PDF = ROOT / "tests" / "data" / "model_cards.pdf"
PDF_QUESTION = "How should a model card report fairness?"
OFF_TOPIC = "Who won the 2022 World Cup?"


def wait_for_answer(page, n_bots: int, frames: list | None = None, timeout: float = 240) -> None:
    """Wait until the n-th bot message stops changing (optionally grabbing frames)."""
    bots = page.get_by_test_id("bot")
    start, last, stable = time.time(), None, 0
    while time.time() - start < timeout:
        if frames is not None:
            frames.append((page.screenshot(), 400))
        if bots.count() >= n_bots:
            text = bots.nth(n_bots - 1).inner_text()
            stable = stable + 1 if text == last and not text.rstrip().endswith("▌") else 0
            if stable >= 6 and text.strip():   # ~3 s without change
                return
            last = text
        time.sleep(0.5)
    raise TimeoutError("the answer did not finish in time")


def ask(page, question: str, frames: list | None = None) -> None:
    """Type a question (filmed a few characters at a time when frames is given)."""
    box = page.get_by_test_id("textbox")
    box.wait_for(timeout=60000)
    if frames is None:
        box.fill(question)
    else:
        for i in range(3, len(question) + 3, 3):
            box.fill(question[:i])
            frames.append((page.screenshot(), 90))
    box.press("Enter")


# Scrolling happens inside the chat box only; the page itself stays at the top.
_ALIGN = """(e) => { window.scrollTo(0, 0); let s = e.parentElement;
  while (s && s.scrollHeight <= s.clientHeight + 2) s = s.parentElement;
  if (s) s.scrollTop += e.getBoundingClientRect().top - s.getBoundingClientRect().top - 8; }"""
_STEP = """(e, step) => { let s = e.parentElement;
  while (s && s.scrollHeight <= s.clientHeight + 2) s = s.parentElement;
  if (!s) return false; const before = s.scrollTop; s.scrollTop += step;
  return s.scrollTop > before; }"""
_BOTTOM = """(e) => { let s = e.parentElement;
  while (s && s.scrollHeight <= s.clientHeight + 2) s = s.parentElement;
  if (s) s.scrollTop = s.scrollHeight; window.scrollTo(0, 0); }"""


def show_answer(page, n: int) -> None:
    """Scroll the chat so the n-th question and the start of its answer are in view."""
    page.get_by_test_id("user").nth(n - 1).evaluate(_ALIGN)
    time.sleep(0.4)


def to_bottom(page) -> None:
    """Back to the latest message, so the chat keeps following new answers."""
    page.get_by_test_id("bot").last.evaluate(_BOTTOM)
    time.sleep(0.3)


def scroll_through_answer(page, n: int, frames: list, step: int = 140) -> None:
    show_answer(page, n)
    frames.append((page.screenshot(), 1600))
    bot = page.get_by_test_id("bot").nth(n - 1)
    for _ in range(40):
        if not bot.evaluate(_STEP, step):
            break
        time.sleep(0.15)
        frames.append((page.screenshot(), 180))


def save_gif(frames: list[tuple[bytes, int]], path: Path, width: int = 960) -> None:
    from PIL import Image

    images, durations, previous = [], [], None
    for raw, duration in frames:
        if raw == previous:                    # merge identical frames
            durations[-1] += duration
            continue
        previous = raw
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((width, round(img.height * width / img.width)))
        images.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        durations.append(duration)
    durations[-1] = 4000                       # hold the end
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True)


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description="Record the README demo media.")
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--out", default=str(ROOT / "docs" / "images"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1.5)
        for attempt in range(40):          # the app may still be loading its models
            try:
                page.goto(args.url, timeout=10000)
                break
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(5)
        page.get_by_test_id("textbox").wait_for(timeout=180000)

        frames: list = [(page.screenshot(), 1200)]
        ask(page, QUESTION, frames)
        wait_for_answer(page, 1, frames)
        scroll_through_answer(page, 1, frames)
        save_gif(frames, out / "demo.gif")
        show_answer(page, 1)
        page.screenshot(path=out / "demo_answer.png")

        for n, (question, name) in enumerate(
                [(FOLLOW_UP, "demo_followup.png"), (OFF_TOPIC, "demo_offtopic.png")], start=2):
            to_bottom(page)
            ask(page, question)
            wait_for_answer(page, n)
            show_answer(page, n)
            page.screenshot(path=out / name)

        to_bottom(page)
        page.get_by_test_id("file-upload").set_input_files(str(PDF))
        page.get_by_text("Indexed").wait_for(timeout=180000)
        ask(page, PDF_QUESTION)
        wait_for_answer(page, 4)
        show_answer(page, 4)
        page.screenshot(path=out / "demo_pdf.png")
        browser.close()
    print(f"Wrote demo.gif and demo_answer/followup/offtopic/pdf.png to {out}")


if __name__ == "__main__":
    main()
