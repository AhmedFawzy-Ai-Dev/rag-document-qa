"""Download the scikit-learn user guide and convert it to Markdown for the corpus.

    python scripts/fetch_sklearn_docs.py            # -> data/sklearn/

The source is pinned to a release tag so the corpus (and the eval results built
on it) are reproducible. scikit-learn's documentation is BSD-3-Clause licensed;
its license is copied next to the converted files.

The conversion keeps what matters for retrieval (headings, prose, math, code)
and drops Sphinx plumbing (labels, image options, module directives). It is a
pragmatic RST subset, not a full parser.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
import urllib.request
from pathlib import Path

TAG = "1.9.1"
REPO = "scikit-learn/scikit-learn"
RAW = f"https://raw.githubusercontent.com/{REPO}/{TAG}"
API = f"https://api.github.com/repos/{REPO}/contents/doc/modules?ref={TAG}"
OUT = Path(__file__).resolve().parents[1] / "data" / "sklearn"

HEADING_CHARS = "=-^~\"'*+#"
# Directives whose body is dropped entirely.
DROP = {"currentmodule", "image", "raw", "only", "plot", "autosummary", "include",
        "toctree", "testsetup", "testcleanup", "list-table"}
# Admonitions: keep the body, prefixed with a label.
LABEL = {"note": "Note", "warning": "Warning", "seealso": "See also", "topic": None,
         "dropdown": None, "rubric": None, "centered": None, "figure": None}
CODE = {"code-block", "code", "prompt", "doctest"}
# Roles that name API objects: shown as code, `~` keeps only the last component.
API_ROLES = {"class", "func", "meth", "mod", "attr", "obj", "data", "exc"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-document-qa"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def convert_inline(text: str) -> str:
    def role(m: re.Match) -> str:
        name, body = m.group(1), m.group(2)
        link = re.match(r"(.*?)\s*<(.+)>$", body, re.DOTALL)
        if link and link.group(1):
            body = link.group(1)
        elif link:
            body = link.group(2)
        if name.endswith("math"):
            return f"${body}$"
        if name in API_ROLES:
            body = body.lstrip("!")
            if body.startswith("~"):
                body = body[1:].split(".")[-1]
            return f"`{body}`"
        if name == "ref" and not link:
            body = body.replace("_", " ")
        return body

    text = re.sub(r":(?:py:)?([a-z]+(?::[a-z]+)?):`([^`]+)`", role, text)
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`__?", r"\1", text)   # `text <url>`_
    text = re.sub(r"`([^`]+)`_\b", r"\1", text)                # `target`_
    text = re.sub(r"``([^`]+)``", r"`\1`", text)              # ``code``
    text = re.sub(r"\[([A-Za-z0-9-]+)\]_", r"[\1]", text)      # citations
    text = re.sub(r"\|([A-Za-z_]+)\|", r"\1", text)            # substitutions
    return text


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def convert(rst: str) -> str:
    lines = rst.replace("\t", "    ").splitlines()
    out: list[str] = []
    levels: list[tuple[str, bool]] = []   # heading styles in order of appearance
    i = 0

    def heading(title: str, style: tuple[str, bool]) -> None:
        if style not in levels:
            levels.append(style)
        out.extend(["", "#" * (levels.index(style) + 1) + " " + convert_inline(title.strip()), ""])

    def block(start: int, base: int) -> tuple[list[str], int]:
        """Lines after `start` indented deeper than `base` (blank lines included)."""
        j, body = start, []
        while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > base):
            body.append(lines[j])
            j += 1
        while body and not body[-1].strip():
            body.pop()
        return body, j

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        # Overlined title: ====== / Title / ======
        if (stripped and set(stripped) <= set(HEADING_CHARS) and len(set(stripped)) == 1
                and len(stripped) >= 3 and i + 2 < len(lines)
                and lines[i + 2].strip() == stripped):
            heading(nxt, (stripped[0], True))
            i += 3
            continue
        # Underlined title: Title / ------
        if (stripped and not line.startswith(" ") and nxt.strip()
                and set(nxt.strip()) <= set(HEADING_CHARS) and len(set(nxt.strip())) == 1
                and len(nxt.strip()) >= max(3, len(stripped) - 2)):
            heading(line, (nxt.strip()[0], False))
            i += 2
            continue

        m = re.match(r"^(\s*)\.\. ([a-zA-Z-]+)::\s*(.*)$", line)
        if m:
            base, name, arg = len(m.group(1)), m.group(2).lower(), m.group(3).strip()
            body, i = block(i + 1, base)
            # directive options (":width: 400px") come first in the body
            while body and re.match(r"^\s*:[\w-]+:", body[0]):
                body.pop(0)
            text = textwrap.dedent("\n".join(body)).strip("\n")
            if name in DROP:
                continue
            if name == "math":
                out.extend(["", "$$", (arg + "\n" + text).strip(), "$$", ""])
            elif name in CODE:
                out.extend(["", "```", text, "```", ""])
            else:
                label = LABEL.get(name, None)
                title = convert_inline(arg) if arg and name != "figure" else ""
                # (a figure's body, once its options are gone, is its caption)
                prefix = f"**{label}:** " if label else (f"**{title}**" if title else "")
                inner = convert(text) if text else ""
                out.extend(["", prefix.strip()] if prefix else [""])
                out.extend([inner, ""])
            continue

        # labels (.. _name:), comments (..) and citation targets (.. [X] text)
        if re.match(r"^\s*\.\. _[^:]+:\s*$", line) or stripped == "..":
            i += 1
            continue
        cm = re.match(r"^(\s*)\.\. \[([^\]]+)\]\s*(.*)$", line)
        if cm:
            body, i = block(i + 1, len(cm.group(1)))
            ref = " ".join([cm.group(3)] + [b.strip() for b in body])
            out.append(f"[{cm.group(2)}] {convert_inline(ref)}")
            continue
        if re.match(r"^\s*\.\. ", line):     # other comments
            _, i = block(i + 1, _indent(line))
            continue

        # literal block: a paragraph ending in "::" introduces indented code
        if stripped.endswith("::") and not stripped.startswith(".."):
            text = line.rstrip()[:-1] if stripped != "::" else ""
            out.append(convert_inline(text.rstrip(": ") + (":" if text.strip() else "")))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _indent(lines[j]) > _indent(line):
                body, i = block(j, _indent(line))
                out.extend(["", "```", textwrap.dedent("\n".join(body)).strip("\n"), "```", ""])
            else:
                i += 1
            continue

        out.append(line.rstrip())
        i += 1

    # Roles and links often wrap across lines, so convert them on the whole text,
    # leaving code blocks and display math untouched.
    parts = re.split(r"(```.*?```|\$\$.*?\$\$)", "\n".join(out), flags=re.DOTALL)
    md = "".join(p if k % 2 else convert_inline(p) for k, p in enumerate(parts))
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(e["name"] for e in json.loads(fetch(API)) if e["name"].endswith(".rst"))
    written = 0
    for name in names:
        rst = fetch(f"{RAW}/doc/modules/{name}").decode("utf-8")
        if rst.lstrip().startswith(":orphan:"):   # redirect stubs, e.g. pipeline.rst
            continue
        source = f"https://github.com/{REPO}/blob/{TAG}/doc/modules/{name}"
        md = f"<!-- Converted from {source} (BSD-3-Clause) -->\n\n" + convert(rst)
        (OUT / name.replace(".rst", ".md")).write_text(md, encoding="utf-8")
        print(f"{name:40s} {len(md.split()):>7,} words", file=sys.stderr)
        written += 1
    (OUT / "LICENSE").write_bytes(fetch(f"{RAW}/COPYING"))
    print(f"Wrote {written} files to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
