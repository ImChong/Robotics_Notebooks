#!/usr/bin/env python3
"""导出 wiki/formalizations 与精选 concepts 为 Anki 兼容 TSV。"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "exports" / "anki-flashcards.tsv"
FORMALIZATIONS_DIR = REPO_ROOT / "wiki" / "formalizations"
CONCEPTS_DIR = REPO_ROOT / "wiki" / "concepts"


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("\n---\n", 1)
    return parts[1] if len(parts) == 2 else text


def extract_title(body: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def extract_first_paragraph_after_h1(body: str) -> str:
    lines = body.splitlines()
    h1_seen = False
    paragraph: list[str] = []
    for line in lines:
        if not h1_seen:
            if line.startswith("# "):
                h1_seen = True
            continue
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("## "):
            break
        if stripped.startswith("---"):
            continue
        paragraph.append(stripped)
    return clean_inline_markdown(" ".join(paragraph))


def extract_first_math_block(body: str) -> str:
    match = re.search(r"\$\$(.*?)\$\$", body, re.DOTALL)
    if not match:
        return ""
    content = match.group(1).strip()
    return f"$$\n{content}\n$$"


def extract_section(body: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    match = re.search(pattern, body, re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    rest = body[start:]
    next_heading = re.search(r"^##\s+.+$", rest, re.MULTILINE)
    section = rest[: next_heading.start()] if next_heading else rest
    return section.strip()


def extract_related(section_text: str) -> list[str]:
    related: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        label_match = re.search(r"\[([^\]]+)\]", stripped)
        if label_match:
            related.append(clean_inline_markdown(label_match.group(1)))
        else:
            related.append(clean_inline_markdown(stripped[2:]))
    return related


def extract_one_line_definition(body: str) -> str:
    section = extract_section(body, "一句话定义")
    if not section:
        return ""
    lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith(("## ", "### ")):
            break
        lines.append(stripped.lstrip("> "))
    return clean_inline_markdown(" ".join(lines))


def clean_inline_markdown(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def make_back(*parts: str) -> str:
    return "<br><br>".join(part for part in parts if part)


def make_related_html(related: Iterable[str]) -> str:
    items = [name for name in related if name]
    if not items:
        return ""
    lis = "".join(f"<li>{escape_html(name)}</li>" for name in items[:8])
    return f"<strong>关联概念</strong><ul>{lis}</ul>"


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def formalization_cards() -> list[tuple[str, str, str]]:
    cards: list[tuple[str, str, str]] = []
    for path in sorted(FORMALIZATIONS_DIR.glob("*.md")):
        body = strip_frontmatter(path.read_text(encoding="utf-8"))
        title = extract_title(body, path.stem)
        summary = extract_first_paragraph_after_h1(body)
        formula = extract_first_math_block(body)
        related = extract_related(extract_section(body, "关联页面"))
        front = make_back(f"<strong>{escape_html(title)}</strong>", escape_html(summary))
        formula_html = (
            f"<strong>核心公式</strong><pre>{escape_html(formula)}</pre>" if formula else ""
        )
        back = make_back(formula_html, make_related_html(related))
        tags = f"robotics::formalization::{path.stem}"
        cards.append((front, back, tags))
    return cards


def concept_cards() -> list[tuple[str, str, str]]:
    cards: list[tuple[str, str, str]] = []
    for path in sorted(CONCEPTS_DIR.glob("*.md")):
        body = strip_frontmatter(path.read_text(encoding="utf-8"))
        definition = extract_one_line_definition(body)
        if not definition:
            continue
        title = extract_title(body, path.stem)
        related = extract_related(extract_section(body, "关联页面"))
        front = f"<strong>{escape_html(title)}</strong><br><br>一句话定义是什么？"
        back = make_back(escape_html(definition), make_related_html(related))
        tags = f"robotics::concept::{path.stem}"
        cards.append((front, back, tags))
    return cards


DECK_SOURCES: dict[str, list[tuple[str, str, str]]] = {}


def get_deck(deck: str) -> list[tuple[str, str, str]]:
    """Return cards for a specific deck name."""
    if deck == "formalization":
        return formalization_cards()
    if deck == "concepts-core":
        return concept_cards()
    if deck == "control-stability":
        # control-stability deck: formalizations + control-related concepts
        stability_stems = {"lyapunov", "hjb", "lqr", "gae", "bellman-equation",
                           "contact-complementarity", "control-lyapunov-function"}
        f_cards = [c for c in formalization_cards()
                   if any(stem in c[2] for stem in stability_stems)]
        c_cards = [c for c in concept_cards()
                   if any(tag in c[2] for tag in
                          ("stability", "cbf", "clf", "lyapunov", "control-barrier",
                           "whole-body-control", "optimal-control"))]
        return f_cards + c_cards
    # default: all
    return formalization_cards() + concept_cards()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="导出 Anki 兼容 TSV 闪卡")
    parser.add_argument(
        "--deck",
        choices=["formalization", "concepts-core", "control-stability", "all"],
        default="all",
        help="选择导出的牌组（默认：all）",
    )
    args = parser.parse_args()

    if args.deck == "all":
        cards = formalization_cards() + concept_cards()
        out_path = OUT_PATH
    else:
        cards = get_deck(args.deck)
        out_path = REPO_ROOT / "exports" / f"anki-{args.deck}.tsv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["Front", "Back", "Tags"])
        writer.writerows(cards)

    f_count = len(formalization_cards())
    c_count = len(concept_cards())
    print(
        f"✅ {out_path.name}: {len(cards)} cards"
        + (f" ({f_count} formalizations + {c_count} concepts)" if args.deck == "all" else f" (deck={args.deck})")
        + f" → {out_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
