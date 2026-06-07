#!/usr/bin/env python3
"""Sync one-line summaries from Humanoid Paper Notebooks into wiki stub entities."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
FULL_MAP_PATH = SCHEMA_DIR / "paper-notebook-wiki-full-map.yml"
INDEX_PATH = SCHEMA_DIR / "paper-notebook-index.json"
NOTEBOOK_RAW = (
    "https://raw.githubusercontent.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/main"
)

GENERIC_ONE_LINE = "以 Paper Notebooks 深读笔记为首要编译来源"
GENERIC_SUMMARY = "Humanoid Paper Notebooks 深读笔记索引实体；待从笔记与论文 PDF 深化归纳"
GENERIC_SOURCE = "的深读笔记索引；正文以笔记站与 arXiv 为准"


def load_mapping() -> dict[str, list[str]]:
    data = yaml.safe_load(FULL_MAP_PATH.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides", data)
    result: dict[str, list[str]] = {}
    for key, value in overrides.items():
        result[key] = [value] if isinstance(value, str) else list(value)
    return result


def load_index() -> dict[str, dict]:
    papers = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {p["dir"]: p for p in papers}


def fetch_notebook_markdown(paper: dict) -> str:
    url = f"{NOTEBOOK_RAW}/{paper['folder']}/{paper['dir']}.md"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def extract_notebook_summary(markdown: str) -> str | None:
    match = re.search(
        r"##\s*🎯\s*一句话总结\s*\n\n(.+?)(?:\n\n---|\n\n## )",
        markdown,
        re.DOTALL,
    )
    if not match:
        return None
    text = match.group(1).strip()
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = " ".join(text.split())
    if not text or len(text) < 20 or "待读" in text:
        return None
    return text


def yaml_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def split_frontmatter(content: str) -> tuple[str, str]:
    match = re.match(r"^(---\n.*?\n---\n)([\s\S]*)$", content, re.DOTALL)
    if not match:
        return "", content
    return match.group(1), match.group(2)


def replace_summary_frontmatter(fm_block: str, summary: str) -> str:
    quoted = yaml_quote(summary)
    if re.search(r"^summary\s*:", fm_block, re.MULTILINE):
        return re.sub(r'^summary:\s*".*?"\s*$', f"summary: {quoted}", fm_block, flags=re.MULTILINE)
    lines = fm_block.splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("updated:"):
            out.append(f"summary: {quoted}")
            inserted = True
    if not inserted:
        out.insert(-1, f"summary: {quoted}")
    return "\n".join(out) + ("\n" if not fm_block.endswith("\n") else "")


def replace_one_line_section(body: str, summary: str) -> str:
    section = "## 一句话定义"
    if section not in body:
        return body
    pattern = re.compile(
        rf"({re.escape(section)}\s*\n\n)(.+?)(\n\n## )",
        re.DOTALL,
    )
    match = pattern.search(body)
    if not match:
        return body
    return body[: match.start()] + match.group(1) + summary + match.group(3) + body[match.end() :]


def update_source_file(source_path: Path, summary: str, dry_run: bool) -> bool:
    if not source_path.exists():
        return False
    text = source_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(- \*\*一句话说明：\*\* ).+?\n")
    if not pattern.search(text):
        return False
    new_text = pattern.sub(lambda m: m.group(1) + summary + "\n", text, count=1)
    if new_text == text:
        return False
    if not dry_run:
        source_path.write_text(new_text, encoding="utf-8")
    return True


def needs_summary(wiki_path: Path) -> bool:
    if not wiki_path.exists():
        return False
    text = wiki_path.read_text(encoding="utf-8")
    return GENERIC_ONE_LINE in text or GENERIC_SUMMARY in text


def update_wiki_page(wiki_path: Path, summary: str, dry_run: bool) -> bool:
    text = wiki_path.read_text(encoding="utf-8")
    fm_block, body = split_frontmatter(text)
    if not fm_block:
        return False
    new_fm = replace_summary_frontmatter(fm_block, summary)
    new_body = replace_one_line_section(body, summary)
    new_text = new_fm + new_body
    if new_text == text:
        return False
    if not dry_run:
        wiki_path.write_text(new_text, encoding="utf-8")
    return True


def source_needs_summary(source_path: Path) -> bool:
    if not source_path.exists():
        return False
    return GENERIC_SOURCE in source_path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned edits only")
    args = parser.parse_args()

    mapping = load_mapping()
    index = load_index()
    wiki_changed = 0
    source_changed = 0
    skipped = 0
    failures: list[str] = []

    for paper_dir, wiki_paths in sorted(mapping.items()):
        primary = wiki_paths[0]
        if not primary.startswith("wiki/entities/paper-notebook-"):
            continue
        wiki_path = ROOT / primary
        if not wiki_path.exists():
            failures.append(f"missing wiki page: {primary}")
            continue

        source_path: Path | None = None
        fm_match = re.match(
            r"^---\n(.*?)\n---\n",
            wiki_path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        if fm_match:
            for line in fm_match.group(1).splitlines():
                src_match = re.match(
                    r"^\s*-\s+(\.\./\.\./sources/papers/[^\n]+\.md)\s*$",
                    line,
                )
                if src_match:
                    source_path = (wiki_path.parent / src_match.group(1)).resolve()
                    break

        wiki_needs = needs_summary(wiki_path)
        source_needs = source_needs_summary(source_path) if source_path else False
        if not wiki_needs and not source_needs:
            skipped += 1
            continue

        paper = index.get(paper_dir)
        if not paper:
            failures.append(f"missing index entry: {paper_dir}")
            continue
        try:
            markdown = fetch_notebook_markdown(paper)
        except OSError as exc:
            failures.append(f"fetch failed {paper_dir}: {exc}")
            continue
        summary = extract_notebook_summary(markdown)
        if not summary:
            failures.append(f"no summary in notebook md: {paper_dir}")
            continue

        if wiki_needs and update_wiki_page(wiki_path, summary, args.dry_run):
            wiki_changed += 1
            action = "would update" if args.dry_run else "updated"
            print(f"{action} wiki: {primary}")
        if source_path and source_needs and update_source_file(source_path, summary, args.dry_run):
            source_changed += 1
            action = "would update" if args.dry_run else "updated"
            print(f"{action} source: {source_path.relative_to(ROOT)}")

    print(
        f"{'would change' if args.dry_run else 'changed'} "
        f"{wiki_changed} wiki pages, {source_changed} sources; "
        f"skipped {skipped} (already have summaries)"
    )
    if failures:
        print(f"failures ({len(failures)}):", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
