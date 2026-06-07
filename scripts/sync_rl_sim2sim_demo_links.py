#!/usr/bin/env python3
"""Sync RL Sim2Sim Demo Website links into wiki pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
INDEX_PATH = SCHEMA_DIR / "rl-sim2sim-demo-index.json"
MAP_PATH = SCHEMA_DIR / "rl-sim2sim-demo-wiki-map.yml"
LINK_PREFIX = "RL Sim2Sim 在线演示："
SITE_MARKER = "RL_Sim2Sim_Demo_Website"


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        print(f"missing index: {INDEX_PATH}", file=sys.stderr)
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def load_mapping() -> dict[str, list[str]]:
    if not MAP_PATH.exists():
        return {}
    data = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides", {})
    result: dict[str, list[str]] = {}
    for key, value in overrides.items():
        result[key] = [value] if isinstance(value, str) else list(value)
    return result


def link_line(entry: dict) -> str:
    label = entry.get("label") or entry["title"]
    if entry["id"] == "homepage":
        label = entry.get("label") or "MuJoCo WASM + ONNX"
        text = f"{LINK_PREFIX}{label}"
    else:
        text = f"{LINK_PREFIX}{label}"
    return f"- [{text}]({entry['url']})"


def has_demo_link(text: str, entry: dict) -> bool:
    if SITE_MARKER not in text:
        return False
    label = entry.get("label") or entry["title"]
    return label in text or entry["id"] in text


def inject_link(wiki_rel: str, line: str, entry: dict, dry_run: bool) -> bool:
    wiki_path = ROOT / wiki_rel
    if not wiki_path.exists():
        print(f"missing wiki page: {wiki_rel}", file=sys.stderr)
        return False
    text = wiki_path.read_text(encoding="utf-8")
    if line in text or has_demo_link(text, entry):
        return False

    section = "## 推荐继续阅读"
    if section in text:
        parts = text.split(section, 1)
        body = parts[1].lstrip("\n")
        new_text = parts[0] + section + "\n\n" + line + "\n" + body
    else:
        ref_section = "## 参考来源"
        if ref_section in text:
            parts = text.split(ref_section, 1)
            new_text = parts[0] + ref_section + "\n\n" + line + "\n" + parts[1].lstrip("\n")
        else:
            new_text = text.rstrip() + f"\n\n{section}\n\n{line}\n"

    if not dry_run:
        wiki_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned edits without writing files"
    )
    args = parser.parse_args()

    entries = load_index()
    mapping = load_mapping()
    entry_by_id = {entry["id"]: entry for entry in entries}

    changed = 0
    for demo_id, wiki_paths in mapping.items():
        entry = entry_by_id.get(demo_id)
        if not entry:
            print(f"warning: unknown demo id in map: {demo_id}", file=sys.stderr)
            continue
        line = link_line(entry)
        for wiki_rel in wiki_paths:
            if inject_link(wiki_rel, line, entry, args.dry_run):
                changed += 1
                action = "would update" if args.dry_run else "updated"
                print(f"{action}: {wiki_rel} <- {entry['title']}")

    print(
        f"{'would change' if args.dry_run else 'changed'} {changed} wiki pages; "
        f"mapped demos {len(mapping)} / {len(entries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
