#!/usr/bin/env python3
"""Merge PAI awesome index stubs into pre-existing canonical wiki entities.

The first deepen pass reused arXiv / GitHub / exact-H1 matches, but missed
same-substance pages whose canonical copy has no frontmatter arXiv or uses a
project-page URL. This script deletes those stubs, rewrites inbound links,
registers page-aliases, and patches hub/catalog stats.

Idempotent: missing stub files are skipped.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITIES = ROOT / "wiki" / "entities"
ALIASES = ROOT / "schema" / "page-aliases.json"
TODAY = date.today().isoformat()
SKIP_PARTS = {".git", "node_modules", ".cursor-artifacts", "raw"}

# stub stem (wiki/entities) → canonical stem (wiki/entities)
MERGE_MAP: dict[str, str] = {
    "paper-pai-2403-04436-h2ohumantohumanoidrealtimewhol": "paper-hrl-stack-07-learning_human_to_humanoid_real_time",
    "painode-191-expressivewholebodycontrol": "paper-exbody-expressive-humanoid",
    "painode-171-stanfordpupper": "stanford-doggo-and-pupper",
    "painode-264-nav2": "navigation2",
    "painode-335-learningrobustperceptivelocomotio": "paper-robust-perceptive-locomotion-wild",
    "paper-pai-2207-07802-rapidlocomotionviarl": "paper-rapid-locomotion-rl",
    "painode-166-openmanipulator": "robotis-open-manipulator-line",
    "painode-157-dexumi": "paper-notebook-dexumi-using-human-hand-as-the-universal-manipul",
}

# Extra frontmatter to inject on canonical pages after merge.
CANONICAL_PATCH: dict[str, dict[str, str]] = {
    "paper-hrl-stack-07-learning_human_to_humanoid_real_time": {
        "arxiv": "2403.04436",
    },
}

TECH_MAP_RELATED = "../overview/awesome-physical-ai-technology-map.md"
CATALOG_SRC = "../../sources/repos/awesome-physical-ai-union-catalog.md"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise SystemExit(f"frontmatter 缺失: {text[:80]!r}")
    return match.group(1), text[match.end() :]


def add_list_entries(frontmatter: str, key: str, entries: list[str]) -> str:
    block = re.search(rf"^{key}:\n((?:  - .+\n)+)", frontmatter + "\n", re.MULTILINE)
    if not block:
        return frontmatter
    existing = block.group(1)
    extra = "".join(f"  - {e}\n" for e in entries if f"  - {e}\n" not in existing)
    if not extra:
        return frontmatter
    padded = frontmatter + "\n"
    return (padded[: block.end()] + extra + padded[block.end() :]).rstrip("\n")


def ensure_arxiv(frontmatter: str, arxiv: str) -> str:
    if re.search(r"(?m)^arxiv:\s*", frontmatter):
        return frontmatter
    if re.search(r"(?m)^updated:", frontmatter):
        return re.sub(
            r"(?m)^(updated:.*)$",
            rf'\1\narxiv: "{arxiv}"',
            frontmatter,
            count=1,
        )
    return frontmatter + f'\narxiv: "{arxiv}"'


def rewrite_links(text: str, stub: str, canonical: str) -> str:
    return text.replace(f"{stub}.md", f"{canonical}.md").replace(
        f"]({stub})", f"]({canonical})"
    )


def walk_text_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".json", ".yml", ".yaml", ".py"}:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        out.append(path)
    return out


def merge_into_canonical(stub: str, canonical: str) -> None:
    path = ENTITIES / f"{canonical}.md"
    frontmatter, body = split_frontmatter(load_text(path))
    frontmatter = re.sub(r"^updated:.*$", f"updated: {TODAY}", frontmatter, count=1, flags=re.M)
    extra = CANONICAL_PATCH.get(canonical, {})
    if extra.get("arxiv"):
        frontmatter = ensure_arxiv(frontmatter, extra["arxiv"])
    rel_tech = TECH_MAP_RELATED if "/entities/" in str(path) else TECH_MAP_RELATED
    # related paths are relative to the canonical file (always wiki/entities/* here)
    frontmatter = add_list_entries(
        frontmatter,
        "related",
        [rel_tech, "./awesome-physical-ai-natnew.md"],
    )
    frontmatter = add_list_entries(
        frontmatter,
        "sources",
        [CATALOG_SRC, "../../sources/repos/awesome-physical-ai-natnew.md"],
    )
    path.write_text("---\n" + frontmatter.strip() + "\n---\n" + body, encoding="utf-8")


def update_aliases() -> None:
    raw = load_text(ALIASES)
    data = json.loads(raw)
    additions = [
        f'    "entity-{stub}": "entity-{canonical}",\n'
        for stub, canonical in sorted(MERGE_MAP.items())
        if f"entity-{stub}" not in data["aliases"]
    ]
    if not additions:
        return
    anchor = '  "aliases": {\n'
    raw = raw.replace(anchor, anchor + "".join(additions), 1)
    json.loads(raw)
    ALIASES.write_text(raw, encoding="utf-8")


def patch_coverage_numbers() -> None:
    new_count = 249 - len(MERGE_MAP)
    reused = 135 + len(MERGE_MAP)
    pattern = re.compile(
        r"去重后 \*\*384\*\* 条独立详情节点（新建 \d+，复用 \d+；两清单同时出现 33）"
    )
    replacement = (
        f"去重后 **384** 条独立详情节点（新建 {new_count}，复用 {reused}；两清单同时出现 33）"
    )
    for rel in (
        "wiki/entities/awesome-physical-ai-natnew.md",
        "wiki/entities/awesome-physical-ai-aichr.md",
        "wiki/comparisons/awesome-physical-ai-curated-lists.md",
        "wiki/overview/awesome-physical-ai-technology-map.md",
    ):
        path = ROOT / rel
        text = load_text(path)
        text2 = pattern.sub(replacement, text)
        text2 = text2.replace("新建 249、复用 135", f"新建 {new_count}、复用 {reused}")
        text2 = text2.replace("新建 **249**，复用 **135**", f"新建 **{new_count}**，复用 **{reused}**")
        if text2 != text:
            path.write_text(text2, encoding="utf-8")


def main() -> None:
    merged: list[str] = []
    for stub, canonical in MERGE_MAP.items():
        stub_path = ENTITIES / f"{stub}.md"
        canon_path = ENTITIES / f"{canonical}.md"
        if not stub_path.exists():
            continue
        if not canon_path.exists():
            raise SystemExit(f"canonical 页不存在: {canonical}")
        merge_into_canonical(stub, canonical)
        stub_path.unlink()
        merged.append(stub)

    changed = 0
    for path in walk_text_files():
        try:
            text = original = load_text(path)
        except UnicodeDecodeError:
            continue
        for stub, canonical in MERGE_MAP.items():
            if stub not in text:
                continue
            text = rewrite_links(text, stub, canonical)
        if path.name == "awesome-physical-ai-union-catalog.md":
            for stub, canonical in MERGE_MAP.items():
                text = text.replace(
                    f"[`{stub}.md`](../../wiki/entities/{canonical}.md) | 新建",
                    f"[`{canonical}.md`](../../wiki/entities/{canonical}.md) | 复用",
                )
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1

    update_aliases()
    patch_coverage_numbers()
    print(f"merged={len(merged)} files_rewritten={changed}")
    for s in merged:
        print(f"  {s} -> {MERGE_MAP[s]}")


if __name__ == "__main__":
    main()
