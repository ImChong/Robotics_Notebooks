#!/usr/bin/env python3
"""Merge duplicate paper-notebook-* stubs into canonical deep wiki entities."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml
from sync_paper_notebook_links import (
    ROOT,
    SCHEMA_DIR,
    inject_link,
    parse_frontmatter,
    short_label,
)

FULL_MAP_PATH = SCHEMA_DIR / "paper-notebook-wiki-full-map.yml"
WIKI = ROOT / "wiki"
SOURCES = ROOT / "sources" / "papers"
LINK_PREFIX = "机器人论文阅读笔记："


def fm_arxiv(text: str) -> str | None:
    fm = parse_frontmatter(text)
    match = re.search(r'^arxiv:\s*"?([0-9]+\.[0-9]+)"?\s*$', fm, re.M)
    return match.group(1) if match else None


def entity_score(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    score = 0
    if "paper-notebook-" in path.name:
        score -= 50
    if "status: complete" in fm or "status: complete\n" in fm:
        score += 30
    if "paper-notebook-stub" in fm or "paper-notebook-planned" in fm:
        score -= 20
    if "paper-notebook-planned" in fm:
        score -= 10
    score += min(len(text) // 500, 20)
    if path.name.startswith("paper-behavior-foundation-model"):
        score += 5
    if path.name.startswith("paper-pilot-"):
        score += 5
    if "paper-bfm-" in path.name or "paper-hrl-stack-" in path.name:
        score -= 15
    return score


def find_merge_pairs() -> list[tuple[str, str]]:
    by_arxiv: dict[str, list[Path]] = {}
    for path in (WIKI / "entities").glob("paper*.md"):
        arxiv = fm_arxiv(path.read_text(encoding="utf-8"))
        if not arxiv:
            continue
        by_arxiv.setdefault(arxiv, []).append(path)

    pairs: list[tuple[str, str]] = []
    for paths in by_arxiv.values():
        if len(paths) < 2:
            continue
        notebooks = [p for p in paths if "paper-notebook-" in p.name]
        if not notebooks:
            continue
        keepers = sorted(
            [p for p in paths if "paper-notebook-" not in p.name],
            key=entity_score,
            reverse=True,
        )
        if not keepers:
            continue
        keeper = keepers[0]
        for stub in notebooks:
            pairs.append((str(stub.relative_to(ROOT)), str(keeper.relative_to(ROOT))))
    return pairs


def replace_links(remove_rel: str, keep_rel: str, dry_run: bool) -> int:
    remove_name = Path(remove_rel).name
    keep_name = Path(keep_rel).name
    changed = 0
    for path in WIKI.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        new = text
        new = new.replace(f"]({remove_rel})", f"]({keep_rel})")
        new = new.replace(f"](../entities/{remove_name})", f"](../entities/{keep_name})")
        new = new.replace(f"](./{remove_name})", f"](./{keep_name})")
        new = new.replace(
            f"](../../wiki/entities/{remove_name})", f"](../../wiki/entities/{keep_name})"
        )
        if new != text:
            changed += 1
            if not dry_run:
                path.write_text(new, encoding="utf-8")
    for path in SOURCES.glob("humanoid_pnb_*.md"):
        text = path.read_text(encoding="utf-8")
        new = text.replace(remove_rel, keep_rel).replace(remove_name, keep_name)
        if new != text:
            changed += 1
            if not dry_run:
                path.write_text(new, encoding="utf-8")
    return changed


def patch_notebook_link(keeper_rel: str, stub_rel: str, dry_run: bool) -> bool:
    keeper = ROOT / keeper_rel
    stub = ROOT / stub_rel
    if not keeper.exists() or not stub.exists():
        return False
    stub_text = stub.read_text(encoding="utf-8")
    url_match = re.search(
        r"<(https://imchong\.github\.io/Humanoid_Robot_Learning_Paper_Notebooks[^>]+)>", stub_text
    )
    if not url_match:
        return False
    title_match = re.search(r"\*\*([^*]+)\*\*", stub_text)
    title = title_match.group(1) if title_match else keeper.stem
    paper = {"title": title, "url": url_match.group(1)}
    line = f"- [{LINK_PREFIX}{short_label(paper['title'])}]({paper['url']})"
    return inject_link(keeper_rel, line, dry_run)


def update_full_map(remove_rel: str, keep_rel: str, dry_run: bool) -> int:
    if not FULL_MAP_PATH.exists():
        return 0
    data = yaml.safe_load(FULL_MAP_PATH.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides", {})
    updated = 0
    for key, targets in list(overrides.items()):
        new_targets = [keep_rel if t == remove_rel else t for t in targets]
        if new_targets != targets:
            overrides[key] = new_targets
            updated += 1
    if updated and not dry_run:
        FULL_MAP_PATH.write_text(
            yaml.safe_dump({"overrides": overrides}, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs = find_merge_pairs()
    if not pairs:
        print("no duplicate paper-notebook stubs found")
        return 0

    deleted_src = 0
    for remove_rel, keep_rel in pairs:
        print(f"merge {remove_rel} -> {keep_rel}")
        replace_links(remove_rel, keep_rel, args.dry_run)
        patch_notebook_link(keep_rel, remove_rel, args.dry_run)
        update_full_map(remove_rel, keep_rel, args.dry_run)
        stub_path = ROOT / remove_rel
        if stub_path.exists():
            stub_text = stub_path.read_text(encoding="utf-8")
            fm = parse_frontmatter(stub_text)
            for line in fm.splitlines():
                m = re.match(r"^\s*-\s+\.\./\.\./sources/papers/([^\s]+\.md)\s*$", line)
                if not m:
                    continue
                src_path = ROOT / "sources" / "papers" / m.group(1)
                if src_path.exists():
                    if not args.dry_run:
                        src_path.unlink()
                    deleted_src += 1
            if not args.dry_run:
                stub_path.unlink()

    print(
        f"{'would merge' if args.dry_run else 'merged'} {len(pairs)} stub pairs; "
        f"{'would delete' if args.dry_run else 'deleted'} {deleted_src} sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
