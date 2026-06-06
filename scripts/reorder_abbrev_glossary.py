#!/usr/bin/env python3
"""Move ## 英文缩写速查 to canonical position (after 一句话定义, before 为什么重要).

用法:
    python3 scripts/reorder_abbrev_glossary.py            # 写入
    python3 scripts/reorder_abbrev_glossary.py --dry-run  # 只统计
"""

from __future__ import annotations

import sys
from pathlib import Path

from wiki_abbrev_section import is_abbrev_glossary_well_placed, reorder_abbrev_glossary

ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = ROOT / "wiki"


def main() -> int:
    dry = "--dry-run" in sys.argv
    changed = 0
    skipped = 0
    wrong: list[str] = []

    for page in sorted(WIKI_DIR.rglob("*.md")):
        rel_str = str(page.relative_to(ROOT))
        if page.name.lower() in ("readme.md", "index.md") or any(
            seg in rel_str for seg in ("references/", "roadmaps/")
        ):
            continue

        content = page.read_text(encoding="utf-8")
        if is_abbrev_glossary_well_placed(content):
            skipped += 1
            continue

        new_content, did_change = reorder_abbrev_glossary(content)
        rel = page.relative_to(ROOT)
        wrong.append(str(rel))
        if did_change:
            if not dry:
                page.write_text(new_content, encoding="utf-8")
            changed += 1
            print(f"{'[dry-run] ' if dry else ''}REORDER {rel}")
        else:
            print(f"SKIP (no anchor) {rel}")

    print(
        f"\n{'[dry-run] ' if dry else ''}reordered={changed} already_ok={skipped} misplaced={len(wrong)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
