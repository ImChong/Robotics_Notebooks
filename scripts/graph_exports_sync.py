#!/usr/bin/env python3
"""Copy graph-related JSON from exports/ to docs/exports/ for GitHub Pages.

Single place for filenames and copy logic: used by ``make graph``,
``sync_all_stats.py``, and ``sync_wiki.sh`` to avoid drift.
"""

from __future__ import annotations

from pathlib import Path

GRAPH_EXPORT_FILES: tuple[str, ...] = ("link-graph.json", "graph-stats.json", "home-stats.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def copy_graph_exports_to_docs() -> None:
    root = repo_root()
    dst_dir = root / "docs" / "exports"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_EXPORT_FILES:
        src = root / "exports" / name
        dst = dst_dir / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"✅ 已同步: {name} -> docs/exports/")


def main() -> None:
    copy_graph_exports_to_docs()


if __name__ == "__main__":
    main()
