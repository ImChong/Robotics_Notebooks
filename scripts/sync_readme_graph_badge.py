#!/usr/bin/env python3
"""将 README.md 顶部 Knowledge Graph 徽章与 exports/graph-stats.json 对齐。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
GRAPH_STATS = REPO_ROOT / "exports" / "graph-stats.json"


def main() -> None:
    if not GRAPH_STATS.is_file():
        print(f"Missing {GRAPH_STATS}", file=sys.stderr)
        sys.exit(1)
    if not README.is_file():
        print(f"Missing {README}", file=sys.stderr)
        sys.exit(1)

    graph_stats = json.loads(GRAPH_STATS.read_text(encoding="utf-8"))
    node_count = int(graph_stats["node_count"])
    edge_count = int(graph_stats["edge_count"])
    graph_badge = (
        f"[![Knowledge Graph](https://img.shields.io/badge/知识图谱-{node_count}节点_{edge_count}边-blue?logo=d3.js)]"
        f"(https://imchong.github.io/Robotics_Notebooks/graph.html)"
    )

    content = README.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"\[!\[Knowledge Graph\]\([^)]+\)\]\([^)]+\)",
        graph_badge,
        content,
        count=1,
    )
    if count == 0:
        print("README 缺少 Knowledge Graph badge", file=sys.stderr)
        sys.exit(1)

    README.write_text(updated, encoding="utf-8")
    print(f"README graph badge synced: {node_count} nodes / {edge_count} edges")


if __name__ == "__main__":
    main()
