#!/usr/bin/env python3
"""update_badge.py — 根据当前仓库真实统计更新 README.md 顶部 badges。"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"
GRAPH_STATS = REPO_ROOT / "exports" / "graph-stats.json"
CHECKLIST_DIR = REPO_ROOT / "docs" / "checklists"


def latest_checklist_path() -> str:
    candidates = sorted(CHECKLIST_DIR.glob("tech-stack-next-phase-checklist-v*.md"))
    if not candidates:
        print("No checklist files found under docs/checklists", file=sys.stderr)
        sys.exit(1)
    latest = max(candidates, key=lambda p: int(re.search(r"v(\d+)", p.stem).group(1)))
    return str(latest.relative_to(REPO_ROOT))


result = subprocess.run(
    ["python3", "scripts/lint_wiki.py"],
    capture_output=True, text=True, cwd=REPO_ROOT
)
m = re.search(r"(\d+)/(\d+) \((\d+)%\)", result.stdout)
if not m:
    print("Coverage not found in lint output:", result.stdout[-200:], file=sys.stderr)
    sys.exit(1)

pct = int(m.group(3))
color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
checklist_path = latest_checklist_path()
source_badge = (
    f"[![Sources Coverage](https://img.shields.io/badge/sources覆盖率-{pct}%25-{color})]"
    f"({checklist_path})"
)

if not GRAPH_STATS.exists():
    print(f"Missing {GRAPH_STATS}", file=sys.stderr)
    sys.exit(1)

graph_stats = json.loads(GRAPH_STATS.read_text(encoding="utf-8"))
node_count = graph_stats["node_count"]
edge_count = graph_stats["edge_count"]
graph_badge = (
    f"[![Knowledge Graph](https://img.shields.io/badge/知识图谱-{node_count}节点_{edge_count}边-blue?logo=d3.js)]"
    f"(https://imchong.github.io/Robotics_Notebooks/graph.html)"
)

content = README.read_text(encoding="utf-8")
content = re.sub(
    r"\[!\[Sources Coverage\]\([^)]+\)\]\([^)]+\)",
    source_badge,
    content,
)
content = re.sub(
    r"\[!\[Knowledge Graph\]\([^)]+\)\]\([^)]+\)",
    graph_badge,
    content,
)
README.write_text(content, encoding="utf-8")
print(f"Badges updated: graph={node_count}节点/{edge_count}边, coverage={pct}% ({color})")
