#!/usr/bin/env python3
"""generate_home_stats.py — 生成首页 Hero 统计所需的轻量 JSON。"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GRAPH_STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
OUT_PATH = REPO_ROOT / "exports" / "home-stats.json"

if not GRAPH_STATS_PATH.exists():
    print(f"Missing {GRAPH_STATS_PATH}. Run make graph first.", file=sys.stderr)
    sys.exit(1)

graph_stats = json.loads(GRAPH_STATS_PATH.read_text(encoding="utf-8"))
lint_result = subprocess.run(
    ["python3", "scripts/lint_wiki.py"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
)
coverage_match = re.search(r"Sources 覆盖率：(\d+)/(\d+) \((\d+)%\)", lint_result.stdout)
if not coverage_match:
    print("Failed to parse Sources coverage from lint_wiki.py output", file=sys.stderr)
    print(lint_result.stdout[-500:], file=sys.stderr)
    sys.exit(1)

payload = {
    "generated_at": graph_stats.get("generated_at"),
    "node_count": graph_stats.get("node_count"),
    "edge_count": graph_stats.get("edge_count"),
    "coverage": {
        "covered": int(coverage_match.group(1)),
        "total": int(coverage_match.group(2)),
        "percent": int(coverage_match.group(3)),
    },
}
OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT_PATH} with graph={payload['node_count']} nodes/{payload['edge_count']} edges, coverage={payload['coverage']['covered']}/{payload['coverage']['total']}")
