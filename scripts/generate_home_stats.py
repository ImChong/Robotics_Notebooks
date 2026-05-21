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
    check=False,  # We want to continue even if lint fails
)
all_output = lint_result.stdout + lint_result.stderr
coverage_match = re.search(r"Sources 覆盖率：(\d+)/(\d+) \((\d+)%\)", all_output)
if not coverage_match:
    print("Failed to parse Sources coverage from lint_wiki.py output", file=sys.stderr)
    print("Full output:", file=sys.stderr)
    print(all_output[-1000:], file=sys.stderr)
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
latest_nodes = graph_stats.get("latest_wiki_nodes")
latest = graph_stats.get("latest_wiki_node")
if latest_nodes:
    payload["latest_wiki_nodes"] = latest_nodes
if latest:
    payload["latest_wiki_node"] = latest
OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(
    f"Wrote {OUT_PATH} with graph={payload['node_count']} nodes/{payload['edge_count']} edges, coverage={payload['coverage']['covered']}/{payload['coverage']['total']}"
)
