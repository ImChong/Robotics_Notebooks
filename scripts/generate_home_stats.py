#!/usr/bin/env python3
"""generate_home_stats.py — 生成首页 Hero 统计所需的轻量 JSON。"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GRAPH_STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
INDEX_V1_PATH = REPO_ROOT / "exports" / "index-v1.json"
OUT_PATH = REPO_ROOT / "exports" / "home-stats.json"

RECENT_WIKI_NODES_LIMIT = 12


def pick_recent_wiki_nodes(index_data: dict) -> list[dict[str, object]]:
    """按 frontmatter `updated` 新到旧，取 wiki / entity 详情页条目（供首页卡片）。"""
    items = index_data.get("items") or []
    candidates: list[tuple[str, str, dict]] = []
    for it in items:
        if it.get("type") not in ("wiki_page", "entity_page"):
            continue
        iid = it.get("id")
        if not iid:
            continue
        u = str(it.get("updated") or "1970-01-01")
        candidates.append((u, str(iid), it))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out: list[dict[str, object]] = []
    for _, _, it in candidates[:RECENT_WIKI_NODES_LIMIT]:
        summary = str(it.get("summary") or "")
        out.append(
            {
                "id": it["id"],
                "title": it.get("title") or it["id"],
                "summary": summary[:240],
                "path": str(it.get("path") or ""),
                "tags": list(it.get("tags") or [])[:6],
                "page_type": str(it.get("page_type") or ""),
                "updated": str(it.get("updated") or ""),
            }
        )
    return out


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
    "recent_wiki_nodes": [],
}

if INDEX_V1_PATH.exists():
    try:
        index_data = json.loads(INDEX_V1_PATH.read_text(encoding="utf-8"))
        payload["recent_wiki_nodes"] = pick_recent_wiki_nodes(index_data)
    except json.JSONDecodeError:
        pass

OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(
    f"Wrote {OUT_PATH} with graph={payload['node_count']} nodes/{payload['edge_count']} edges, "
    f"coverage={payload['coverage']['covered']}/{payload['coverage']['total']}, "
    f"recent_wiki_nodes={len(payload['recent_wiki_nodes'])}"
)
