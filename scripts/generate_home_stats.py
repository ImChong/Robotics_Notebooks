#!/usr/bin/env python3
"""generate_home_stats.py — 生成首页 Hero 统计所需的轻量 JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
GRAPH_STATS_PATH = REPO_ROOT / "exports" / "graph-stats.json"
OUT_PATH = REPO_ROOT / "exports" / "home-stats.json"


def build_payload(
    graph_stats: dict[str, Any],
    coverage: dict[str, int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": graph_stats.get("generated_at"),
        "node_count": graph_stats.get("node_count"),
        "edge_count": graph_stats.get("edge_count"),
        "coverage": {
            "covered": int(coverage["covered"]),
            "total": int(coverage["total"]),
            "percent": int(coverage["percent"]),
        },
    }
    latest_nodes = graph_stats.get("latest_wiki_nodes")
    latest = graph_stats.get("latest_wiki_node")
    if latest_nodes:
        payload["latest_wiki_nodes"] = latest_nodes
    if latest:
        payload["latest_wiki_node"] = latest
    return payload


def write_home_stats(coverage: dict[str, int] | None = None) -> dict[str, Any]:
    """写入 exports/home-stats.json。coverage 为空时内部跑一次 lint（兼容 make sync-stats）。"""
    if not GRAPH_STATS_PATH.exists():
        print(f"Missing {GRAPH_STATS_PATH}. Run make graph first.", file=sys.stderr)
        sys.exit(1)

    graph_stats = json.loads(GRAPH_STATS_PATH.read_text(encoding="utf-8"))
    if coverage is None:
        import lint_wiki

        coverage = lint_wiki.coverage_stats(lint_wiki.lint())

    payload = build_payload(graph_stats, coverage)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {OUT_PATH} with graph={payload['node_count']} nodes/"
        f"{payload['edge_count']} edges, coverage="
        f"{payload['coverage']['covered']}/{payload['coverage']['total']}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="生成首页 Hero 统计 JSON")
    parser.add_argument(
        "--coverage-json",
        help="复用 lint 覆盖率 JSON（键：covered/total/percent），避免重复跑 lint_wiki",
    )
    args = parser.parse_args()

    coverage: dict[str, int] | None = None
    if args.coverage_json:
        coverage = json.loads(Path(args.coverage_json).read_text(encoding="utf-8"))
    write_home_stats(coverage=coverage)


if __name__ == "__main__":
    main()
