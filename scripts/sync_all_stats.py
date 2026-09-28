#!/usr/bin/env python3
"""
sync_all_stats.py — 部署时统计数据同步工具（pages.yml 调用；结果不入库）

功能：
1. 调用 generate_link_graph.py (make graph) 更新图谱数据
2. 调用 generate_home_stats.py 更新首页轻量级统计 JSON
3. 自动同步数据文件到 docs/exports/
4. 写 docs/exports/graph-badge.json（README 知识图谱徽章的 shields.io endpoint 数据）
5. 更新 docs/index.html 中的 Hero 兜底统计数字、docs/sw.js 缓存版本

4、5 会改入库文件的工作区副本，只应在部署构建里运行；PR 不要提交这些改动
（否则并行 PR 在同几行冲突，见 scripts/pr_derived_guard.py）。
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from graph_exports_sync import copy_graph_exports_to_docs

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = REPO_ROOT / "docs" / "index.html"
HOME_STATS_JSON = REPO_ROOT / "exports" / "home-stats.json"
GRAPH_BADGE_JSON = REPO_ROOT / "docs" / "exports" / "graph-badge.json"


def graph_badge_payload(nodes: int, edges: int) -> dict:
    """shields.io endpoint 徽章数据（README 通过 img.shields.io/endpoint?url=... 引用）。"""
    return {
        "schemaVersion": 1,
        "label": "知识图谱",
        "message": f"{nodes}节点 {edges}边",
        "color": "blue",
        "namedLogo": "d3.js",
    }


def run_command(cmd: list[str], description: str):
    print(f"🚀 {description}...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        # 允许 lint 相关的脚本返回非零（如果是警告）
        if "lint" not in cmd[1]:
            print(f"❌ 命令失败: {' '.join(cmd)}")
            print(result.stderr)
            sys.exit(1)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="部署时同步图谱统计、首页 JSON、徽章与 docs 硬编码"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="跳过 generate_link_graph（ci-preflight 已单独跑过时使用）",
    )
    parser.add_argument(
        "--skip-home-stats",
        action="store_true",
        help="跳过 generate_home_stats（ci-preflight 已注入 lint coverage 时使用）",
    )
    args = parser.parse_args()

    # 1. 生成图谱数据和统计
    if not args.skip_graph:
        run_command(["python3", "scripts/generate_link_graph.py"], "生成图谱数据")

    # 2. 生成首页统计 JSON（无 --coverage-json 时会内部跑一次 lint）
    if not args.skip_home_stats:
        run_command(["python3", "scripts/generate_home_stats.py"], "生成首页统计 JSON")

    # 3. 确保 docs/exports 目录存在并同步图谱相关 JSON（与 make graph 共用逻辑）
    copy_graph_exports_to_docs()

    # 3b. 与 graph-stats.generated_at 对齐 Service Worker 缓存版本
    run_command(["python3", "scripts/sync_sw_cache_version.py"], "同步 SW 缓存版本")

    # 读取最新统计数据
    if not HOME_STATS_JSON.exists():
        print(f"❌ 找不到统计文件: {HOME_STATS_JSON}")
        sys.exit(1)

    stats = json.loads(HOME_STATS_JSON.read_text(encoding="utf-8"))
    nodes = stats["node_count"]
    edges = stats["edge_count"]
    cov_done = stats["coverage"]["covered"]
    cov_total = stats["coverage"]["total"]

    # 4. README 知识图谱徽章的 endpoint 数据（README 本身不再写入数字）
    GRAPH_BADGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_BADGE_JSON.write_text(
        json.dumps(graph_badge_payload(nodes, edges), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"✅ 已写入 {GRAPH_BADGE_JSON.relative_to(REPO_ROOT)}")

    # 5. 更新 docs/index.html（Hero 盘点硬编码数字；结构由前端维护，此处只刷 node/edge）
    if INDEX_HTML.exists():
        print("📝 更新 docs/index.html...")
        content = INDEX_HTML.read_text(encoding="utf-8")

        content = re.sub(
            r'(id="heroNodeCount"[^>]*>)\d+',
            rf"\g<1>{nodes}",
            content,
            count=1,
        )
        content = re.sub(
            r'(id="heroEdgeCount"[^>]*>)\d+',
            rf"\g<1>{edges}",
            content,
            count=1,
        )

        INDEX_HTML.write_text(content, encoding="utf-8")
        print("✅ docs/index.html 更新完成")

    print(
        f"\n✨ 所有统计数据同步完成！当前状态: {nodes} Nodes, {edges} Edges, Coverage {cov_done}/{cov_total}"
    )


if __name__ == "__main__":
    main()
