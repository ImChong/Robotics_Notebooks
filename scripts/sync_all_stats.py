#!/usr/bin/env python3
"""
sync_all_stats.py — 自动化统计数据同步工具

功能：
1. 调用 generate_link_graph.py (make graph) 更新图谱数据
2. 调用 generate_home_stats.py 更新首页轻量级统计 JSON
3. 自动同步数据文件到 docs/exports/
4. 更新 README.md 中的 Badges 和最后更新时间戳
5. 更新 docs/index.html 中的硬编码 Hero 统计数据
"""

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = REPO_ROOT / "docs" / "index.html"
README_MD = REPO_ROOT / "README.md"
HOME_STATS_JSON = REPO_ROOT / "exports" / "home-stats.json"


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
    # 1. 生成图谱数据和统计
    run_command(["python3", "scripts/generate_link_graph.py"], "生成图谱数据")

    # 2. 生成首页统计 JSON (内部会调用 lint_wiki.py)
    run_command(["python3", "scripts/generate_home_stats.py"], "生成首页统计 JSON")

    # 3. 确保 docs/exports 目录存在并同步文件
    docs_exports = REPO_ROOT / "docs" / "exports"
    docs_exports.mkdir(parents=True, exist_ok=True)

    files_to_sync = ["link-graph.json", "graph-stats.json", "home-stats.json"]
    for f in files_to_sync:
        src = REPO_ROOT / "exports" / f
        dst = docs_exports / f
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"✅ 已同步: {f} -> docs/exports/")

    # 读取最新统计数据
    if not HOME_STATS_JSON.exists():
        print(f"❌ 找不到统计文件: {HOME_STATS_JSON}")
        sys.exit(1)

    stats = json.loads(HOME_STATS_JSON.read_text(encoding="utf-8"))
    nodes = stats["node_count"]
    edges = stats["edge_count"]
    cov_done = stats["coverage"]["covered"]
    cov_total = stats["coverage"]["total"]
    cov_pct = stats["coverage"]["percent"]

    # 4. 更新 README.md
    if README_MD.exists():
        print("📝 更新 README.md...")
        content = README_MD.read_text(encoding="utf-8")

        # 更新 Badge
        graph_badge = f"[![Knowledge Graph](https://img.shields.io/badge/知识图谱-{nodes}节点_{edges}边-blue?logo=d3.js)]"
        content = re.sub(
            r"\[!\[Knowledge Graph\]\([^)]+\)\]\([^)]+\)",
            f"{graph_badge}(https://imchong.github.io/Robotics_Notebooks/graph.html)",
            content,
        )

        cov_color = "green" if cov_pct >= 90 else "yellow"
        cov_badge = f"[![Sources Coverage](https://img.shields.io/badge/sources覆盖率-{cov_pct}%25-{cov_color})]"
        content = re.sub(
            r"\[!\[Sources Coverage\]\([^)]+\)\]\([^)]+\)",
            f"{cov_badge}(docs/checklists/tech-stack-next-phase-checklist-v21.md)",
            content,
        )

        # 更新时间戳注释
        today_str = date.today().isoformat()
        content = re.sub(
            r"<!-- Last updated: .* \(V\d+ .*：图谱 \d+ 节点 \d+ 边\) -->",
            f"<!-- Last updated: {today_str} (V21 自动更新：图谱 {nodes} 节点 {edges} 边) -->",
            content,
        )

        README_MD.write_text(content, encoding="utf-8")
        print("✅ README.md 更新完成")

    # 5. 更新 docs/index.html (硬编码部分)
    if INDEX_HTML.exists():
        print("📝 更新 docs/index.html...")
        content = INDEX_HTML.read_text(encoding="utf-8")

        # 匹配 <span id="heroNodeCount">...</span> 等
        new_stats_html = (
            f'<div class="hero-stat-row-mini" aria-label="知识库当前规模">\n'
            f'            <span id="heroNodeCount">{nodes}</span> Nodes ·\n'
            f'            <span id="heroEdgeCount">{edges}</span> Links ·\n'
            f'            <span id="heroCoverageCount">{cov_done}/{cov_total}</span> Sources\n'
            f"          </div>"
        )

        content = re.sub(
            r'<div class="hero-stat-row-mini" aria-label="知识库当前规模">.*?</div>',
            new_stats_html,
            content,
            flags=re.DOTALL,
        )

        INDEX_HTML.write_text(content, encoding="utf-8")
        print("✅ docs/index.html 更新完成")

    print(
        f"\n✨ 所有统计数据同步完成！当前状态: {nodes} Nodes, {edges} Edges, Coverage {cov_done}/{cov_total}"
    )


if __name__ == "__main__":
    main()
