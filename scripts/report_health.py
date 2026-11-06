#!/usr/bin/env python3
"""打印 wiki 页的节点健康度（0-3），供 ingest PR 正文汇报。

用法：python3 scripts/report_health.py wiki/entities/paper-x.md [...]
      python3 scripts/report_health.py --changed   # 相对 origin/main 改动/新增的 wiki 页
判据见 generate_link_graph.compute_health_score；缺项给出原因。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_link_graph as g  # noqa: E402
from original_reading import original_fully_read  # noqa: E402


def changed_pages() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", "origin/main...HEAD", "--", "wiki"],
        capture_output=True, text=True, check=True, cwd=g.REPO_ROOT,
    ).stdout
    return [p for p in out.split() if p.endswith(".md")]


def main(argv: list[str]) -> int:
    pages = changed_pages() if argv == ["--changed"] else argv
    if not pages:
        print("无页面")
        return 0
    print("| 页面 | 健康度 | 缺项 |\n|---|---|---|")
    for p in pages:
        path = (g.REPO_ROOT / p).resolve()
        content = path.read_text(encoding="utf-8")
        score = g.compute_health_score(content, path)
        miss = []
        if not g.has_reading_skeleton(content, g.parse_frontmatter_type(content)):
            miss.append("阅读骨架")
        if not original_fully_read(content, path, g.SOURCES_DIR):
            miss.append("原文未通读")
        if "status: draft" in content or "status: stub" in content:
            miss.append("status 未 complete")
        print(f"| {p} | {score}/3 | {'、'.join(miss) or '—'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
