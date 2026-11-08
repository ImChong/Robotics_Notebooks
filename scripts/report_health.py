#!/usr/bin/env python3
"""打印 wiki 页的节点健康度（0-3），供 ingest PR 正文汇报。

用法：python3 scripts/report_health.py wiki/entities/paper-x.md [...]
      python3 scripts/report_health.py --changed   # 相对 origin/main 新增/改动/重命名的 wiki 页
      python3 scripts/report_health.py --check-pr-body   # CI：比对环境变量 PR_BODY 里的汇报表
判据见 generate_link_graph.compute_health_score；缺项给出原因。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_link_graph as g  # noqa: E402
from original_reading import original_fully_read  # noqa: E402

HealthRow = tuple[str, int, list[str]]

# 正文表格的健康度单元格须是脚本原样输出的 N/3（允许加粗），「预计 3/3」之类不算运行过
SCORE_CELL_RE = re.compile(r"\**\s*([0-3])\s*/\s*3\s*\**")
PAGE_IN_CELL_RE = re.compile(r"wiki/[^\s`)\]|]+\.md")


def changed_pages() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", "origin/main...HEAD", "--", "wiki"],
        capture_output=True,
        text=True,
        check=True,
        cwd=g.REPO_ROOT,
    ).stdout
    return [p for p in out.split() if p.endswith(".md")]


def health_rows(pages: list[str]) -> list[HealthRow]:
    rows = []
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
        rows.append((p, score, miss))
    return rows


def format_table(rows: list[HealthRow]) -> str:
    lines = ["| 页面 | 健康度 | 缺项 |", "|---|---|---|"]
    lines += [f"| {p} | {score}/3 | {'、'.join(miss) or '—'} |" for p, score, miss in rows]
    return "\n".join(lines)


def reported_rows(body: str) -> dict[str, list[tuple[int | None, str]]]:
    """正文 Markdown 表格里每个 wiki 页的（分数, 缺项列文字）；分数单元格不规范时为 None。"""
    reported: dict[str, list[tuple[int | None, str]]] = {}
    for line in body.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        page = PAGE_IN_CELL_RE.search(cells[0]) if len(cells) >= 2 else None
        if not page:
            continue
        score = SCORE_CELL_RE.fullmatch(cells[1])
        reported.setdefault(page.group(0), []).append(
            (int(score.group(1)) if score else None, " ".join(cells[2:]))
        )
    return reported


def has_own_reason(reason: str, miss: list[str]) -> bool:
    """缺项列除脚本自动生成的缺项名外，还须有作者写的原因文字。"""
    for item in miss:
        reason = reason.replace(item, "")
    return bool(re.search(r"\w", reason))


def check_pr_body(body: str, rows: list[HealthRow]) -> list[str]:
    reported = reported_rows(body)
    problems = []
    for page, score, miss in rows:
        entries = reported.get(page)
        if not entries:
            problems.append(f"{page}：正文健康度表格缺这一页（实测 {score}/3）")
            continue
        reasons = [reason for got, reason in entries if got == score]
        scores = [got for got, _ in entries if got is not None]
        if not reasons and not scores:
            problems.append(f"{page}：健康度单元格不是脚本输出的 N/3，请运行脚本并原样粘贴")
        elif not reasons:
            got = "、".join(f"{s}/3" for s in scores)
            problems.append(f"{page}：正文写 {got}，实测 {score}/3（改页后须重跑脚本并更新正文）")
        elif score < 3 and not any(has_own_reason(reason, miss) for reason in reasons):
            problems.append(f"{page}：{score}/3 未满分，缺项列须写明原因（如原文无法获取）")
    return problems


def check_pr_body_main() -> int:
    rows = health_rows(changed_pages())
    if not rows:
        print("本 PR 未新增/改动 wiki 页，无需汇报节点健康度")
        return 0
    table = format_table(rows)
    print(f"CI 实测健康度：\n{table}\n")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(f"\n### 节点健康度（CI 实测）\n\n{table}\n")
    problems = check_pr_body(os.environ.get("PR_BODY", ""), rows)
    if not problems:
        print("PR 正文的节点健康度汇报与实测一致")
        return 0
    print("PR 正文的节点健康度汇报不合格（schema/ingest-workflow.md 步骤 8.5）：")
    for problem in problems:
        print(f"- {problem}")
    print("\n把上面的实测表格贴进 PR 正文「节点健康度」小节；未满分的行在缺项列写明原因。")
    print("编辑 PR 正文会自动重跑本检查。")
    return 1


def main(argv: list[str]) -> int:
    if argv == ["--check-pr-body"]:
        return check_pr_body_main()
    pages = changed_pages() if argv == ["--changed"] else argv
    if not pages:
        print("无页面")
        return 0
    print(format_table(health_rows(pages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
