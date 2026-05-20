#!/usr/bin/env python3
"""
append_log.py — 向 log.md 顶部插入一条操作记录（新记录在上，与首页 latest_wiki_nodes 解析一致）

用法:
    python3 scripts/append_log.py <op> "<描述>"

    <op>: ingest | query | lint | index | structural

示例:
    python3 scripts/append_log.py ingest "sources/papers/mpc.md — Mayne 2000 等 5 篇"
    python3 scripts/append_log.py lint "0 issues，覆盖率 75%"
    python3 scripts/append_log.py query "locomotion reward → wiki/queries/locomotion-reward-design-guide.md"
"""

import sys
from datetime import date
from pathlib import Path

VALID_OPS = {"ingest", "query", "lint", "index", "structural"}

LOG_PATH = Path(__file__).resolve().parent.parent / "log.md"


def prepend_log_entry(text: str, entry: str) -> str:
    """在首条 `## [日期]` 日志标题之前插入 entry（保留文件顶部说明行）。"""
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            insert_at = i
            break
    else:
        insert_at = len(lines)
    if insert_at > 0 and lines[insert_at - 1].strip() == "":
        # 保持说明段与首条日志之间的空行
        pass
    elif insert_at > 0 and not lines[insert_at - 1].endswith("\n\n"):
        entry = "\n" + entry
    return "".join(lines[:insert_at]) + entry + "".join(lines[insert_at:])


def main() -> None:
    if len(sys.argv) < 3:
        print('用法: python3 scripts/append_log.py <op> "<描述>"', file=sys.stderr)
        print(f"  op 可选值: {', '.join(sorted(VALID_OPS))}", file=sys.stderr)
        sys.exit(1)

    op = sys.argv[1].strip().lower()
    desc = sys.argv[2].strip()

    if op not in VALID_OPS:
        print(f"⚠️  未知 op '{op}'，有效值: {', '.join(sorted(VALID_OPS))}", file=sys.stderr)
        sys.exit(1)

    if not desc:
        print("错误：描述不能为空", file=sys.stderr)
        sys.exit(1)

    today = date.today().isoformat()
    entry = f"## [{today}] {op} | {desc}\n\n"

    if LOG_PATH.is_file():
        text = LOG_PATH.read_text(encoding="utf-8")
    else:
        text = "> 核心规范：所有日常动作（ingest / query / lint / structural）必须追加记录到此文件。\n\n"

    LOG_PATH.write_text(prepend_log_entry(text, entry), encoding="utf-8")

    print(f"✅ 已插入 log.md 顶部: [{today}] {op} | {desc}")


if __name__ == "__main__":
    main()
