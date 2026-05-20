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

from log_md import DEFAULT_LOG_PATH, write_log_prepend

VALID_OPS = {"ingest", "query", "lint", "index", "structural"}


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
    write_log_prepend(entry, DEFAULT_LOG_PATH)

    print(f"✅ 已插入 log.md 顶部: [{today}] {op} | {desc}")


if __name__ == "__main__":
    main()
