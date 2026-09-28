#!/usr/bin/env python3
"""
append_log.py — 新增一条操作记录到 log.d/ 碎片（叙事层；站点活动以 git 为准）

碎片由 main 上的 export.yml 自动并入 log.md 顶部；PR 不要直接改 log.md（避免合并冲突）。

用法:
    python3 scripts/append_log.py <op> "<描述>"

    <op>: ingest | query | lint | catalog | index | structural

示例:
    python3 scripts/append_log.py ingest "sources/papers/mpc.md — Mayne 2000 等 5 篇"
    python3 scripts/append_log.py lint "0 issues，覆盖率 75%"
    python3 scripts/append_log.py query "locomotion reward → wiki/queries/locomotion-reward-design-guide.md"
"""

import sys
from datetime import date

from log_md import write_log_fragment

VALID_OPS = {"ingest", "query", "lint", "catalog", "index", "structural"}


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
    path = write_log_fragment(entry, op, today=today)

    print(f"✅ 已写入日志碎片 {path.name}: [{today}] {op} | {desc}")


if __name__ == "__main__":
    main()
