#!/usr/bin/env python3
"""
append_log.py — 向 log.md 追加一条操作记录

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
    entry = f"\n## [{today}] {op} | {desc}\n"

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)

    print(f"✅ 已追加到 log.md: [{today}] {op} | {desc}")


if __name__ == "__main__":
    main()
