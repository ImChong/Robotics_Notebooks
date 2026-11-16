#!/usr/bin/env python3
"""把 log.d/ 下的日志碎片并入 log.md 顶部并删除碎片（由 main 上的 export.yml 调用）。"""

from log_md import fold_log_fragments, rotate_old_log_entries


def main() -> None:
    folded = fold_log_fragments()
    if not folded:
        print("log.d/ 无待并入碎片")
    else:
        for path in folded:
            print(f"✅ 已并入 log.md: {path.name}")
    archived = rotate_old_log_entries()
    for path in archived:
        print(f"📦 已按季度归档历史日志: {path.name}")


if __name__ == "__main__":
    main()
