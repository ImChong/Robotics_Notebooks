#!/usr/bin/env python3
"""把 log.d/ 下的日志碎片并入 log.md 顶部并删除碎片（由 main 上的 export.yml 调用）。"""

from log_md import fold_log_fragments


def main() -> None:
    folded = fold_log_fragments()
    if not folded:
        print("log.d/ 无待并入碎片")
        return
    for path in folded:
        print(f"✅ 已并入 log.md: {path.name}")


if __name__ == "__main__":
    main()
