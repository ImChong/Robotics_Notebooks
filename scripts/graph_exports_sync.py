#!/usr/bin/env python3
"""Copy graph-related JSON from exports/ to docs/exports/ for GitHub Pages.

Single place for filenames and copy logic: used by ``make graph``,
``sync_all_stats.py``, and ``sync_wiki.sh`` to avoid drift.
"""

from __future__ import annotations

import json
from pathlib import Path

GRAPH_EXPORT_FILES: tuple[str, ...] = (
    "link-graph.json",
    "graph-stats.json",
    "home-stats.json",
    "wiki-activity.json",
    "hub-rankings.json",
)

# 条目更新（新增日 / 最近触达日）的权威口径是 git commit 历史：
# generate_link_graph 的「新建日完全以 git 首次加入日为准」，log.md 只是
# 浅克隆下的回退。回退产物只覆盖日志里显式写过的路径，会丢失仅存在于 git
# 历史的条目（首页「最近更新」少条目、recency 偏旧）。docs/exports/ 是要
# 提交并上线的产物，降级数据同步进去就会当成正常数据发布，故在此拦下。
ACTIVITY_FILE = "wiki-activity.json"
REQUIRED_ACTIVITY_SOURCE = "git"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def activity_source(root: Path) -> str | None:
    """``exports/wiki-activity.json`` 里 generate_link_graph 记的 recency 来源。

    文件缺失或不可解析时返回 ``None``：此时本轮没有新生成图谱产物
    （``sync_wiki.sh`` 式的分步流程），无从判断降级与否，交由调用方。
    """
    src = root / "exports" / ACTIVITY_FILE
    if not src.exists():
        return None
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload.get("source") if isinstance(payload, dict) else None


def assert_git_driven_activity(root: Path) -> None:
    """同步前断言条目更新仍由 git commit 历史驱动，否则拒绝同步。

    仅在确实读到降级来源时抛错；来源缺失不阻塞（见 :func:`activity_source`）。
    """
    source = activity_source(root)
    if source is None or source == REQUIRED_ACTIVITY_SOURCE:
        return
    raise SystemExit(
        f"❌ 拒绝同步 docs/exports/：条目更新口径已降级为 {source}"
        f"（应为 {REQUIRED_ACTIVITY_SOURCE}）。\n"
        f"   exports/{ACTIVITY_FILE} 的 source={source}，说明 generate_link_graph "
        "检测到浅克隆后跳过了 git 活动采集并回退 log.md；\n"
        "   这份 recency 只覆盖日志里显式写过的路径，会丢失仅存在于 git 历史的条目。\n"
        "   修复：git fetch --unshallow && make graph"
        "（CI 侧确认 actions/checkout 的 fetch-depth: 0）。"
    )


def copy_graph_exports_to_docs() -> None:
    root = repo_root()
    assert_git_driven_activity(root)
    dst_dir = root / "docs" / "exports"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_EXPORT_FILES:
        src = root / "exports" / name
        dst = dst_dir / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"✅ 已同步: {name} -> docs/exports/")


def main() -> None:
    copy_graph_exports_to_docs()


if __name__ == "__main__":
    main()
