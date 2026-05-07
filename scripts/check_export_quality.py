#!/usr/bin/env python3
"""导出质量检查脚本。

检查 exports/ 和 docs/ 目录下的输出文件是否完整、一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS = REPO_ROOT / "exports"
DOCS_EXPORTS = REPO_ROOT / "docs" / "exports"
DOCS = REPO_ROOT / "docs"


def check(name: str, passed: bool, detail: str = "") -> bool:
    icon = "✅" if passed else "❌"
    msg = f"{icon} {name}"
    if detail:
        msg += f"：{detail}"
    print(msg)
    return passed


def main() -> None:
    print("=" * 55)
    print("导出质量检查")
    print("=" * 55)

    results: list[bool] = []

    # 1. search-index.json 存在且非空
    si = DOCS / "search-index.json"
    if si.exists():
        data = json.loads(si.read_text(encoding="utf-8"))
        doc_count = len(data.get("docs", []))
        results.append(check("search-index.json 存在", True, f"{doc_count} 文档"))
        results.append(check("search-index.json 非空", doc_count > 0, f"docs={doc_count}"))
    else:
        results.append(check("search-index.json 存在", False, "文件缺失"))
        results.append(check("search-index.json 非空", False, "文件缺失"))

    # 2. index-v1.json 存在且文档数合理
    iv1 = EXPORTS / "index-v1.json"
    if iv1.exists():
        data = json.loads(iv1.read_text(encoding="utf-8"))
        items = data.get("items", data if isinstance(data, list) else [])
        count = len(items)
        results.append(check("exports/index-v1.json 存在", True, f"{count} 页面"))
        results.append(check("index-v1.json 文档数合理（≥50）", count >= 50, f"count={count}"))
    else:
        results.append(check("exports/index-v1.json 存在", False, "文件缺失"))
        results.append(check("index-v1.json 文档数合理", False, "文件缺失"))

    # 3. search-index.json 与 index-v1.json 文档数量大体一致
    if si.exists() and iv1.exists():
        si_data = json.loads(si.read_text(encoding="utf-8"))
        iv1_data = json.loads(iv1.read_text(encoding="utf-8"))
        si_count = len(si_data.get("docs", []))
        iv1_items = iv1_data.get("items", iv1_data if isinstance(iv1_data, list) else [])
        iv1_count = len(iv1_items)
        # 允许 ±20% 差异（search-index 只含 wiki pages，index-v1 含全部页面）
        ratio = si_count / max(iv1_count, 1)
        consistent = 0.3 <= ratio <= 1.05
        results.append(
            check(
                "search-index 与 index-v1 数量合理",
                consistent,
                f"search={si_count} index={iv1_count} ratio={ratio:.2f}",
            )
        )

    # 4. docs/exports/ 与 exports/ 关键文件同步
    key_files = ["link-graph.json", "graph-stats.json", "site-data-v1.json"]
    for fname in key_files:
        src = EXPORTS / fname
        dst = DOCS_EXPORTS / fname
        if not src.exists():
            results.append(check(f"docs/exports/{fname} 同步", False, "源文件不存在"))
            continue
        if not dst.exists():
            results.append(
                check(
                    f"docs/exports/{fname} 同步", False, "目标文件不存在，请运行 make graph/export"
                )
            )
            continue
        src_size = src.stat().st_size
        dst_size = dst.stat().st_size
        in_sync = abs(src_size - dst_size) <= max(src_size * 0.01, 100)
        results.append(
            check(f"docs/exports/{fname} 同步", in_sync, f"src={src_size}B dst={dst_size}B")
        )

    # 5. graph-stats.json 节点数与 wiki 页面数大体一致
    gs = EXPORTS / "graph-stats.json"
    wiki_pages = list((REPO_ROOT / "wiki").rglob("*.md"))
    wiki_count = len([p for p in wiki_pages if p.name != "README.md"])
    if gs.exists():
        gs_data = json.loads(gs.read_text(encoding="utf-8"))
        node_count = gs_data.get("node_count", 0)
        ratio = node_count / max(wiki_count, 1)
        reasonable = 0.5 <= ratio <= 1.1
        results.append(
            check(
                "graph 节点数与 wiki 页面数合理",
                reasonable,
                f"nodes={node_count} wiki_pages={wiki_count} ratio={ratio:.2f}",
            )
        )
    else:
        results.append(check("graph-stats.json 存在", False, "文件缺失"))

    # 6. graph-stats.json 中孤儿节点应为空
    if gs.exists():
        gs_data = json.loads(gs.read_text(encoding="utf-8"))
        orphan_nodes = gs_data.get("orphan_nodes", [])
        orphan_free = len(orphan_nodes) == 0
        detail = "0 个孤儿节点" if orphan_free else f"⚠️ 发现 {len(orphan_nodes)} 个孤儿节点"
        results.append(check("graph-stats.json 无孤儿节点", orphan_free, detail))
    else:
        results.append(check("graph-stats.json 无孤儿节点", False, "文件缺失"))

    # 7. lint-report.md 存在（weekly action 已生成过）
    lr = EXPORTS / "lint-report.md"
    results.append(
        check("exports/lint-report.md 存在", lr.exists(), "（首次运行 weekly action 后生成）")
    )

    # 8. index.md 同步检查（Karpathy: LLM updates index on every ingest）
    index_md = REPO_ROOT / "index.md"
    index_json = EXPORTS / "index-v1.json"
    if index_md.exists() and index_json.exists():
        md_mtime = index_md.stat().st_mtime
        json_mtime = index_json.stat().st_mtime
        # lag_days > 0: json 比 index.md 新；可能 index.md 未及时更新
        lag_days = (json_mtime - md_mtime) / 86400
        # 允许 7 天内的偏差（硬失败阈值），超过则提示但不阻止 CI
        in_sync = lag_days <= 7.0
        detail = (
            f"⚠️ index-v1.json 比 index.md 新了 {lag_days:.1f} 天（建议 make catalog | head -200 >> index.md）"
            if not in_sync
            else f"同步正常（差 {lag_days:.1f} 天）"
        )
        results.append(check("index.md 与 exports/index-v1.json 同步（7 天内）", in_sync, detail))

    print("=" * 55)
    passed = sum(results)
    total = len(results)
    print(f"\n通过：{passed}/{total}")

    if passed < total:
        print(f"⚠️  {total - passed} 项检查未通过，请处理上方 ❌ 项。")
        sys.exit(1)
    else:
        print("✅ 所有导出质量检查通过。")


if __name__ == "__main__":
    main()
