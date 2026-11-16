#!/usr/bin/env python3
"""导出质量检查脚本。

检查 exports/ 和 docs/ 目录下的输出文件是否完整、一致。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from export_minimal import BODY_ONLY_FIELDS, CATALOG_DROPPED_FIELDS

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


def validate_page_exports(
    source_dir: Path,
    mirror_dir: Path,
    *,
    expected_site_data: dict | None = None,
) -> int:
    """Verify the split site-catalog-v1 + page-content contract and exact mirror."""
    catalog_bytes = (source_dir / "site-catalog-v1.json").read_bytes()
    if catalog_bytes != (mirror_dir / "site-catalog-v1.json").read_bytes():
        raise ValueError("目录镜像不一致")
    catalog = json.loads(catalog_bytes)
    if catalog.pop("content_mode", None) != "per-page-v1":
        raise ValueError("目录版本不正确")
    pages = catalog.get("pages")
    details = pages.get("detail_pages") if isinstance(pages, dict) else None
    if not isinstance(details, dict):
        raise ValueError("缺少 pages.detail_pages 目录")
    if expected_site_data is not None and set(details) != set(
        expected_site_data["pages"]["detail_pages"]
    ):
        raise ValueError("页面 ID 集合不一致")
    expected_files = set()
    for page_id, page in details.items():
        url = page.pop("content_url", "")
        if not re.fullmatch(r"exports/page-content/[a-f0-9]{64}\.json", url):
            raise ValueError(f"{page_id}: 正文路径不合法")
        filename = url.rsplit("/", 1)[-1]
        body_bytes = (source_dir / "page-content" / filename).read_bytes()
        if filename != f"{hashlib.sha256(body_bytes).hexdigest()}.json":
            raise ValueError(f"{page_id}: 正文哈希不匹配")
        if body_bytes != (mirror_dir / "page-content" / filename).read_bytes():
            raise ValueError(f"{page_id}: 正文镜像不一致")
        body = json.loads(body_bytes)
        if (
            body.get("version") != "v1"
            or body.get("id") != page_id
            or not isinstance(body.get("content_markdown"), str)
            or any(key in page for key in CATALOG_DROPPED_FIELDS)
        ):
            raise ValueError(f"{page_id}: 正文字段契约不正确")
        # 目录键 + 正文 JSON 携带的字段合起来必须能无损还原完整条目
        page["id"] = page_id
        page["content_markdown"] = body["content_markdown"]
        for key in BODY_ONLY_FIELDS:
            if key in body:
                page[key] = body[key]
        expected_files.add(filename)
    if expected_site_data is not None and catalog != expected_site_data:
        raise ValueError("拆分后不能无损重建完整站点数据")
    for directory in (source_dir, mirror_dir):
        if {p.name for p in (directory / "page-content").glob("*.json")} != expected_files:
            raise ValueError("存在未引用或缺失的正文文件")
    return len(details)


def main() -> None:
    print("=" * 55)
    print("导出质量检查")
    print("=" * 55)

    results: list[bool] = []
    try:
        page_count = validate_page_exports(EXPORTS, DOCS_EXPORTS)
        results.append(check("按页正文 ID、哈希、契约与镜像一致", True, f"{page_count} 页面"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        results.append(check("按页正文 ID、哈希、契约与镜像一致", False, str(error)))

    # 1. search-index.json 存在且非空
    si = DOCS / "search-index.json"
    si_data: dict | None = None
    doc_count = 0
    if si.exists():
        si_data = json.loads(si.read_text(encoding="utf-8"))
        doc_count = len(si_data.get("docs", []))
        results.append(check("search-index.json 存在", True, f"{doc_count} 文档"))
        results.append(check("search-index.json 非空", doc_count > 0, f"docs={doc_count}"))
    else:
        results.append(check("search-index.json 存在", False, "文件缺失"))
        results.append(check("search-index.json 非空", False, "文件缺失"))
    # 1b. 轻量索引与全文索引文档数一致（首页搜索默认只加载轻量索引）
    lite = DOCS / "search-index-lite.json"
    if lite.exists() and si.exists():
        lite_count = len(json.loads(lite.read_text(encoding="utf-8")).get("docs", []))
        results.append(
            check(
                "search-index-lite.json 与全文索引文档数一致",
                lite_count == doc_count,
                f"lite={lite_count} full={doc_count}",
            )
        )
    else:
        results.append(check("search-index-lite.json 存在", False, "文件缺失"))

    # 2. site-catalog-v1.json 存在且文档数合理
    sc = EXPORTS / "site-catalog-v1.json"
    sc_data: dict | None = None
    sc_count = 0
    if sc.exists():
        sc_data = json.loads(sc.read_text(encoding="utf-8"))
        sc_pages = sc_data.get("pages", {}).get("detail_pages", {})
        sc_count = len(sc_pages)
        results.append(check("exports/site-catalog-v1.json 存在", True, f"{sc_count} 页面"))
        results.append(
            check("site-catalog-v1.json 文档数合理（≥50）", sc_count >= 50, f"count={sc_count}")
        )
    else:
        results.append(check("exports/site-catalog-v1.json 存在", False, "文件缺失"))
        results.append(check("site-catalog-v1.json 文档数合理", False, "文件缺失"))

    # 3. search-index.json 与 site-catalog-v1.json 文档数量大体一致
    if si_data is not None and sc_data is not None:
        si_count = doc_count
        # search-index 与 site-catalog-v1 的 detail_pages 数量基本一致（允许少量差）
        ratio = si_count / max(sc_count, 1)
        consistent = 0.8 <= ratio <= 1.05
        results.append(
            check(
                "search-index 与 site-catalog-v1 数量合理",
                consistent,
                f"search={si_count} catalog={sc_count} ratio={ratio:.2f}",
            )
        )

    # 4. docs/exports/ 与 exports/ 关键文件同步
    #    站点只镜像 site-catalog-v1.json + page-content/；index-v1 / site-data-v1 不再进 docs/
    key_files = [
        "link-graph.json",
        "link-graph-summaries.json",
        "graph-stats.json",
        "site-catalog-v1.json",
    ]
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
    gs_data: dict | None = None
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
    if gs_data is not None:
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
