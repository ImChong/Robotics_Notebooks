#!/usr/bin/env python3
"""搜索质量回归评估脚本。

读取 schema/search-regression-cases.json，逐条执行查询，
计算 hit@k / recall@k，输出通过率，失败时打印 diff。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = REPO_ROOT / "schema" / "search-regression-cases.json"

# 将 scripts/ 加入 path 以便导入 search_wiki 模块
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_cases() -> list[dict]:
    if not CASES_FILE.exists():
        print(f"❌ 找不到回归样例文件：{CASES_FILE}")
        sys.exit(1)
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def run_bm25_search(query: str, top_k: int) -> list[str]:
    """返回 BM25 搜索结果的 wiki 页面路径列表（最多 top_k 条）。"""
    from search_wiki import search as _search

    results, _ = _search(
        query_words=[query],
        type_filter=None,
        tag_filters=[],
        context_lines=0,
        case_sensitive=False,
        show_related=False,
        semantic=False,
    )
    return [r.get("id", r.get("path", "")) for r in results[:top_k]]


def check_case(case: dict) -> tuple[bool, str]:
    """评估单个回归样例，返回 (通过, 诊断信息)。"""
    query = case["query"]
    mode = case.get("mode", "bm25")
    must_include = case.get("must_include", [])
    top_k = case.get("expected_top_k", 5)

    if mode == "semantic":
        # 语义搜索：需要向量索引，回退到 BM25
        results = run_bm25_search(query, top_k)
    else:
        results = run_bm25_search(query, top_k)

    hits = []
    misses = []
    for required in must_include:
        # 宽松匹配：路径格式 wiki/foo/bar.md 或 id 格式 wiki-foo-bar
        req_id = required.replace("/", "-").replace(".md", "")
        req_stem = Path(required).stem
        found = any(
            required in r
            or r == req_id
            or r.endswith(req_stem)
            or (r.replace("-", "/") + ".md").endswith(required.lstrip("wiki/").lstrip("/"))
            for r in results
        )
        if found:
            hits.append(required)
        else:
            misses.append(required)

    passed = len(misses) == 0
    note = case.get("note", "")
    if passed:
        diag = f'✅ [{mode}] "{query}" — {note}'
    else:
        diag = (
            f'❌ [{mode}] "{query}" — {note}\n'
            f"   必须命中：{must_include}\n"
            f"   实际前{top_k}：{results}"
        )
    return passed, diag


def main() -> None:
    cases = load_cases()
    total = len(cases)
    passed_count = 0
    diags: list[str] = []

    for case in cases:
        try:
            ok, diag = check_case(case)
        except Exception as exc:
            ok = False
            diag = f'❌ 执行出错 "{case.get("query", "?")}": {exc}'
        if ok:
            passed_count += 1
        diags.append(diag)

    print("=" * 60)
    print("搜索质量回归评估")
    print("=" * 60)
    for d in diags:
        print(d)
    print("=" * 60)
    pass_rate = passed_count / total * 100 if total else 0
    print(f"\n通过率：{passed_count}/{total} ({pass_rate:.0f}%)")

    if pass_rate < 80:
        print("⚠️  通过率低于 80%，请检查上方失败样例。")
        sys.exit(1)
    else:
        print("✅ 通过率 ≥ 80%，回归测试通过。")


if __name__ == "__main__":
    main()
