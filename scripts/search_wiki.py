#!/usr/bin/env python3
"""
search_wiki.py — wiki 内容搜索 CLI（实现见 search_wiki_core）。
支持 BM25 / 关联页面 / 本地混合语义搜索。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from search_indexing import iter_wiki_documents
from search_wiki_core import (
    _cache_key,
    _filter_doc,
    _find_matched_lines,
    collect_known_terms,
    load_cache,
    print_results,
    save_cache,
    search,
    suggest_terms,
)

# eval_search_quality 等从本模块导入 `search`

# 单元测试仍从本模块导入以下符号
__all__ = ["_filter_doc", "_find_matched_lines", "main", "search"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="搜索 Robotics_Notebooks wiki 内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python3 scripts/search_wiki.py "MPC locomotion"
  python3 scripts/search_wiki.py "diffusion" --type method
  python3 scripts/search_wiki.py --tag rl --tag humanoid
  python3 scripts/search_wiki.py "稳定性" --semantic
        """,
    )
    parser.add_argument("query", nargs="*", help="搜索关键词（多词默认作为整体查询）")
    parser.add_argument(
        "--type", dest="type_filter", help="按页面类型过滤（concept/method/task/comparison/...）"
    )
    parser.add_argument(
        "--tag",
        dest="tag_filters",
        action="append",
        default=[],
        help="按标签过滤（可多次指定，AND 逻辑）",
    )
    parser.add_argument("--context", type=int, default=1, help="每处匹配显示的上下文行数（默认 1）")
    parser.add_argument("--case", action="store_true", help="区分大小写（仅影响高亮显示）")
    parser.add_argument("--related", action="store_true", help="同时输出每个匹配页面的关联页面")
    parser.add_argument("--semantic", action="store_true", help="启用本地混合 BM25 + 向量搜索")
    parser.add_argument("--json", dest="json_out", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    if not args.query and not args.type_filter and not args.tag_filters:
        parser.print_help()
        sys.exit(0)

    cache = load_cache()
    cache_key = _cache_key(args.query, args.type_filter, args.tag_filters, args.semantic)
    cached_entry = next((e for e in cache.get("entries", []) if e.get("key") == cache_key), None)
    if cached_entry and not args.json_out:
        print(f"\033[2m[缓存命中：{cached_entry['ts']}]\033[0m")

    results, semantic_notice = search(
        query_words=args.query,
        type_filter=args.type_filter,
        tag_filters=args.tag_filters,
        context_lines=args.context,
        case_sensitive=args.case,
        show_related=args.related,
        semantic=args.semantic,
    )
    if args.query:
        save_cache(cache, cache_key, results)

    suggestions: list[tuple[str, int]] = []
    if not results and args.query:
        suggestions = suggest_terms(
            " ".join(args.query), collect_known_terms(iter_wiki_documents())
        )

    if args.json_out:
        out = []
        for r in results:
            snippet = ""
            if r["matches"]:
                _, ctx_lines, offset = r["matches"][0]
                snippet = ctx_lines[offset] if offset < len(ctx_lines) else ""
            out.append(
                {
                    "path": str(r["path"]),
                    "score": round(r.get("score", 0.0), 6),
                    "rerank_score": round(r.get("bm25_score", r.get("score", 0.0)), 6),
                    "vector_score": round(r.get("vector_score", 0.0), 6),
                    "hybrid_score": round(r.get("hybrid_score", r.get("score", 0.0)), 6),
                    "type": r["fm"].get("type", r.get("page_type", "")),
                    "tags": r["fm"].get("tags", []),
                    "title": r.get("title", ""),
                    "snippet": snippet.strip()[:200],
                }
            )
        suggestion_payload = [{"term": term, "distance": dist} for term, dist in suggestions]
        json_payload: list[dict[str, Any]] | dict[str, Any]
        if semantic_notice or suggestion_payload:
            json_payload = {
                "notice": semantic_notice,
                "suggestions": suggestion_payload,
                "results": out,
            }
        else:
            json_payload = out
        print(json.dumps(json_payload, ensure_ascii=False, indent=2))
    else:
        print_results(
            results,
            args.query,
            args.case,
            semantic_notice=semantic_notice,
            suggestions=suggestions,
        )


if __name__ == "__main__":
    main()
