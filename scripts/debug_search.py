#!/usr/bin/env python3
"""
debug_search.py — 搜索权重调试工具
输入查询词，显式输出每个结果的原始得分、提权系数及最终排名原因。
"""

import sys
import json
from pathlib import Path
from search_wiki import search, tokenize_text

def debug_query(query_text):
    print(f"\n🔍 调试查询: '{query_text}'")
    print("=" * 60)
    
    query_words = query_text.split()
    results, notice = search(
        query_words=query_words,
        type_filter=None,
        tag_filters=[],
        context_lines=0,
        case_sensitive=False,
        show_related=False,
        semantic=False
    )
    
    if notice:
        print(f"提示: {notice}")

    for i, r in enumerate(results[:10], 1):
        path = r["path"]
        page_type = r["fm"].get("type", "unknown")
        score = r["score"]
        bm25 = r.get("bm25_score", 0)
        
        # 计算系数（根据 search_wiki.py 逻辑反推）
        boost = 1.0
        if page_type == "comparison":
            boost = 1.3
        elif page_type == "query":
            boost = 0.7
            
        print(f"{i}. [{score:.4f}] {path}")
        print(f"   类型: {page_type} (系数: {boost}) | 原始 BM25: {bm25:.4f}")
        
        # 命中词分析
        content = (Path(__file__).resolve().parent.parent / path).read_text(encoding="utf-8").lower()
        query_tokens = tokenize_text(query_text)
        hits = [t for t in query_tokens if t in content]
        print(f"   命中词: {hits}")
        print("-" * 40)

def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/debug_search.py <查询词>")
        sys.exit(1)
    
    debug_query(" ".join(sys.argv[1:]))

if __name__ == "__main__":
    main()
