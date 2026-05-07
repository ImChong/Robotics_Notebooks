#!/usr/bin/env python3
"""
discover_facts.py — 自动化知识挖掘工具

1. 扫描 wiki/ 目录下的所有页面，利用正则表达式识别“一句话定义”、“核心假设”等区块，
   提取潜在的 CANONICAL_FACTS 候选条目供人工审核。
2. 根据 Tag 重合度自动生成相关页面 (related) 补全建议。
"""

import re
from pathlib import Path

from search_indexing import REPO_ROOT, iter_wiki_documents

WIKI_DIR = REPO_ROOT / "wiki"

# 匹配模式：常用的事实定义区块
PATTERNS = [
    # 匹配 ## 一句话定义 下的内容
    r"## 一句话定义\n\n>(.*?)\n",
    # 匹配 **名词**：定义
    r"\*\*(.*?)\*\*[:：](.*?)\n",
    # 匹配核心假设
    r"## 核心假设\n\n(.*?)(?=\n\n##|\Z)",
]


def discover(docs=None):
    candidates = []
    if docs is None:
        docs = iter_wiki_documents()

    for doc in docs:
        content = doc["body"]
        rel_path = doc["path"]

        for p in PATTERNS:
            matches = re.finditer(p, content, re.DOTALL)
            for m in matches:
                if len(m.groups()) == 1:
                    fact = m.group(1).strip()
                    term = Path(rel_path).stem.replace("-", " ").title()
                else:
                    term = m.group(1).strip()
                    fact = m.group(2).strip()

                if 10 < len(fact) < 200:
                    candidates.append({"term": term, "fact": fact, "source": rel_path})

    return candidates


def suggest_related_links(min_overlap=2, docs=None):
    suggestions = []
    if docs is None:
        docs = iter_wiki_documents()

    # 预先计算所有文档的 tag 集合和已存在的相关链接 stem
    doc_metadata = []
    for doc in docs:
        tags = set(t.lower() for t in (doc.get("tags") or []))
        existing_related = doc.get("frontmatter", {}).get("related") or []
        existing_stems = set(Path(r).stem for r in existing_related)
        doc_metadata.append(
            {
                "tags": tags,
                "existing_stems": existing_stems,
                "path": doc["path"],
                "title": doc["title"],
            }
        )

    for i, meta_a in enumerate(doc_metadata):
        path_a = meta_a["path"]
        tags_a = meta_a["tags"]
        existing_stems_a = meta_a["existing_stems"]

        doc_suggestions = []
        for j, meta_b in enumerate(doc_metadata):
            if i == j:
                continue

            overlap = tags_a.intersection(meta_b["tags"])
            if len(overlap) >= min_overlap:
                path_b = meta_b["path"]
                stem_b = Path(path_b).stem
                if stem_b not in existing_stems_a:
                    doc_suggestions.append(
                        {"path": path_b, "title": meta_b["title"], "overlap": list(overlap)}
                    )

        if doc_suggestions:
            suggestions.append(
                {
                    "source": path_a,
                    "title": meta_a["title"],
                    "targets": sorted(
                        doc_suggestions, key=lambda x: len(x["overlap"]), reverse=True
                    )[:5],
                }
            )

    return suggestions


def main():
    print("🔍 正在读取 Wiki 文档...")
    docs = iter_wiki_documents()

    print("🔍 正在扫描 Wiki 以挖掘新事实...")
    candidates = discover(docs=docs)

    print(f"✅ 发现 {len(candidates)} 条潜在事实候选：")
    for i, c in enumerate(candidates[:20], 1):
        print(f"{i}. [{c['term']}] {c['fact']}")
        print(f"   来源: {c['source']}\n")
    if len(candidates) > 20:
        print(f"... 以及另外 {len(candidates) - 20} 条候选。\n")

    print("🔗 正在分析 Tag 重合度以推荐内链补全...")
    related_suggestions = suggest_related_links(docs=docs)
    print(f"✅ 发现 {len(related_suggestions)} 个页面有内链补全潜力：")
    for s in related_suggestions[:10]:
        print(f"📄 {s['title']} ({s['source']}) 建议增加：")
        for target in s["targets"]:
            print(f"   → {target['title']} (重合 Tags: {', '.join(target['overlap'])})")
        print()

    print("-" * 60)
    print("建议：挑选上述事实加入 scripts/lint_wiki.py，或根据推荐补全页面 related 区块。")


if __name__ == "__main__":
    main()
