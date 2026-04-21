#!/usr/bin/env python3
"""
discover_facts.py — 自动化知识挖掘工具

扫描 wiki/ 目录下的所有页面，利用正则表达式识别“一句话定义”、“核心假设”等区块，
提取潜在的 CANONICAL_FACTS 候选条目供人工审核。
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"

# 匹配模式：常用的事实定义区块
PATTERNS = [
    # 匹配 ## 一句话定义 下的内容
    r"## 一句话定义\n\n>(.*?)\n",
    # 匹配 **名词**：定义
    r"\*\*(.*?)\*\*[:：](.*?)\n",
    # 匹配核心假设
    r"## 核心假设\n\n(.*?)(?=\n\n##|\Z)"
]

def discover():
    candidates = []
    pages = list(WIKI_DIR.glob("**/*.md"))
    
    for page in pages:
        content = page.read_text(encoding="utf-8")
        rel_path = page.relative_to(REPO_ROOT)
        
        for p in PATTERNS:
            matches = re.finditer(p, content, re.DOTALL)
            for m in matches:
                if len(m.groups()) == 1:
                    fact = m.group(1).strip()
                    term = page.stem.replace("-", " ").title()
                else:
                    term = m.group(1).strip()
                    fact = m.group(2).strip()
                
                if len(fact) > 10 and len(fact) < 200:
                    candidates.append({
                        "term": term,
                        "fact": fact,
                        "source": str(rel_path)
                    })
    
    return candidates

def main():
    print("🔍 正在扫描 Wiki 以挖掘新事实...")
    candidates = discover()
    
    print(f"✅ 发现 {len(candidates)} 条潜在事实候选：\n")
    for i, c in enumerate(candidates, 1):
        print(f"{i}. [{c['term']}] {c['fact']}")
        print(f"   来源: {c['source']}\n")
    
    print("-" * 60)
    print(f"建议：挑选上述内容加入 scripts/lint_wiki.py 的 CANONICAL_FACTS 中。")

if __name__ == "__main__":
    main()
