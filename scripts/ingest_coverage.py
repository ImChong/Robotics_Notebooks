#!/usr/bin/env python3
"""
ingest_coverage.py — Ingest 覆盖审计工具

给定 sources/papers/*.md，找出所有提及相关关键词但未被覆盖的 wiki 页面，
帮助实现 Karpathy 目标：每次 ingest 覆盖 8-12 个 wiki 页面。

用法：
  python3 scripts/ingest_coverage.py sources/papers/mpc.md
  make coverage F=sources/papers/mpc.md
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"

STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "at",
    "to",
    "by",
    "with",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "from",
    "that",
    "this",
    "it",
    "as",
    "can",
    "will",
    "more",
    "not",
    "no",
    "than",
    "but",
    "also",
    "based",
    "using",
    "used",
    "paper",
    "work",
    "control",
    "robot",
    "learning",
    "method",
    "system",
    "approach",
}

VENUE_ABBR = {
    "RSS",
    "NeurIPS",
    "CoRL",
    "ICRA",
    "RAL",
    "IEEE",
    "IROS",
    "ICLR",
    "ICML",
    "IJRR",
    "TRO",
    "DOI",
    "URL",
    "API",
    "CLI",
    "LLM",
    "MIT",
    "ETH",
    "GPU",
    "CPU",
    "DDP",
    "ODE",
    "FEM",
    "SPH",
    "ACM",
    "PDF",
    "TBD",
    "MVP",
}


def extract_covered_wiki_paths(content: str, sources_path: Path) -> set[Path]:
    """从 sources 文件中提取已覆盖的 wiki 页面路径（解析 wiki 相对链接）。"""
    covered = set()
    for m in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", content):
        href = m.group(1).split("#")[0]
        resolved = (sources_path.parent / href).resolve()
        if resolved.exists():
            covered.add(resolved)
    return covered


def extract_key_terms(content: str) -> list[str]:
    """从 sources 文件提取技术关键词用于 wiki 扫描。"""
    terms: set[str] = set()

    # 1. 论文标题中的英文词（大写开头，≥4 字符）
    for m in re.finditer(r"^###\s+\d+\)\s+(.+?)(?:（|\(|$)", content, re.MULTILINE):
        for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", m.group(1)):
            if w.lower() not in STOP_WORDS and w not in VENUE_ABBR:
                terms.add(w)

    # 2. wiki 路径中的主题词（kebab-case → 单词）
    for m in re.finditer(r"wiki/\w+/([\w-]+)\.md", content):
        for part in m.group(1).split("-"):
            if len(part) >= 4 and part.lower() not in STOP_WORDS:
                terms.add(part)

    # 3. 全大写技术缩写（2-5 字符，排除场馆/通用词）
    for m in re.finditer(r"\b([A-Z]{2,5})\b", content):
        abbr = m.group(1)
        if abbr not in VENUE_ABBR:
            terms.add(abbr)

    return sorted(terms)


def scan_wiki_for_terms(terms: list[str], covered: set[Path]) -> list[dict]:
    """扫描 wiki 页面，找出提及关键词但未被 sources 覆盖的页面。"""
    suggestions = []
    flags = re.IGNORECASE

    for page in sorted(WIKI_DIR.rglob("*.md")):
        if page in covered or page.name == "README.md":
            continue

        content = page.read_text(encoding="utf-8")
        matched = []
        for term in terms:
            # 英文词用词边界，中文缩写直接搜索
            pat = rf"\b{re.escape(term)}\b" if re.match(r"^[A-Za-z]+$", term) else re.escape(term)
            if re.search(pat, content, flags):
                matched.append(term)

        if matched:
            suggestions.append(
                {
                    "path": page.relative_to(REPO_ROOT),
                    "matched": matched[:6],
                }
            )

    return suggestions


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python3 scripts/ingest_coverage.py <sources/papers/xxx.md>")
        sys.exit(1)

    sources_path = REPO_ROOT / sys.argv[1]
    if not sources_path.exists():
        print(f"❌ 文件不存在: {sources_path}")
        sys.exit(1)

    content = sources_path.read_text(encoding="utf-8")
    covered = extract_covered_wiki_paths(content, sources_path)
    terms = extract_key_terms(content)

    print(f"\n📂 {sources_path.relative_to(REPO_ROOT)}")
    print(f"   当前已覆盖 wiki 页面：{len(covered)} 个")
    print(f"   提取关键词：{len(terms)} 个  示例：{', '.join(terms[:8])}")

    if covered:
        print(f"\n✅ 已覆盖（{len(covered)} 个）：")
        for p in sorted(covered):
            print(f"   - {p.relative_to(REPO_ROOT)}")

    suggestions = scan_wiki_for_terms(terms, covered)
    if suggestions:
        print(f"\n💡 建议补充覆盖（{len(suggestions)} 个）：")
        for s in suggestions:
            print(f"   + {s['path']}  [{', '.join(s['matched'])}]")
    else:
        print("\n✅ 未发现可补充的 wiki 页面")

    print(
        f"\n合计：当前 {len(covered)} 页 → 建议增补 {len(suggestions)} 页 "
        f"→ 潜在覆盖 {len(covered) + len(suggestions)} 页"
    )


if __name__ == "__main__":
    main()
