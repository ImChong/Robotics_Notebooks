#!/usr/bin/env python3
"""
search_wiki.py — wiki 内容搜索工具
基于 Karpathy LLM Wiki 模式的 CLI 搜索工具建议实现。

功能：
  - 全文关键词搜索（支持多词 AND 逻辑）
  - 按页面类型过滤（concept / method / task / comparison 等）
  - 按标签过滤（从 YAML frontmatter 读取）
  - 显示匹配行上下文
  - 可选输出页面「关联页面」列表（--related）

用法：
  python3 scripts/search_wiki.py "MPC locomotion"
  python3 scripts/search_wiki.py "diffusion" --type method
  python3 scripts/search_wiki.py --tag rl --tag humanoid
  python3 scripts/search_wiki.py "state estimation" --context 3
  python3 scripts/search_wiki.py "sim2real" --related
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"

# ── YAML frontmatter 解析 ─────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> dict:
    """从页面内容中提取 YAML frontmatter（仅支持简单键值和列表）。"""
    fm = {}
    if not content.startswith("---"):
        return fm
    end = content.find("\n---", 3)
    if end == -1:
        return fm
    block = content[3:end].strip()
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # 列表形式：tags: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip('"\'') for v in val[1:-1].split(",")]
                fm[key] = items
            else:
                fm[key] = val.strip('"\'')
    return fm

def extract_related_pages(body: str) -> list[str]:
    """从 markdown 正文中提取「## 关联页面」区块下的链接文本。"""
    m = re.search(r"^##\s+关联页面\s*$", body, re.MULTILINE)
    if not m:
        return []

    start = m.end()
    remaining = body[start:]
    next_heading = re.search(r"^##\s+", remaining, re.MULTILINE)
    section = remaining[: next_heading.start()] if next_heading else remaining

    # 提取 markdown 链接 [title](url)
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", section)
    if not links:
        return []
    return [f"{title} -> {url}" for title, url in links]

def strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end == -1:
        return content
    return content[end + 4:].lstrip()

# ── 搜索逻辑 ──────────────────────────────────────────────────────────────────

def search(
    query_words: list[str],
    type_filter: str | None,
    tag_filters: list[str],
    context_lines: int,
    case_sensitive: bool,
    show_related: bool,
) -> list[dict]:
    results = []
    pages = sorted(WIKI_DIR.rglob("*.md"))

    for page in pages:
        raw = page.read_text(encoding="utf-8")
        fm = parse_frontmatter(raw)
        body = strip_frontmatter(raw)

        # 类型过滤
        if type_filter and fm.get("type", "").lower() != type_filter.lower():
            continue

        # 标签过滤（AND 逻辑）
        if tag_filters:
            page_tags = [t.lower() for t in fm.get("tags", [])]
            if not all(tf.lower() in page_tags for tf in tag_filters):
                continue

        # 关键词匹配（AND 逻辑，搜索 body）
        flags = 0 if case_sensitive else re.IGNORECASE
        if query_words:
            if not all(re.search(re.escape(w), body, flags) for w in query_words):
                continue

        # 提取匹配行和上下文
        lines = body.splitlines()
        matched_lines = []
        if query_words:
            pattern = "|".join(re.escape(w) for w in query_words)
            for i, line in enumerate(lines):
                if re.search(pattern, line, flags):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    matched_lines.append((i + 1, lines[start:end], i - start))
        else:
            # 无关键词时只显示页面标题行
            matched_lines = [(1, [lines[0]] if lines else [""], 0)]

        results.append({
            "path": page.relative_to(REPO_ROOT),
            "fm": fm,
            "matches": matched_lines[:5],  # 最多显示 5 处匹配
            "related": extract_related_pages(body) if show_related else [],
        })

    return results

def highlight(text: str, words: list[str], case_sensitive: bool) -> str:
    """在终端用 ANSI 粗体标亮匹配词。"""
    flags = 0 if case_sensitive else re.IGNORECASE
    for w in words:
        text = re.sub(f"({re.escape(w)})", r"\033[1;33m\1\033[0m", text, flags=flags)
    return text

def print_results(results: list[dict], query_words: list[str], case_sensitive: bool, show_related: bool):
    if not results:
        print("未找到匹配结果。")
        return

    for r in results:
        fm = r["fm"]
        type_str = f"[{fm.get('type', '?')}]"
        tags_str = ", ".join(fm.get("tags", [])) or "-"
        status_str = fm.get("status", "?")

        print(f"\n\033[1;36m{r['path']}\033[0m  {type_str}  status={status_str}")
        print(f"  tags: {tags_str}")

        for lineno, ctx_lines, match_offset in r["matches"]:
            for j, line in enumerate(ctx_lines):
                prefix = f"  {lineno - match_offset + j:>4} │ "
                if j == match_offset:
                    print(prefix + highlight(line, query_words, case_sensitive))
                else:
                    print(f"\033[2m{prefix}{line}\033[0m")

        if show_related:
            related = r.get("related", [])
            if related:
                print("  related:")
                for rel in related:
                    print(f"    - {rel}")
            else:
                print("  related: -")

    print(f"\n共找到 {len(results)} 个页面。")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="搜索 Robotics_Notebooks wiki 内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python3 scripts/search_wiki.py "MPC locomotion"
  python3 scripts/search_wiki.py "diffusion" --type method
  python3 scripts/search_wiki.py --tag rl --tag humanoid
  python3 scripts/search_wiki.py "bellman" --context 5
  python3 scripts/search_wiki.py "sim2real" --related
        """,
    )
    parser.add_argument("query", nargs="*", help="搜索关键词（多词为 AND 逻辑）")
    parser.add_argument("--type", dest="type_filter", help="按页面类型过滤（concept/method/task/comparison/...）")
    parser.add_argument("--tag", dest="tag_filters", action="append", default=[], help="按标签过滤（可多次指定，AND 逻辑）")
    parser.add_argument("--context", type=int, default=1, help="每处匹配显示的上下文行数（默认 1）")
    parser.add_argument("--case", action="store_true", help="区分大小写")
    parser.add_argument("--related", action="store_true", help="输出匹配页面的关联页面列表")
    args = parser.parse_args()

    if not args.query and not args.type_filter and not args.tag_filters:
        parser.print_help()
        sys.exit(0)

    results = search(
        query_words=args.query,
        type_filter=args.type_filter,
        tag_filters=args.tag_filters,
        context_lines=args.context,
        case_sensitive=args.case,
        show_related=args.related,
    )
    print_results(results, args.query, args.case, args.related)

if __name__ == "__main__":
    main()
