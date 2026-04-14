#!/usr/bin/env python3
"""
search_wiki.py — wiki 内容搜索工具
基于 Karpathy LLM Wiki 模式的 CLI 搜索工具建议实现。

功能：
  - 全文关键词搜索（支持多词 AND 逻辑）
  - 按页面类型过滤（concept / method / task / comparison 等）
  - 按标签过滤（从 YAML frontmatter 读取）
  - 显示匹配行上下文

用法：
  python3 scripts/search_wiki.py "MPC locomotion"
  python3 scripts/search_wiki.py "diffusion" --type method
  python3 scripts/search_wiki.py --tag rl --tag humanoid
  python3 scripts/search_wiki.py "state estimation" --context 3
"""

import argparse
import json
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

def strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end == -1:
        return content
    return content[end + 4:].lstrip()

# ── 搜索逻辑 ──────────────────────────────────────────────────────────────────

def extract_related_links(content: str, source_path: Path) -> list[str]:
    """从页面的关联页面区块中提取链接标题和路径。"""
    related = []
    # 找关联页面区块
    in_related = False
    for line in content.splitlines():
        if re.match(r'^##\s+.*(关联|related|相关)', line, re.IGNORECASE):
            in_related = True
            continue
        if in_related:
            if line.startswith("##"):
                break
            # 提取 [title](path) 格式的链接
            for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
                title, href = m.group(1), m.group(2)
                if not href.startswith("http") and href.endswith(".md"):
                    resolved = (source_path.parent / href).resolve()
                    related.append(f"{title}  ({resolved.relative_to(REPO_ROOT) if resolved.is_relative_to(REPO_ROOT) else href})")
    return related


def search(
    query_words: list[str],
    type_filter: str | None,
    tag_filters: list[str],
    context_lines: int,
    case_sensitive: bool,
    show_related: bool = False,
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

        title_m = re.search(r'^#\s+(.+)', body, re.MULTILINE)
        page_title = title_m.group(1) if title_m else ""
        score = compute_score(body, query_words, page_title, case_sensitive)

        related_links = extract_related_links(raw, page) if show_related else []
        results.append({
            "path": page.relative_to(REPO_ROOT),
            "fm": fm,
            "matches": matched_lines[:5],  # 最多显示 5 处匹配
            "related": related_links,
            "score": score,
            "title": page_title,
        })

    if query_words:
        results.sort(key=lambda r: r["score"], reverse=True)
    return results

def compute_score(body: str, query_words: list[str], title: str = "",
                  case_sensitive: bool = False) -> float:
    """TF × coverage × title_boost 相关度评分（无外部依赖）。"""
    if not query_words:
        return 0.0
    flags = 0 if case_sensitive else re.IGNORECASE
    n = max(len(body.split()), 1)
    scores = []
    for w in query_words:
        count = len(re.findall(re.escape(w), body, flags))
        tf = count / n
        boost = 3.0 if re.search(re.escape(w), title, flags) else 1.0
        scores.append(tf * boost)
    coverage = sum(1 for s in scores if s > 0) / len(query_words)
    return coverage * sum(scores)


def highlight(text: str, words: list[str], case_sensitive: bool) -> str:
    """在终端用 ANSI 粗体标亮匹配词。"""
    flags = 0 if case_sensitive else re.IGNORECASE
    for w in words:
        text = re.sub(f"({re.escape(w)})", r"\033[1;33m\1\033[0m", text, flags=flags)
    return text

def print_results(results: list[dict], query_words: list[str], case_sensitive: bool):
    if not results:
        print("未找到匹配结果。")
        return

    for r in results:
        fm = r["fm"]
        type_str = f"[{fm.get('type', '?')}]"
        tags_str = ", ".join(fm.get("tags", [])) or "-"
        status_str = fm.get("status", "?")

        score_str = f"  \033[2m[{r['score']:.4f}]\033[0m" if r.get("score", 0) > 0 else ""
        print(f"\n\033[1;36m{r['path']}\033[0m{score_str}  {type_str}  status={status_str}")
        print(f"  tags: {tags_str}")

        for lineno, ctx_lines, match_offset in r["matches"]:
            for j, line in enumerate(ctx_lines):
                prefix = f"  {lineno - match_offset + j:>4} │ "
                if j == match_offset:
                    print(prefix + highlight(line, query_words, case_sensitive))
                else:
                    print(f"\033[2m{prefix}{line}\033[0m")

        if r.get("related"):
            print("  \033[2m关联页面：\033[0m")
            for rel in r["related"]:
                print(f"  \033[2m  → {rel}\033[0m")

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
        """,
    )
    parser.add_argument("query", nargs="*", help="搜索关键词（多词为 AND 逻辑）")
    parser.add_argument("--type", dest="type_filter", help="按页面类型过滤（concept/method/task/comparison/...）")
    parser.add_argument("--tag", dest="tag_filters", action="append", default=[], help="按标签过滤（可多次指定，AND 逻辑）")
    parser.add_argument("--context", type=int, default=1, help="每处匹配显示的上下文行数（默认 1）")
    parser.add_argument("--case", action="store_true", help="区分大小写")
    parser.add_argument("--related", action="store_true", help="同时输出每个匹配页面的关联页面（用于快速找邻居）")
    parser.add_argument("--json", dest="json_out", action="store_true", help="JSON 格式输出（便于 LLM 处理）")
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

    if args.json_out:
        out = []
        for r in results:
            snippet = ""
            if r["matches"]:
                _, ctx_lines, offset = r["matches"][0]
                snippet = ctx_lines[offset] if offset < len(ctx_lines) else ""
            out.append({
                "path": str(r["path"]),
                "score": round(r.get("score", 0.0), 6),
                "type": r["fm"].get("type", ""),
                "tags": r["fm"].get("tags", []),
                "title": r.get("title", ""),
                "snippet": snippet.strip()[:200],
            })
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_results(results, args.query, args.case)

if __name__ == "__main__":
    main()
