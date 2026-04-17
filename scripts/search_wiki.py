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
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
CACHE_FILE = REPO_ROOT / "exports" / "search-cache.json"
CACHE_MAX = 30

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

    # 计算 avgdl（BM25 文档长度归一化参数）
    doc_lengths = []
    page_contents = {}
    for page in pages:
        raw = page.read_text(encoding="utf-8")
        page_contents[page] = raw
        doc_lengths.append(max(len(strip_frontmatter(raw).split()), 1))
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)

    for page in pages:
        raw = page_contents[page]
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
        score = compute_score(body, query_words, page_title, case_sensitive, avgdl=avgdl)

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
                  case_sensitive: bool = False,
                  avgdl: float = 0.0, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 相关度评分（无外部依赖）。IDF 使用简化近似，df=1。

    BM25 公式：score = Σ IDF(q) * tf(q,D)*(k1+1) / (tf(q,D) + k1*(1-b+b*|D|/avgdl))
    """
    if not query_words:
        return 0.0
    flags = 0 if case_sensitive else re.IGNORECASE
    tokens = len(body.split())
    dl = max(tokens, 1)
    if avgdl <= 0:
        avgdl = dl  # 单文档时 avgdl = dl → 归一化为 1

    score = 0.0
    for w in query_words:
        tf = len(re.findall(re.escape(w), body, flags))
        if tf == 0:
            continue
        # 简化 IDF：log(1 + 1/1) ≈ 0.693（单调正数，不影响排序）
        idf = 0.693
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avgdl)
        term_score = idf * numerator / denominator
        # 标题命中加权
        if re.search(re.escape(w), title, flags):
            term_score *= 3.0
        score += term_score
    return score


# ── 搜索缓存（最近 30 次查询）─────────────────────────────────────────────────

def _cache_key(query_words: list[str], type_filter: str | None, tag_filters: list[str]) -> str:
    return json.dumps({"q": sorted(query_words), "t": type_filter or "", "tags": sorted(tag_filters)},
                      sort_keys=True, ensure_ascii=False)

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"entries": []}

def save_cache(cache: dict, key: str, results: list[dict]) -> None:
    serializable = []
    for r in results[:10]:  # 只缓存前 10 条
        serializable.append({
            "path": str(r["path"]),
            "score": round(r.get("score", 0.0), 6),
            "title": r.get("title", ""),
            "type": r["fm"].get("type", ""),
            "tags": r["fm"].get("tags", []),
        })
    entry = {"key": key, "ts": datetime.now().isoformat()[:19], "results": serializable}
    entries = cache.get("entries", [])
    entries = [e for e in entries if e.get("key") != key]  # 去重
    entries.insert(0, entry)
    cache["entries"] = entries[:CACHE_MAX]
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

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

    cache = load_cache()
    cache_key = _cache_key(args.query, args.type_filter, args.tag_filters)
    cached_entry = next((e for e in cache.get("entries", []) if e.get("key") == cache_key), None)
    if cached_entry and not args.json_out:
        print(f"\033[2m[缓存命中：{cached_entry['ts']}]\033[0m")

    results = search(
        query_words=args.query,
        type_filter=args.type_filter,
        tag_filters=args.tag_filters,
        context_lines=args.context,
        case_sensitive=args.case,
        show_related=args.related,
    )
    if args.query:
        save_cache(cache, cache_key, results)

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
