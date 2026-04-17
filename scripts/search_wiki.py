#!/usr/bin/env python3
"""
search_wiki.py — wiki 内容搜索工具
支持 BM25 / 关联页面 / 本地混合语义搜索。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from search_indexing import (
    REPO_ROOT,
    WIKI_DIR,
    hash_embed_text,
    iter_wiki_documents,
    parse_frontmatter,
    strip_frontmatter,
    tokenize_text,
    truncate_for_embedding,
)

CACHE_FILE = REPO_ROOT / "exports" / "search-cache.json"
CACHE_MAX = 30
VECTOR_INDEX_FILE = REPO_ROOT / "exports" / "vector-index.npz"
VECTOR_META_FILE = REPO_ROOT / "exports" / "vector-index-meta.json"


def extract_related_links(content: str, source_path: Path) -> list[str]:
    related = []
    in_related = False
    for line in content.splitlines():
        if line.startswith("##") and any(key in line.lower() for key in ["关联", "related", "相关"]):
            in_related = True
            continue
        if in_related:
            if line.startswith("##"):
                break
            for part in __import__("re").finditer(r"\[([^\]]+)\]\(([^)]+)\)", line):
                title, href = part.group(1), part.group(2)
                if not href.startswith("http") and href.endswith(".md"):
                    resolved = (source_path.parent / href).resolve()
                    rel = resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else href
                    related.append(f"{title}  ({rel})")
    return related


def compute_avgdl(docs: list[dict]) -> float:
    lengths = [max(sum(doc["token_counts"].values()), 1) for doc in docs]
    return sum(lengths) / max(len(lengths), 1)


def compute_score(
    token_counts: dict[str, int],
    query_tokens: list[str],
    title: str = "",
    avgdl: float = 0.0,
    k1: float = 1.5,
    b: float = 0.75,
    fm: dict | None = None,
    page_type: str = "",
) -> float:
    if not query_tokens:
        return 0.0
    score = 0.0
    fm = fm or {}
    dl = max(sum(token_counts.values()), 1)
    avgdl = avgdl or dl
    summary = str(fm.get("summary", fm.get("description", ""))).lower()
    title_l = (title or "").lower()

    for token in query_tokens:
        tf = token_counts.get(token, 0)
        if tf == 0:
            continue
        idf = 0.693
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avgdl)
        term_score = idf * numerator / denominator
        if token in title_l:
            term_score *= 5.0
        elif summary and token in summary:
            term_score *= 2.0
        score += term_score

    updated_str = fm.get("updated", fm.get("created", ""))
    if updated_str:
        try:
            from datetime import date as _date

            upd = _date.fromisoformat(str(updated_str)[:10])
            if (_date.today() - upd).days <= 30:
                score *= 1.2
        except (ValueError, TypeError):
            pass

    if page_type == "query":
        score *= 0.7
    return score


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score <= 0 and min_score <= 0:
        return [0.0 for _ in scores]
    if abs(max_score - min_score) < 1e-9:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def load_vector_resources() -> tuple[np.ndarray, dict] | tuple[None, None]:
    if not VECTOR_INDEX_FILE.exists() or not VECTOR_META_FILE.exists():
        return None, None
    matrix = np.load(VECTOR_INDEX_FILE)["embeddings"]
    meta = json.loads(VECTOR_META_FILE.read_text(encoding="utf-8"))
    return matrix, meta


def encode_query_vector(query: str, meta: dict) -> np.ndarray | None:
    embedding_meta = (meta or {}).get("embedding", {})
    backend = embedding_meta.get("backend")
    if backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_meta.get("model"))
            vec = model.encode([truncate_for_embedding(query)], normalize_embeddings=True, show_progress_bar=False)[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None
    if backend == "hashed-token-projection":
        return hash_embed_text(query, dim=int(embedding_meta.get("dim", 256)))
    return None


def search(
    query_words: list[str],
    type_filter: str | None,
    tag_filters: list[str],
    context_lines: int,
    case_sensitive: bool,
    show_related: bool = False,
    semantic: bool = False,
) -> tuple[list[dict], str | None]:
    del case_sensitive  # tokenizer is normalized; keep CLI flag for output highlighting compatibility
    docs = iter_wiki_documents()
    avgdl = compute_avgdl(
        [
            {
                "token_counts": __import__("collections").Counter(tokenize_text(doc["body"]))
            }
            for doc in docs
        ]
    )

    vector_matrix = None
    vector_meta = None
    semantic_notice = None
    query_text = " ".join(query_words).strip()
    query_tokens = tokenize_text(query_text)
    query_set = set(query_tokens)

    if semantic:
        vector_matrix, vector_meta = load_vector_resources()
        if vector_matrix is None or vector_meta is None:
            semantic_notice = "语义索引不存在，已回退到纯 BM25。请先运行：make vectors"
            semantic = False

    prepared = []
    for doc in docs:
        fm = doc["frontmatter"]
        if type_filter and str(doc["page_type"]).lower() != type_filter.lower():
            continue
        page_tags = [str(tag).lower() for tag in doc.get("tags", [])]
        if tag_filters and not all(tag.lower() in page_tags for tag in tag_filters):
            continue

        raw = (REPO_ROOT / doc["path"]).read_text(encoding="utf-8")
        body = strip_frontmatter(raw)
        token_counts = __import__("collections").Counter(tokenize_text(body))
        query_match = True
        if query_set and not semantic:
            query_match = all(token in token_counts for token in query_set)
        if not query_match:
            continue

        score = compute_score(
            token_counts=token_counts,
            query_tokens=query_tokens,
            title=doc["title"],
            avgdl=avgdl,
            fm=fm,
            page_type=doc["page_type"],
        )

        lines = body.splitlines()
        matched_lines = []
        if query_words:
            lowered_words = [word.lower() for word in query_words if word.strip()]
            for i, line in enumerate(lines):
                if lowered_words and any(word in line.lower() for word in lowered_words):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    matched_lines.append((i + 1, lines[start:end], i - start))
        if not matched_lines:
            summary_line = doc["summary"] or (lines[0] if lines else "")
            matched_lines = [(1, [summary_line], 0)]

        prepared.append(
            {
                "path": Path(doc["path"]),
                "fm": fm,
                "matches": matched_lines[:5],
                "related": extract_related_links(raw, REPO_ROOT / doc["path"]) if show_related else [],
                "score": score,
                "bm25_score": score,
                "title": doc["title"],
                "id": doc["id"],
                "summary": doc["summary"],
                "page_type": doc["page_type"],
                "vector_score": 0.0,
                "hybrid_score": score,
            }
        )

    if semantic and prepared:
        doc_order = {str(item["path"]): idx for idx, item in enumerate((vector_meta or {}).get("documents", []))}
        query_vector = encode_query_vector(query_text, vector_meta)
        if query_vector is None:
            semantic_notice = "向量编码器不可用，已回退到纯 BM25。请安装 sentence-transformers 或重建索引。"
            semantic = False
        else:
            bm25_norm = normalize_scores([item["bm25_score"] for item in prepared])
            vector_scores = []
            for item in prepared:
                doc_idx = doc_order.get(item["path"].as_posix())
                cosine = float(np.dot(vector_matrix[doc_idx], query_vector)) if doc_idx is not None else 0.0
                item["vector_score"] = cosine
                vector_scores.append(cosine)
            vector_norm = normalize_scores(vector_scores)
            for item, bm25_n, vector_n in zip(prepared, bm25_norm, vector_norm):
                item["hybrid_score"] = 0.6 * bm25_n + 0.4 * vector_n
                item["score"] = item["hybrid_score"]
            prepared.sort(key=lambda r: r["hybrid_score"], reverse=True)
    else:
        prepared.sort(key=lambda r: r["score"], reverse=True)

    return prepared, semantic_notice


def _cache_key(query_words: list[str], type_filter: str | None, tag_filters: list[str], semantic: bool) -> str:
    return json.dumps(
        {"q": sorted(query_words), "t": type_filter or "", "tags": sorted(tag_filters), "semantic": semantic},
        sort_keys=True,
        ensure_ascii=False,
    )


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"entries": []}


def save_cache(cache: dict, key: str, results: list[dict]) -> None:
    serializable = []
    for r in results[:10]:
        serializable.append(
            {
                "path": str(r["path"]),
                "score": round(r.get("score", 0.0), 6),
                "title": r.get("title", ""),
                "type": r["fm"].get("type", r.get("page_type", "")),
                "tags": r["fm"].get("tags", []),
                "vector_score": round(r.get("vector_score", 0.0), 6),
                "hybrid_score": round(r.get("hybrid_score", r.get("score", 0.0)), 6),
            }
        )
    entry = {"key": key, "ts": datetime.now().isoformat()[:19], "results": serializable}
    entries = [e for e in cache.get("entries", []) if e.get("key") != key]
    entries.insert(0, entry)
    cache["entries"] = entries[:CACHE_MAX]
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def highlight(text: str, words: list[str], case_sensitive: bool) -> str:
    import re

    flags = 0 if case_sensitive else re.IGNORECASE
    for w in words:
        text = re.sub(f"({re.escape(w)})", r"\033[1;33m\1\033[0m", text, flags=flags)
    return text


def print_results(results: list[dict], query_words: list[str], case_sensitive: bool, semantic_notice: str | None = None) -> None:
    if semantic_notice:
        print(f"\033[2m[提示] {semantic_notice}\033[0m")
    if not results:
        print("未找到匹配结果。")
        return

    for r in results:
        fm = r["fm"]
        type_str = f"[{fm.get('type', r.get('page_type', '?'))}]"
        tags_str = ", ".join(fm.get("tags", [])) or "-"
        status_str = fm.get("status", "?")
        extra_score = ""
        if r.get("vector_score"):
            extra_score = f"  \033[2m[bm25={r.get('bm25_score', 0.0):.4f} vec={r.get('vector_score', 0.0):.4f} hybrid={r.get('hybrid_score', 0.0):.4f}]\033[0m"
        else:
            extra_score = f"  \033[2m[{r.get('score', 0.0):.4f}]\033[0m" if r.get("score", 0) > 0 else ""
        print(f"\n\033[1;36m{r['path']}\033[0m{extra_score}  {type_str}  status={status_str}")
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
    parser.add_argument("--type", dest="type_filter", help="按页面类型过滤（concept/method/task/comparison/...）")
    parser.add_argument("--tag", dest="tag_filters", action="append", default=[], help="按标签过滤（可多次指定，AND 逻辑）")
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
        payload = {"notice": semantic_notice, "results": out} if semantic_notice else out
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_results(results, args.query, args.case, semantic_notice=semantic_notice)


if __name__ == "__main__":
    main()
