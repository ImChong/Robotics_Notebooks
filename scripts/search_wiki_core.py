"""
search_wiki_core — BM25/向量混合搜索、缓存与结果展示（由 search_wiki.py CLI 调用）。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from search_indexing import (
    REPO_ROOT,
    hash_embed_text,
    iter_wiki_documents,
    tokenize_text,
    truncate_for_embedding,
)

CACHE_FILE = REPO_ROOT / "exports" / "search-cache.json"
CACHE_MAX = 30
VECTOR_INDEX_FILE = REPO_ROOT / "exports" / "vector-index.npz"
VECTOR_META_FILE = REPO_ROOT / "exports" / "vector-index-meta.json"

# 缩写/别名归一化表：检索时与全称双向展开，命中后在 print_results 中给出"已展开为 …"提示。
WIKI_ABBREVIATIONS: dict[str, list[str]] = {
    "wbc": ["whole-body control"],
    "vla": ["vision-language-action"],
    "il": ["imitation learning"],
    "rl": ["reinforcement learning"],
    "mpc": ["model predictive control"],
    "ppo": ["proximal policy optimization"],
    "sac": ["soft actor-critic"],
    "hqp": ["hierarchical quadratic program"],
    "cbf": ["control barrier function"],
    "clf": ["control lyapunov function"],
    "bc": ["behavior cloning"],
    "ik": ["inverse kinematics"],
    "fk": ["forward kinematics"],
    "lip": ["linear inverted pendulum"],
    "zmp": ["zero moment point"],
    "tsid": ["task-space inverse dynamics"],
}


def _build_alias_indexes() -> tuple[dict[str, list[str]], dict[str, str]]:
    forward = {k.lower(): list(v) for k, v in WIKI_ABBREVIATIONS.items()}
    reverse: dict[str, str] = {}
    for abbrev, fulls in forward.items():
        for full in fulls:
            reverse[full.lower()] = abbrev
    return forward, reverse


_ALIAS_FORWARD, _ALIAS_REVERSE = _build_alias_indexes()


def expand_query_aliases(
    query_words: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """检索时双向展开缩写与全称。

    返回 ``(expanded_words, expansions)``：``expanded_words`` 在原始 query 之上追加
    规范化别名（缩写→全称、全称短语→缩写），``expansions`` 记录 ``(原输入, 展开形式)``
    供 CLI 在结果上方提示"已展开为 …"。
    """
    if not query_words:
        return [], []

    expansions: list[tuple[str, str]] = []
    expanded: list[str] = list(query_words)
    seen = {w.lower() for w in expanded}

    joined = " ".join(query_words).strip().lower()
    if joined and joined in _ALIAS_REVERSE:
        abbrev = _ALIAS_REVERSE[joined]
        if abbrev not in seen:
            expanded.append(abbrev.upper())
            seen.add(abbrev)
            expansions.append((" ".join(query_words), abbrev.upper()))

    for word in query_words:
        key = word.lower()
        if key not in _ALIAS_FORWARD:
            continue
        for full in _ALIAS_FORWARD[key]:
            if full.lower() in seen:
                continue
            expanded.append(full)
            seen.add(full.lower())
            expansions.append((word, full))

    return expanded, expansions


def extract_related_links(content: str, source_path: Path) -> list[str]:
    related: list[str] = []
    in_related = False
    for line in content.splitlines():
        if line.startswith("##") and any(
            key in line.lower() for key in ["关联", "related", "相关"]
        ):
            in_related = True
            continue
        if in_related:
            if line.startswith("##"):
                break
            for part in __import__("re").finditer(r"\[([^\]]+)\]\(([^)]+)\)", line):
                title, href = part.group(1), part.group(2)
                if not href.startswith("http") and href.endswith(".md"):
                    resolved = (source_path.parent / href).resolve()
                    rel = (
                        resolved.relative_to(REPO_ROOT).as_posix()
                        if resolved.is_relative_to(REPO_ROOT)
                        else href
                    )
                    related.append(f"{title}  ({rel})")
    return related


def compute_avgdl(docs: list[dict]) -> float:
    lengths = [doc.get("dl", max(sum(doc.get("token_counts", {}).values()), 1)) for doc in docs]
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
    dl: int = 1,
) -> float:
    if not query_tokens:
        return 0.0
    score = 0.0
    fm = fm or {}
    avgdl = avgdl or dl
    summary = str(fm.get("summary", fm.get("description", ""))).lower()
    title_l = (title or "").lower()

    # Pre-compute document-level constants outside the loop
    len_norm = 1 - b + b * dl / avgdl
    k1_plus_1 = k1 + 1

    for token in query_tokens:
        tf = token_counts.get(token, 0)
        if tf == 0:
            continue
        idf = 0.693
        numerator = tf * k1_plus_1
        denominator = tf + k1 * len_norm
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

    # V16: 提权 comparison 类型页面，鼓励用户查看对比总结
    if page_type == "comparison":
        score *= 1.3

    return score


def levenshtein_distance(a: str, b: str) -> int:
    """计算两个字符串的 Levenshtein 编辑距离（迭代版本，O(len(a)*len(b))）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def collect_known_terms(docs: list[dict]) -> dict[str, str]:
    """收集 wiki 中所有 title 与 tag，返回 {小写形式: 原始展示形式}。"""
    terms: dict[str, str] = {}
    for doc in docs:
        title = (doc.get("title") or "").strip()
        if title:
            terms.setdefault(title.lower(), title)
        for tag in doc.get("tags", []) or []:
            tag_str = str(tag).strip()
            if tag_str:
                terms.setdefault(tag_str.lower(), tag_str)
    return terms


def suggest_terms(query: str, terms: dict[str, str], top_k: int = 5) -> list[tuple[str, int]]:
    """给定查询，返回距离最近的 top_k 个 (display_term, distance)。

    阈值：距离不超过 max(2, ceil(len(query)/2))，避免无意义的远距离推荐。
    """
    query_norm = query.strip().lower()
    if not query_norm:
        return []
    max_dist = max(2, (len(query_norm) + 1) // 2)
    candidates: list[tuple[str, int]] = []
    for key, display in terms.items():
        d = levenshtein_distance(query_norm, key)
        if d == 0 or d > max_dist:
            continue
        candidates.append((display, d))
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[:top_k]


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


def load_vector_resources() -> tuple[Any, dict[str, Any]] | tuple[None, None]:
    if not VECTOR_INDEX_FILE.exists() or not VECTOR_META_FILE.exists():
        return None, None
    import numpy as np

    matrix = np.load(VECTOR_INDEX_FILE)["embeddings"]
    meta = json.loads(VECTOR_META_FILE.read_text(encoding="utf-8"))
    return matrix, meta


def encode_query_vector(query: str, meta: dict[str, Any]) -> Any:
    embedding_meta = (meta or {}).get("embedding", {})
    backend = embedding_meta.get("backend")
    if backend == "sentence-transformers":
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            model = SentenceTransformer(embedding_meta.get("model"))
            vec = model.encode(
                [truncate_for_embedding(query)], normalize_embeddings=True, show_progress_bar=False
            )[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None
    if backend == "hashed-token-projection":
        return hash_embed_text(query, dim=int(embedding_meta.get("dim", 256)))
    return None


def _filter_doc(doc: dict, type_filter: str | None, tag_filters: list[str] | None) -> bool:
    if type_filter and str(doc["page_type"]).lower() != type_filter.lower():
        return False
    page_tags = [str(tag).lower() for tag in doc.get("tags", [])]
    if tag_filters and not all(tag.lower() in page_tags for tag in tag_filters):
        return False
    return True


def _find_matched_lines(
    lines: list[str], query_words: list[str], context_lines: int
) -> list[tuple[int, list[str], int]]:
    matched_lines: list[tuple[int, list[str], int]] = []
    if not query_words:
        return matched_lines
    lowered_words = [word.lower() for word in query_words if word.strip()]
    if not lowered_words:
        return matched_lines

    for i, line in enumerate(lines):
        line_lower = line.lower()
        found = False
        for word in lowered_words:
            if word in line_lower:
                found = True
                break
        if found:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            matched_lines.append((i + 1, lines[start:end], i - start))
    return matched_lines


def _apply_vector_scores(
    prepared: list[dict], query_text: str, vector_matrix: Any, vector_meta: dict[str, Any]
) -> str | None:
    doc_order = {
        str(item["path"]): idx for idx, item in enumerate((vector_meta or {}).get("documents", []))
    }
    query_vector = encode_query_vector(query_text, vector_meta)
    if query_vector is None:
        return "向量编码器不可用，已回退到纯 BM25。请安装 sentence-transformers 或重建索引。"

    bm25_norm = normalize_scores([item["bm25_score"] for item in prepared])
    import numpy as np

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
    del (
        case_sensitive
    )  # tokenizer is normalized; keep CLI flag for output highlighting compatibility
    docs = iter_wiki_documents()

    # ⚡ Bolt Optimization: Cache token counts to avoid redundant tokenization
    # Tokenizing the body text is expensive. Doing it twice (once for avgdl, once for scoring)
    # doubles the search latency. We cache it here on the doc objects.
    for doc in docs:
        counts = Counter(tokenize_text(doc["body"]))
        doc["token_counts"] = counts
        doc["dl"] = max(sum(counts.values()), 1)

    avgdl = compute_avgdl(docs)

    vector_matrix = None
    vector_meta = None
    semantic_notice = None

    effective_query_words, alias_expansions = expand_query_aliases(query_words)
    if alias_expansions:
        alias_notice = "缩写归一化：已展开为 " + "；".join(
            f"'{src}' → '{dst}'" for src, dst in alias_expansions
        )
        semantic_notice = alias_notice

    query_text = " ".join(effective_query_words).strip()
    query_tokens = tokenize_text(query_text)

    if semantic:
        vector_matrix, vector_meta = load_vector_resources()
        if vector_matrix is None or vector_meta is None:
            fallback = "语义索引不存在，已回退到纯 BM25。请先运行：make vectors"
            semantic_notice = f"{semantic_notice}；{fallback}" if semantic_notice else fallback
            semantic = False

    prepared = []
    for doc in docs:
        if not _filter_doc(doc, type_filter, tag_filters):
            continue

        body = doc["body"]
        token_counts = doc["token_counts"]
        fm = doc["frontmatter"]

        score = compute_score(
            token_counts=token_counts,
            query_tokens=query_tokens,
            title=doc["title"],
            avgdl=avgdl,
            fm=fm,
            page_type=doc["page_type"],
            dl=doc.get("dl", 1),
        )
        if query_tokens and not semantic and score <= 0:
            continue

        lines = body.splitlines()
        matched_lines = _find_matched_lines(lines, effective_query_words, context_lines)
        if not matched_lines:
            summary_line = doc["summary"] or (lines[0] if lines else "")
            matched_lines = [(1, [summary_line], 0)]

        prepared.append(
            {
                "path": Path(doc["path"]),
                "fm": fm,
                "matches": matched_lines[:5],
                "related": extract_related_links(body, REPO_ROOT / doc["path"])
                if show_related
                else [],
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
        assert vector_matrix is not None and vector_meta is not None
        new_notice = _apply_vector_scores(prepared, query_text, vector_matrix, vector_meta)
        if new_notice:
            semantic_notice = f"{semantic_notice}；{new_notice}" if semantic_notice else new_notice
            prepared.sort(key=lambda r: r["score"], reverse=True)
    else:
        prepared.sort(key=lambda r: r["score"], reverse=True)

    return prepared, semantic_notice


def _cache_key(
    query_words: list[str], type_filter: str | None, tag_filters: list[str], semantic: bool
) -> str:
    return json.dumps(
        {
            "q": sorted(query_words),
            "t": type_filter or "",
            "tags": sorted(tag_filters),
            "semantic": semantic,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def load_cache() -> dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return cast(dict[str, Any], json.loads(CACHE_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return {"entries": []}


def save_cache(cache: dict[str, Any], key: str, results: list[dict]) -> None:
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


def print_results(
    results: list[dict],
    query_words: list[str],
    case_sensitive: bool,
    semantic_notice: str | None = None,
    suggestions: list[tuple[str, int]] | None = None,
) -> None:
    if semantic_notice:
        print(f"\033[2m[提示] {semantic_notice}\033[0m")
    if not results:
        print("未找到匹配结果。")
        if suggestions:
            print("您是否想搜索：")
            for term, dist in suggestions:
                print(f"  \033[2m·\033[0m {term} \033[2m(编辑距离 {dist})\033[0m")
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
            extra_score = (
                f"  \033[2m[{r.get('score', 0.0):.4f}]\033[0m" if r.get("score", 0) > 0 else ""
            )
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
