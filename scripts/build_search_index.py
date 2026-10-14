#!/usr/bin/env python3
"""生成站内搜索索引（docs/search-index.json 全文 + docs/search-index-lite.json 轻量）。

两层索引（网站优化计划阶段 B 第 9 条）：
- 轻量索引只含标题 / 摘要 / 标签 / 别名等元数据（约 0.9 MB gzip），首页搜索框聚焦即加载，
  标题 / 标签 / 别名 / 摘要命中立刻可搜；
- 全文索引含每页正文 token（约 26 MB gzip），只在用户点「搜索全文」时加载。
  全文索引写 df（文档频次，整数）而不是 idf 浮点数，前端按 BM25 公式现算，文件小约 1/4。
命令行 search_wiki.py 不读这两个文件（每次直接扫 wiki/），格式只影响前端。
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from search_indexing import REPO_ROOT, iter_wiki_documents, token_counts

OUTPUT = REPO_ROOT / "docs" / "search-index.json"
LITE_OUTPUT = REPO_ROOT / "docs" / "search-index-lite.json"
LITE_FIELDS = ("id", "path", "title", "summary", "page_type", "tags", "search_aliases")


def build_search_payloads(docs: list[dict]) -> tuple[dict, dict]:
    """由 iter_wiki_documents() 的文档列表构建 (全文索引, 轻量索引) 两个 payload。"""
    total_docs = len(docs)
    doc_freq: Counter[str] = Counter()
    serialized_docs = []
    total_length = 0

    for doc in docs:
        text = "\n".join(
            [
                doc["title"],
                doc["summary"],
                doc["path"],
                " ".join(doc.get("tags", [])),
                " ".join(doc.get("search_aliases", [])),
                doc["body"],
            ]
        )
        counts = token_counts(text)
        dl = sum(counts.values())
        total_length += dl
        for token in counts:
            doc_freq[token] += 1
        serialized_docs.append(
            {
                "id": doc["id"],
                "path": doc["path"],
                "title": doc["title"],
                "summary": doc["summary"],
                "page_type": doc["page_type"],
                "tags": doc.get("tags", []),
                "search_aliases": doc.get("search_aliases", []),
                "dl": dl,
                "tokens": dict(sorted(counts.items())),
            }
        )

    avgdl = (total_length / total_docs) if total_docs else 0.0
    full = {
        "meta": {
            "version": "v1",
            "tier": "full",
            "avgdl": avgdl,
            "N": total_docs,
            "k1": 1.5,
            "b": 0.75,
        },
        # 前端 idf = ln(1 + (N - df + 0.5) / (df + 0.5))，与 search_indexing.bm25_idf 一致
        "df": {token: freq for token, freq in sorted(doc_freq.items())},
        "docs": serialized_docs,
    }
    lite = {
        "meta": {"version": "v1", "tier": "lite", "N": total_docs},
        "docs": [{key: entry[key] for key in LITE_FIELDS} for entry in serialized_docs],
    }
    return full, lite


def _dump(payload: dict) -> bytes:
    # 不缩进：序列化更快、文件更小
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def generate_search_index(output_path: Path = OUTPUT, lite_path: Path = LITE_OUTPUT) -> dict:
    full, lite = build_search_payloads(iter_wiki_documents())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_bytes = _dump(full)
    output_path.write_bytes(full_bytes)
    # 轻量索引里记下全文索引的压缩后大小，前端「搜索全文」按钮据此提示下载量
    lite["meta"]["full_index_gzip_bytes"] = len(gzip.compress(full_bytes, 6))
    lite_path.parent.mkdir(parents=True, exist_ok=True)
    lite_path.write_bytes(_dump(lite))
    return full


def main() -> None:
    payload = generate_search_index()
    print(f"Wrote {OUTPUT} with {len(payload['docs'])} docs (+ {LITE_OUTPUT.name})")


if __name__ == "__main__":
    main()
