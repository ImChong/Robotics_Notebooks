#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from search_indexing import REPO_ROOT, bm25_idf, iter_wiki_documents, token_counts

OUTPUT = REPO_ROOT / "docs" / "search-index.json"


def generate_search_index(output_path: Path = OUTPUT) -> dict:
    docs = iter_wiki_documents()
    total_docs = len(docs)
    doc_freq: Counter[str] = Counter()
    serialized_docs = []
    total_length = 0

    for doc in docs:
        text = "\n".join([
            doc["title"],
            doc["summary"],
            doc["path"],
            " ".join(doc.get("tags", [])),
            doc["body"],
        ])
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
                "dl": dl,
                "tokens": dict(sorted(counts.items())),
            }
        )

    avgdl = (total_length / total_docs) if total_docs else 0.0
    payload = {
        "meta": {
            "version": "v1",
            "avgdl": avgdl,
            "N": total_docs,
            "k1": 1.5,
            "b": 0.75,
        },
        "idf": {token: bm25_idf(freq, total_docs) for token, freq in sorted(doc_freq.items())},
        "docs": serialized_docs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = generate_search_index()
    print(f"Wrote {OUTPUT} with {len(payload['docs'])} docs")


if __name__ == "__main__":
    main()
