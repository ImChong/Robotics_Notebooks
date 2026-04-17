#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from search_indexing import REPO_ROOT, hash_embed_texts, iter_wiki_documents, truncate_for_embedding

VECTOR_OUTPUT = REPO_ROOT / "exports" / "vector-index.npz"
META_OUTPUT = REPO_ROOT / "exports" / "vector-index-meta.json"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def encode_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> tuple[np.ndarray, dict]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32), {
            "backend": "sentence-transformers",
            "model": model_name,
            "dim": int(embeddings.shape[1]) if len(embeddings) else 0,
        }
    except Exception as exc:
        embeddings = hash_embed_texts(texts, dim=256)
        return embeddings, {
            "backend": "hashed-token-projection",
            "model": "builtin-v1",
            "dim": int(embeddings.shape[1]) if len(embeddings) else 256,
            "fallback_reason": str(exc),
        }


def build_vector_index(vector_output: Path = VECTOR_OUTPUT, meta_output: Path = META_OUTPUT, model_name: str = DEFAULT_MODEL) -> tuple[Path, Path, dict]:
    docs = iter_wiki_documents()
    texts = [truncate_for_embedding("\n".join([doc["title"], doc["summary"], doc["body"]])) for doc in docs]
    embeddings, meta = encode_texts(texts, model_name=model_name)

    vector_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vector_output, embeddings=embeddings)

    payload = {
        "version": "v1",
        "count": len(docs),
        "embedding": meta,
        "documents": [
            {
                "id": doc["id"],
                "path": doc["path"],
                "title": doc["title"],
                "page_type": doc["page_type"],
                "summary": doc["summary"],
            }
            for doc in docs
        ],
    }
    meta_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return vector_output, meta_output, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local vector index for wiki semantic search")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"sentence-transformers model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    vector_path, meta_path, payload = build_vector_index(model_name=args.model)
    backend = payload["embedding"]["backend"]
    print(f"Wrote {vector_path} ({payload['count']} docs, backend={backend})")
    print(f"Wrote {meta_path}")
    if payload["embedding"].get("fallback_reason"):
        print("Warning: sentence-transformers unavailable, used builtin hashed-token fallback")


if __name__ == "__main__":
    main()
