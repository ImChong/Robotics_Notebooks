#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WIKI_DIR = REPO_ROOT / "wiki"

ASCII_TOKEN_RE = re.compile(r"[a-z0-9_+\-.]+")
MIXED_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-.]+|[\u4e00-\u9fff]+")

TOKEN_SYNONYMS = {
    "稳定性": ["stability", "balance", "robustness"],
    "稳定": ["stability", "stable"],
    "运动控制": ["control", "locomotion", "motor-control"],
    "控制": ["control", "controller"],
    "腿足": ["legged", "biped", "humanoid", "locomotion"],
    "步态": ["gait", "locomotion"],
    "强化学习": ["rl", "reinforcement-learning"],
    "模仿学习": ["il", "imitation-learning", "behavior-cloning"],
    "行为克隆": ["behavior-cloning", "bc", "imitation-learning"],
    "李雅普诺夫": ["lyapunov", "stability"],
    "语义": ["semantic", "meaning"],
    "人形": ["humanoid"],
}


def parse_frontmatter(content: str) -> dict:
    fm = {}
    if not content.startswith("---"):
        return fm
    end = content.find("\n---", 3)
    if end == -1:
        return fm
    block = content[3:end].strip()
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [v.strip().strip('"\'') for v in val[1:-1].split(",") if v.strip()]
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
    return content[end + 4 :].lstrip()


def extract_title(body: str, fallback: str = "") -> str:
    match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def extract_summary(body: str, fm: dict | None = None) -> str:
    fm = fm or {}
    summary = (fm.get("summary") or fm.get("description") or "").strip()
    if summary:
        return summary
    lines = [line.strip() for line in body.splitlines()]
    for line in lines:
        if not line or line.startswith("#"):
            continue
        return re.sub(r"\s+", " ", line)[:180]
    return ""


def normalize_token(token: str) -> str:
    return token.strip().lower()


def _expand_cjk_segment(segment: str) -> List[str]:
    out: List[str] = []
    normalized = normalize_token(segment)
    if not normalized:
        return out
    out.append(normalized)
    if len(normalized) > 1:
        out.extend(list(normalized))
    if len(normalized) > 2:
        out.extend(normalized[i : i + 2] for i in range(len(normalized) - 1))
    return out


def tokenize_text(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in MIXED_TOKEN_RE.findall(text or ""):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_expand_cjk_segment(raw))
        else:
            normalized = normalize_token(raw)
            if normalized:
                tokens.append(normalized)
    enriched: List[str] = []
    for token in tokens:
        enriched.append(token)
        enriched.extend(TOKEN_SYNONYMS.get(token, []))
    return enriched


def truncate_for_embedding(text: str, max_tokens: int = 512) -> str:
    tokens: List[str] = []
    for raw in MIXED_TOKEN_RE.findall(text or ""):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(list(raw))
        else:
            tokens.append(raw)
        if len(tokens) >= max_tokens:
            break
    return " ".join(tokens[:max_tokens])


def path_to_id(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).parts
    stem = path.stem
    if parts[0] == "wiki":
        if parts[1] == "entities":
            return f"entity-{stem}"
        return f"wiki-{parts[1]}-{stem}"
    return stem


def page_type_for_path(path: Path, fm: dict) -> str:
    if fm.get("type"):
        return str(fm["type"])
    parts = path.relative_to(REPO_ROOT).parts
    if len(parts) >= 2 and parts[0] == "wiki":
        mapping = {
            "concepts": "concept",
            "methods": "method",
            "tasks": "task",
            "comparisons": "comparison",
            "overview": "overview",
            "roadmaps": "roadmap",
            "formalizations": "formalization",
            "queries": "query",
            "entities": "entity",
            "references": "reference",
        }
        return mapping.get(parts[1], parts[1])
    return ""


def iter_wiki_documents() -> List[Dict]:
    docs: List[Dict] = []
    for path in sorted(WIKI_DIR.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(raw)
        body = strip_frontmatter(raw)
        title = extract_title(body, path.stem)
        summary = extract_summary(body, fm)
        docs.append(
            {
                "id": path_to_id(path),
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "title": title,
                "summary": summary,
                "body": body,
                "frontmatter": fm,
                "page_type": page_type_for_path(path, fm),
                "tags": fm.get("tags", []),
            }
        )
    return docs


def token_counts(text: str) -> Counter:
    return Counter(tokenize_text(text))


def bm25_idf(doc_freq: int, total_docs: int) -> float:
    return math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def _hash_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def hash_embed_text(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    counts = token_counts(text)
    if not counts:
        return vec
    for token, count in counts.items():
        index, sign = _hash_index(token, dim)
        vec[index] += sign * float(count)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def hash_embed_texts(texts: Iterable[str], dim: int = 256) -> np.ndarray:
    vectors = [hash_embed_text(text, dim=dim) for text in texts]
    if not vectors:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)
