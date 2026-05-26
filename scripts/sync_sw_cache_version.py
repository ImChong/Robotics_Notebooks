#!/usr/bin/env python3
"""Bump docs/sw.js CACHE_NAME from exports/graph-stats.json generated_at."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH_STATS = ROOT / "exports" / "graph-stats.json"
SW_PATH = ROOT / "docs" / "sw.js"
CACHE_RE = re.compile(r"const CACHE_NAME = 'robotics-wiki-[^']+';")


def resolve_build_id() -> str:
    if GRAPH_STATS.is_file():
        try:
            data = json.loads(GRAPH_STATS.read_text(encoding="utf-8"))
            generated = data.get("generated_at")
            if isinstance(generated, str) and generated.strip():
                return generated.strip()
        except (json.JSONDecodeError, OSError):
            pass
    return date.today().isoformat()


def main() -> int:
    build_id = resolve_build_id()
    cache_name = f"robotics-wiki-{build_id}"
    if not SW_PATH.is_file():
        print(f"missing {SW_PATH}", file=sys.stderr)
        return 1
    text = SW_PATH.read_text(encoding="utf-8")
    new_line = f"const CACHE_NAME = '{cache_name}';"
    if not CACHE_RE.search(text):
        print("CACHE_NAME pattern not found in sw.js", file=sys.stderr)
        return 1
    new_text = CACHE_RE.sub(new_line, text, count=1)
    if new_text == text:
        print(f"sw cache unchanged ({cache_name})")
        return 0
    SW_PATH.write_text(new_text, encoding="utf-8")
    print(f"bumped sw cache -> {cache_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
