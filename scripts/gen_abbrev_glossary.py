#!/usr/bin/env python3
"""为缺少『英文缩写速查』区块的 wiki 页面生成该区块。

数据源：schema/abbrev-glossary.json（缩写权威词典 + 别名）。
逐页扫描正文，仅列出该页实际出现的缩写，按首次出现顺序排列，
插入到第一个『## 参考来源』之前。已有该区块的页面跳过。

用法：
    python3 scripts/gen_abbrev_glossary.py            # 写入
    python3 scripts/gen_abbrev_glossary.py --dry-run  # 只统计不写盘
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = ROOT / "wiki"
DICT_FILE = ROOT / "schema" / "abbrev-glossary.json"
HEADING = "## 英文缩写速查"
SECTION_RE = re.compile(r"英文缩写速查|abbreviation glossary|abbreviations", re.IGNORECASE)


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    return text


def load_dict() -> tuple[dict, dict]:
    data = json.loads(DICT_FILE.read_text(encoding="utf-8"))
    return data["terms"], data.get("aliases", {})


def build_matchers(terms: dict, aliases: dict) -> list[tuple[str, re.Pattern]]:
    """返回 [(canonical, 正则)]，别名映射到对应主键。"""
    by_key: dict[str, list[str]] = {k: [k] for k in terms}
    for variant, canonical in aliases.items():
        by_key.setdefault(canonical, [canonical]).append(variant)
    matchers = []
    for canonical, surfaces in by_key.items():
        alt = "|".join(re.escape(s) for s in sorted(set(surfaces), key=len, reverse=True))
        pat = re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])")
        matchers.append((canonical, pat))
    return matchers


def detect(text: str, matchers: list[tuple[str, re.Pattern]]) -> list[str]:
    """返回该页出现的缩写主键，按首次出现位置排序。"""
    hits: dict[str, int] = {}
    for canonical, pat in matchers:
        m = pat.search(text)
        if m:
            hits[canonical] = m.start()
    return [k for k, _ in sorted(hits.items(), key=lambda kv: kv[1])]


NO_ABBREV_NOTE = "本页未引入需额外解释的英文缩写（相关术语在正文首次出现处已随文说明）。"


def render_section(keys: list[str], terms: dict) -> str:
    if not keys:
        return HEADING + "\n\n" + NO_ABBREV_NOTE + "\n\n"
    rows = ["| 缩写 | 英文全称 | 简要说明 |", "|------|----------|----------|"]
    for k in keys:
        t = terms[k]
        rows.append(f"| {k} | {t['en']} | {t['zh']} |")
    return HEADING + "\n\n" + "\n".join(rows) + "\n\n"


def insert_section(content: str, section: str) -> str:
    from wiki_abbrev_section import insert_abbrev_section

    return insert_abbrev_section(content, section)


def main() -> int:
    dry = "--dry-run" in sys.argv
    terms, aliases = load_dict()
    matchers = build_matchers(terms, aliases)

    pages = sorted(WIKI_DIR.rglob("*.md"))
    changed = 0
    zero_hits: list[str] = []
    for page in pages:
        rel_str = str(page.relative_to(ROOT))
        # 与 lint_wiki.py 一致：豁免 README/index 及 references/、roadmaps/ 下的元页面
        if page.name.lower() in ("readme.md", "index.md") or any(
            seg in rel_str for seg in ("references/", "roadmaps/")
        ):
            continue
        content = page.read_text(encoding="utf-8")
        if SECTION_RE.search(content):
            continue  # 已有区块
        keys = detect(strip_code(content), matchers)
        rel = page.relative_to(ROOT)
        if not keys:
            zero_hits.append(str(rel))
        section = render_section(keys, terms)
        new = insert_section(content, section)
        if not dry:
            page.write_text(new, encoding="utf-8")
        changed += 1

    print(f"{'[dry-run] ' if dry else ''}已生成区块: {changed} 页")
    if zero_hits:
        print(f"零命中（未写入，需人工处理）: {len(zero_hits)} 页")
        for z in zero_hits:
            print("  -", z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
