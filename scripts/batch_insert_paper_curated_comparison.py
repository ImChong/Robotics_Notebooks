#!/usr/bin/env python3
"""Batch-insert ## 与其他工作对比（索引级） for paper-sa-* curated index pages (skip if present).

与 ``generate_sun254667_awesome_paper_entities.py`` 的模板保持同一段文案，
用于回填该生成器早期产出的 wiki/entities/paper-sa-*.md（lint 检查
``paper_missing_three_sections`` 的历史 backlog）。幂等：已含 ``## ...对比``
区块的页面直接跳过。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 「评测与指标」里的「横向对照」行，用于取回技术地图相对路径
TECH_MAP_RE = re.compile(
    r"^- 横向对照请回到 \[技术地图\]\((?P<rel>[^)]+)\) 同分组条目。$", re.MULTILINE
)
# 「核心信息（索引级）」表格里的分组行
SECTION_RE = re.compile(r"^\| 分组 \| (?P<section>.+?) \|$", re.MULTILINE)
HAS_COMPARE_RE = re.compile(r"^##\s+.*对比", re.MULTILINE)
CONCLUSION_ANCHOR = "\n## 结论\n"


def render_section(tech_map_rel: str, section: str) -> str:
    return (
        "## 与其他工作对比（索引级）\n"
        "\n"
        f"- 本页 **不做** 与具体基线的逐项数值对比：索引级节点只保留清单坐标，同分组横向对照请回到 [技术地图]({tech_map_rel}) 的 **{section}** 分组逐条展开。\n"
        "- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单 Highlights 阅读锚点。同一 arXiv 若已存在深度页，应以深度页为准。\n"
        "- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。\n"
    )


def process(path: Path) -> bool:
    """插入对比区块；已存在或缺少锚点时返回 False。"""
    content = path.read_text(encoding="utf-8")
    if HAS_COMPARE_RE.search(content):
        return False

    tech_map = TECH_MAP_RE.search(content)
    section = SECTION_RE.search(content)
    if not tech_map or not section or CONCLUSION_ANCHOR not in content:
        print(f"跳过（缺少锚点）：{path.relative_to(ROOT)}")
        return False

    block = render_section(tech_map.group("rel"), section.group("section").strip())
    content = content.replace(CONCLUSION_ANCHOR, "\n" + block + CONCLUSION_ANCHOR, 1)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    pages = sorted((ROOT / "wiki" / "entities").glob("paper-sa-*.md"))
    changed = sum(process(p) for p in pages)
    print(f"已插入对比区块：{changed}/{len(pages)} 个页面")


if __name__ == "__main__":
    main()
