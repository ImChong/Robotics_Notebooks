#!/usr/bin/env python3
"""生成 Robotics Notebooks 的完整页面目录 catalog.md。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
WIKI = ROOT / "wiki"
ROADMAP = ROOT / "roadmap"
TECHMAP = ROOT / "tech-map"
CATALOG = ROOT / "catalog.md"

CATALOG_HEADER = """# Robotics Notebooks Page Catalog

> 本文件由 `python3 scripts/generate_page_catalog.py` 自动生成，请勿手工编辑。
> 返回 [核心导航](index.md)。
"""


def extract_frontmatter_date(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
        match = re.search(r"^date:\\s*(\\d{4}-\\d{2}-\\d{2})", content, re.MULTILINE)
        if match:
            return match.group(1)
        match = re.search(r"📅(\\d{4}-\\d{2}-\\d{2})", content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


def strip_frontmatter(content: str) -> str:
    """去除 YAML frontmatter（--- ... ---）。"""
    if not content.startswith("---"):
        return content
    end = content.find("\\n---", 3)
    if end == -1:
        return content
    return content[end + 4 :].lstrip()


def extract_first_sentence(content: str) -> str:
    """取第一段正文作为纯文本摘要，避免目录中的相对链接失效。"""
    for raw_line in strip_frontmatter(content).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(">") or line == "---":
            continue
        summary = line[:100].strip("[]()#*`|")
        summary = re.sub(r"\\[([^\\]]+)\\]\\([^)]+\\)", r"\\1", summary)
        if summary:
            return summary
    return "—"


def get_type(relative: Path) -> str:
    rel_str = str(relative)
    if "entities/" in rel_str:
        return "[entity_page]"
    if "references/" in rel_str:
        return "[reference_page]"
    if "roadmap/" in rel_str or "learning-paths/" in rel_str:
        return "[roadmap_page]"
    if "tech-map/" in rel_str:
        return "[tech_map_node]"
    if "concepts/" in rel_str:
        return "[wiki_page]"
    if "methods/" in rel_str:
        return "[method_page]"
    if "tasks/" in rel_str:
        return "[task_page]"
    if "formalizations/" in rel_str:
        return "[formalization_page]"
    if "comparisons/" in rel_str:
        return "[comparison_page]"
    if "overview/" in rel_str:
        return "[overview_page]"
    return "[wiki_page]"


def collect_pages(dir_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    if not dir_path.exists():
        return pages
    for md in sorted(dir_path.rglob("*.md")):
        relative = md.relative_to(ROOT)
        content = md.read_text(encoding="utf-8", errors="ignore")
        title_match = re.search(r"^#\\s+(.+)$", content, re.MULTILINE)
        pages.append(
            {
                "title": title_match.group(1).strip() if title_match else relative.stem,
                "path": str(relative).replace("\\\\", "/"),
                "summary": extract_first_sentence(content),
                "date": extract_frontmatter_date(md),
                "page_type": get_type(relative),
            }
        )
    return pages


def render_catalog() -> str:
    sections = [
        ("Entities（实体页）", WIKI / "entities"),
        ("Wiki Concepts（概念页）", WIKI / "concepts"),
        ("Wiki Methods（方法页）", WIKI / "methods"),
        ("Wiki Tasks（任务页）", WIKI / "tasks"),
        ("Wiki Formalizations（形式化基础）", WIKI / "formalizations"),
        ("Wiki Comparisons（对比页）", WIKI / "comparisons"),
        ("Wiki Overview（总览）", WIKI / "overview"),
        ("Roadmaps（路线页）", ROADMAP),
        ("Tech-map Nodes（技术栈节点）", TECHMAP),
        ("References（参考资料页）", ROOT / "references"),
    ]
    chunks = [CATALOG_HEADER.rstrip()]
    for section_name, dir_path in sections:
        pages = collect_pages(dir_path)
        if not pages:
            continue
        chunks.extend(["", f"### {section_name}", ""])
        chunks.extend(
            f"- [{page['title']}]({page['path']}) — {page['summary']} "
            f"`📅{page['date']}` `{page['page_type']}`"
            for page in pages
        )
    return "\\n".join(chunks).rstrip() + "\\n"


def write_catalog(path: Path | None = None) -> Path:
    target = path or CATALOG
    target.write_text(render_catalog(), encoding="utf-8")
    return target


def main() -> None:
    target = write_catalog()
    print(f"✓ 已生成 {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
