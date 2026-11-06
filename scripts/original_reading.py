"""原文通读的结构化判据（schema/ingest-workflow.md 步骤 1.6）。

- wiki 页 frontmatter ``original:`` 声明原文归档（相对页面的路径，可多个）；
- 每份原文归档（sources/ 下）有且只有一行 ``- **通读：** 已完成`` 或 ``- **通读：** 未完成``。

lint_wiki 校验两处格式，generate_link_graph 据此计算健康度「原文已读完」。
"""

from __future__ import annotations

import re
from pathlib import Path

from search_indexing import parse_frontmatter

READING_FIELD_RE = re.compile(r"^- \*\*通读：\*\*[ \t]*(\S*)[ \t]*$", re.MULTILINE)
READING_DONE = "已完成"
READING_VALUES = frozenset({READING_DONE, "未完成"})


def reading_status(path: Path) -> str | None:
    """sources/ 归档的通读取值；文件或字段缺失、字段重复、取值不合法时返回 None。"""
    if not path.is_file():
        return None
    values = READING_FIELD_RE.findall(path.read_text(encoding="utf-8"))
    if len(values) == 1 and values[0] in READING_VALUES:
        return values[0]
    return None


def original_entries(content: str) -> list[str]:
    """wiki frontmatter ``original:`` 的条目（标量按单条处理）。"""
    if "original:" not in content:
        return []
    value = parse_frontmatter(content).get("original")
    if not value:
        return []
    return [value] if isinstance(value, str) else [str(item) for item in value]


def original_fully_read(content: str, page: Path, sources_dir: Path) -> bool:
    """页面声明了原文，且每份原文都是 sources/ 下标「通读：已完成」的归档。"""
    root = sources_dir.resolve()
    paths = [(page.parent / entry).resolve() for entry in original_entries(content)]
    return bool(paths) and all(
        path.is_relative_to(root) and reading_status(path) == READING_DONE for path in paths
    )
