"""Shared helpers for ## 英文缩写速查 section placement in wiki pages."""

from __future__ import annotations

import re

ABBREV_HEADING = "## 英文缩写速查"
ABBREV_SECTION_RE = re.compile(
    r"^## 英文缩写速查\s*\n.*?(?=\n## |\Z)",
    re.MULTILINE | re.DOTALL,
)
DEFINITION_HEADING_RE = re.compile(
    r"^## (?:一句话(?:定义|总结|观点)|任务定义|是什么)\s*$",
    re.MULTILINE,
)
WHY_IMPORTANT_RE = re.compile(r"^## 为什么重要", re.MULTILINE)
H2_RE = re.compile(r"^## .+$", re.MULTILINE)
TAIL_SECTIONS = (
    "参考来源",
    "关联页面",
    "推荐继续阅读",
    "与其他页面的关系",
    "与其他系统的关系",
    "实验与评测",
)


def extract_abbrev_section(content: str) -> tuple[str | None, str]:
    """Return (section_text_with_trailing_newlines, content_without_section)."""
    m = ABBREV_SECTION_RE.search(content)
    if not m:
        return None, content
    section = m.group(0)
    if not section.endswith("\n"):
        section += "\n"
    stripped = content[: m.start()] + content[m.end() :]
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return section.rstrip() + "\n\n", stripped


def _section_end(content: str, heading_start: int) -> int:
    """Index after the section body (before next ## or EOF)."""
    rest = content[heading_start + 1 :]
    nxt = H2_RE.search(rest)
    if nxt:
        return heading_start + 1 + nxt.start()
    return len(content)


def _find_insert_index(content: str) -> int | None:
    """Return character index where abbrev section should start."""
    def_m = DEFINITION_HEADING_RE.search(content)
    if def_m:
        return _section_end(content, def_m.start())

    why_m = WHY_IMPORTANT_RE.search(content)
    if why_m:
        return why_m.start()

    # Query / checklist: after title block, before first substantive ##
    h1 = re.search(r"^# .+\n", content, re.MULTILINE)
    if h1:
        after_h1 = h1.end()
        first_h2 = H2_RE.search(content, after_h1)
        if first_h2:
            return first_h2.start()

    first_h2 = H2_RE.search(content)
    if first_h2:
        return first_h2.start()
    return None


def is_abbrev_glossary_well_placed(content: str) -> bool:
    """True if missing abbrev or section is in the canonical position."""
    section, _ = extract_abbrev_section(content)
    if section is None:
        return True

    pos_abbrev = content.find(ABBREV_HEADING)
    if pos_abbrev < 0:
        return True

    why_m = WHY_IMPORTANT_RE.search(content)
    if why_m and pos_abbrev > why_m.start():
        return False

    for tail in TAIL_SECTIONS:
        tail_m = re.search(rf"^## {re.escape(tail)}", content, re.MULTILINE)
        if tail_m and pos_abbrev > tail_m.start():
            return False

    def_m = DEFINITION_HEADING_RE.search(content)
    if def_m and pos_abbrev < def_m.start():
        return False
    if def_m:
        def_end = _section_end(content, def_m.start())
        if pos_abbrev < def_end:
            return False

    return True


def insert_abbrev_section(content: str, section: str) -> str:
    """Insert a new abbrev section at the canonical position."""
    insert_at = _find_insert_index(content)
    if insert_at is None:
        sep = "" if content.endswith("\n") else "\n"
        return content + sep + "\n" + section
    before = content[:insert_at].rstrip() + "\n\n"
    after = content[insert_at:].lstrip("\n")
    return before + section + after


def reorder_abbrev_glossary(content: str) -> tuple[str, bool]:
    """Move ## 英文缩写速查 to canonical position. Returns (new_content, changed)."""
    if is_abbrev_glossary_well_placed(content):
        return content, False

    section, body = extract_abbrev_section(content)
    if section is None:
        return content, False

    insert_at = _find_insert_index(body)
    if insert_at is None:
        return content, False

    # Trim duplicate blank lines at splice points
    before = body[:insert_at].rstrip() + "\n\n"
    after = body[insert_at:].lstrip("\n")
    new_content = before + section + after
    return new_content, True
