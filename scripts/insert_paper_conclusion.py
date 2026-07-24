#!/usr/bin/env python3
"""Insert or replace ## 结论 on paper entity pages at the preferred anchor.

Preferred order: after 实验/评测-like section, before 对比/局限/关联/参考.
Usage:
  python3 scripts/insert_paper_conclusion.py path/to/paper.md <<'EOF'
**总判。**

1. **要点** — 说明
EOF
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

AFTER_PATTERNS = [
    r"实验与评测",
    r"实验要点",
    r"评测与结果",
    r"主要量化结果",
    r"评测",
    r"实验结果",
    r"量化结果",
    r"结果",
]

BEFORE_PATTERNS = [
    r"与其他工作",
    r"与相邻",
    r"与其他页面",
    r"局限与风险",
    r"常见误区",
    r"开源状态",
    r"关联页面",
    r"参考来源",
    r"推荐继续阅读",
]


def _heading_re(patterns: list[str]) -> re.Pattern[str]:
    joined = "|".join(patterns)
    return re.compile(rf"^##\s+.*(?:{joined}).*$", re.MULTILINE | re.IGNORECASE)


AFTER_RE = _heading_re(AFTER_PATTERNS)
BEFORE_RE = _heading_re(BEFORE_PATTERNS)
CONCLUSION_BLOCK_RE = re.compile(
    r"^##\s+结论\s*\n(?:.*?\n)*?(?=^##\s+|\Z)",
    re.MULTILINE,
)


def next_heading_start(text: str, start: int) -> int:
    m = re.search(r"^##\s+", text[start:], re.MULTILINE)
    return start + m.start() if m else len(text)


def find_insert_pos(body: str) -> int:
    """Return character offset where the conclusion block should start."""
    # Prefer: immediately before the first "before" heading that appears
    # after an "after" heading; else before first "before" heading; else EOF.
    after_matches = list(AFTER_RE.finditer(body))
    before_matches = list(BEFORE_RE.finditer(body))

    if after_matches and before_matches:
        after_end = next_heading_start(body, after_matches[0].end())
        for bm in before_matches:
            if bm.start() >= after_end:
                return bm.start()
        # all "before" headings are before the after section — fall through

    if before_matches:
        # Prefer 与其他工作 / 局限 over 关联/参考
        priority = [
            r"与其他工作",
            r"与相邻",
            r"局限与风险",
            r"常见误区",
            r"与其他页面",
            r"开源状态",
            r"关联页面",
            r"参考来源",
            r"推荐继续阅读",
        ]
        for pat in priority:
            for bm in before_matches:
                if re.search(pat, bm.group(0)):
                    return bm.start()
        return before_matches[0].start()

    if after_matches:
        return next_heading_start(body, after_matches[-1].end())

    return len(body)


def normalize_conclusion(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise SystemExit("empty conclusion body")
    if not text.startswith("##"):
        text = "## 结论\n\n" + text
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text


def apply(path: Path, conclusion_raw: str, replace: bool = True) -> str:
    content = path.read_text(encoding="utf-8")
    conclusion = normalize_conclusion(conclusion_raw)

    if re.search(r"^##\s+结论\b", content, re.MULTILINE):
        if not replace:
            return "skip-existing"
        new_content, n = CONCLUSION_BLOCK_RE.subn(conclusion, content, count=1)
        if n != 1:
            raise SystemExit(f"failed to replace existing conclusion in {path}")
        path.write_text(new_content, encoding="utf-8")
        return "replaced"

    # Split frontmatter so heading search stays in body
    m = re.match(r"^(---\n.*?\n---\n)([\s\S]*)$", content, re.DOTALL)
    if m:
        fm, body = m.group(1), m.group(2)
        pos = find_insert_pos(body)
        new_body = body[:pos] + conclusion + body[pos:]
        path.write_text(fm + new_body, encoding="utf-8")
    else:
        pos = find_insert_pos(content)
        path.write_text(content[:pos] + conclusion + content[pos:], encoding="utf-8")
    return "inserted"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: insert_paper_conclusion.py <paper.md> [--no-replace]")
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = REPO / path
    replace = "--no-replace" not in sys.argv
    raw = sys.stdin.read()
    status = apply(path, raw, replace=replace)
    print(f"{status}: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
