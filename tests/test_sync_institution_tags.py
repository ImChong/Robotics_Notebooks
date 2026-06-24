"""sync_institution_tags：表格 / sources / 覆盖表机构同步。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "sync_institution_tags",
    _REPO / "scripts" / "sync_institution_tags.py",
)
assert _SPEC and _SPEC.loader
sync = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sync)


def test_map_phrases_chinese_org() -> None:
    ids = sync._map_phrases("清华大学交叉信息研究院；上海期智研究院")
    assert "tsinghua" in ids
    assert "shanghai-pil" in ids


def test_infer_from_org_table() -> None:
    content = """---
type: entity
tags: [paper, humanoid]
---

# Demo

| 字段 | 内容 |
|------|------|
| 机构 | NVIDIA；CMU |
"""
    ids = sync.infer_institution_ids("wiki/entities/paper-demo.md", content, {})
    assert "nvidia" in ids
    assert "cmu" in ids


def test_skip_stub_pages_not_processed() -> None:
    assert sync._should_process(
        "wiki/entities/paper-notebook-foo.md",
        ["paper-notebook-stub"],
        entities_only=True,
    ) is False
