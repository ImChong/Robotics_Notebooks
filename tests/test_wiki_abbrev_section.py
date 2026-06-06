"""Tests for wiki abbrev glossary section placement helpers."""

from __future__ import annotations

from wiki_abbrev_section import (
    is_abbrev_glossary_well_placed,
    reorder_abbrev_glossary,
)


def test_well_placed_after_definition_before_why() -> None:
    content = """# Title

## 一句话定义

定义正文。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

动机。

## 参考来源

- [x](../sources/a.md)
"""
    assert is_abbrev_glossary_well_placed(content)
    new, changed = reorder_abbrev_glossary(content)
    assert not changed


def test_misplaced_after_why_gets_moved() -> None:
    content = """# Title

intro paragraph.

## 为什么重要

- point

## 核心组件

body

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |

## 参考来源

- [x](../sources/a.md)
"""
    assert not is_abbrev_glossary_well_placed(content)
    new, changed = reorder_abbrev_glossary(content)
    assert changed
    assert new.index("## 英文缩写速查") < new.index("## 为什么重要")
    assert new.index("## 核心组件") > new.index("## 为什么重要")


def test_misplaced_after_definition_section() -> None:
    content = """# Title

## 一句话定义

定义。

## 为什么重要

why.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |

## 参考来源

- [x](../sources/a.md)
"""
    new, changed = reorder_abbrev_glossary(content)
    assert changed
    assert new.index("## 英文缩写速查") < new.index("## 为什么重要")
    assert new.index("## 英文缩写速查") > new.index("## 一句话定义")
