"""scripts/utils/markdown.py 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from utils.markdown import (
    extract_internal_links,
    find_chromium_executable,
    parse_frontmatter,
    parse_frontmatter_list,
    slugify_ascii,
    strip_code_blocks,
    strip_frontmatter,
)


def test_parse_and_strip_frontmatter() -> None:
    doc = "---\ntype: concept\ntags: [rl, control]\n---\n\n# 标题\n\n正文。"
    fm, body = parse_frontmatter(doc)
    assert fm == "type: concept\ntags: [rl, control]"
    assert body == "# 标题\n\n正文。"
    assert strip_frontmatter(doc) == body
    assert parse_frontmatter("# 无 frontmatter\n内容") == ("", "# 无 frontmatter\n内容")
    assert parse_frontmatter("---\nunclosed: true\n正文") == ("", "---\nunclosed: true\n正文")


def test_parse_frontmatter_list_inline_and_block() -> None:
    inline = "---\ntags: [rl, 'ppo', \"sim2real\"]\n---\n正文"
    assert parse_frontmatter_list(inline, "tags") == ["rl", "ppo", "sim2real"]
    block = "---\ntags:\n  - locomotion\n  - 'wbc'\n---\n正文"
    assert parse_frontmatter_list(block, "tags") == ["locomotion", "wbc"]
    assert parse_frontmatter_list(block, "missing") == []


def test_strip_code_blocks_and_extract_internal_links() -> None:
    md = (
        "见 [PPO](../methods/ppo.md#clip) 与 [概念](foo.md)。\n"
        "忽略 [外链](https://example.com) / [HTTP](http://a.b) / [邮件](mailto:a@b) / [锚点](#top)。\n"
        "```\n[代码块内](ignored.md)\n```\n"
        "~~~\n[波浪围栏](ignored2.md)\n~~~\n"
        "行内 `[内联](ignored3.md)`。\n"
    )
    stripped = strip_code_blocks(md)
    assert "ignored.md" not in stripped
    assert "ignored2.md" not in stripped
    assert "ignored3.md" not in stripped
    assert extract_internal_links(md) == ["../methods/ppo.md", "foo.md"]


def test_slugify_ascii() -> None:
    assert slugify_ascii("Model Predictive Control (MPC)") == "model-predictive-control-mpc"
    assert slugify_ascii("纯中文标题") == "item"
    assert slugify_ascii("纯中文标题", default="") == ""


def test_find_chromium_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CHROMIUM_PATH", "CHROME", "CHROME_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "empty-pw"))

    fake = tmp_path / "chrome"
    fake.write_text("", encoding="utf-8")
    assert find_chromium_executable(extra_candidates=[str(fake)]) == str(fake)

    monkeypatch.setenv("CHROME", "/custom/chrome")
    assert find_chromium_executable(extra_candidates=[str(fake)]) == "/custom/chrome"

    monkeypatch.setenv("CHROMIUM_PATH", "/primary/chromium")
    assert find_chromium_executable() == "/primary/chromium"
