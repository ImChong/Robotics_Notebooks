"""Unit tests for link parsing helpers in lint_wiki.py."""

from __future__ import annotations

from pathlib import Path

from lint_wiki import (
    extract_internal_links,
    has_section,
    has_source_reference,
    strip_code_blocks,
    strip_misconception_sections,
    word_count,
)


def test_strip_code_blocks_removes_fenced_and_inline() -> None:
    content = "intro\n```py\n[bad](x.md)\n```\n`inline [z](z.md)`\n[good](y.md)\n"
    out = strip_code_blocks(content)
    assert "x.md" not in out
    assert "z.md" not in out
    assert "[good](y.md)" in out


def test_extract_internal_links_skips_http_and_anchor(tmp_path: Path) -> None:
    page = tmp_path / "wiki" / "concepts" / "example.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    content = "[a](https://x.com/y.md)\n[b](#sec)\n[c](rel.md)\n"
    targets = extract_internal_links(content, page)
    assert len(targets) == 1
    assert targets[0] == (page.parent / "rel.md").resolve()


def test_extract_internal_links_ignores_links_inside_code_fence(tmp_path: Path) -> None:
    wiki_methods = tmp_path / "wiki" / "methods"
    wiki_methods.mkdir(parents=True)
    page = wiki_methods / "foo.md"
    content = "[keep](a.md)\n```text\n[skip](b.md)\n```\n[c](c.md)\n"
    targets = extract_internal_links(content, page)
    names = sorted(p.name for p in targets)
    assert names == ["a.md", "c.md"]


def test_extract_internal_links_resolves_relative_path(tmp_path: Path) -> None:
    concepts = tmp_path / "wiki" / "concepts"
    methods = tmp_path / "wiki" / "methods"
    concepts.mkdir(parents=True)
    methods.mkdir(parents=True)
    page = concepts / "page.md"
    content = "See [m](../methods/x.md)."
    targets = extract_internal_links(content, page)
    assert len(targets) == 1
    assert targets[0] == (methods / "x.md").resolve()


def test_has_section_matches_h2_keywords_case_insensitive() -> None:
    body = "## 关联页面\n\n- [x](a.md)\n"
    assert has_section(body, ["关联"])
    assert has_section(body, ["关联页面"])
    assert not has_section(body, ["参考来源"])


def test_word_count_chinese_and_english() -> None:
    assert word_count("") == 0
    assert word_count("你好 robot learning") == 2 + 2  # 2 个汉字 + 2 个英文词


def test_has_source_reference_detects_sources_md_links() -> None:
    assert has_source_reference("见 [p](../sources/papers/foo.md)。")
    assert has_source_reference("sources/repos/bar.md")
    assert not has_source_reference("仅 wiki 正文无 sources 链接。")


def test_strip_misconception_sections_removes_pitfall_block_until_peer_heading() -> None:
    content = (
        "## Intro\nkeep-intro\n## 常见误区\ndrop-body\n### 子节\ndrop-sub\n## Recovery\nkeep-tail\n"
    )
    out = strip_misconception_sections(content)
    assert "keep-intro" in out
    assert "keep-tail" in out
    assert "drop-body" not in out
    assert "drop-sub" not in out
    assert "常见误区" not in out


def test_strip_misconception_sections_strips_english_pitfall_heading() -> None:
    content = "## OK\nbefore\n## Pitfalls to avoid\nbad\n## After\nafter\n"
    out = strip_misconception_sections(content)
    assert "before" in out and "after" in out
    assert "bad" not in out
