"""Unit tests for link parsing helpers in lint_wiki.py."""

from __future__ import annotations

from pathlib import Path

from lint_wiki import extract_internal_links, strip_code_blocks


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
