"""Tests for lint_wiki._build_link_index."""

from __future__ import annotations

from pathlib import Path

import lint_wiki as lw


def test_build_link_index_records_inbound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    a = wiki / "a.md"
    b = wiki / "b.md"
    a.write_text("Link [b](b.md).\n", encoding="utf-8")
    b.write_text("No backlink\n", encoding="utf-8")
    pages = [a, b]
    page_set = {p.resolve() for p in pages}
    inbound, broken = lw._build_link_index(pages, page_set)
    assert not broken
    br = b.resolve()
    assert br in inbound
    assert a.resolve() in inbound[br]


def test_build_link_index_reports_broken_relative_to_repo_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "REPO_ROOT", tmp_path)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    page = wiki / "a.md"
    page.write_text("[ghost](missing.md)\n", encoding="utf-8")
    pages = [page]
    page_set = {page.resolve()}
    _inbound, broken = lw._build_link_index(pages, page_set)
    assert len(broken) == 1
    broken_paths = next(iter(broken.values()))
    assert len(broken_paths) == 1
    assert broken_paths[0].replace("\\", "/") == "wiki/missing.md"
