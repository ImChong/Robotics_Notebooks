"""Unit tests for scripts/generate_page_catalog.py."""

from __future__ import annotations

from pathlib import Path

import generate_page_catalog as catalog


def _write_page(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
type: concept
date: 2026-07-10
---

# {title}

**{title}**：用于测试目录生成，并链接到 [相关页](../related.md)。
""",
        encoding="utf-8",
    )


def _patch_roots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    monkeypatch.setattr(catalog, "WIKI", tmp_path / "wiki")
    monkeypatch.setattr(catalog, "ROADMAP", tmp_path / "roadmap")
    monkeypatch.setattr(catalog, "TECHMAP", tmp_path / "tech-map")
    monkeypatch.setattr(catalog, "CATALOG", tmp_path / "catalog.md")


def test_render_catalog_uses_repository_relative_links(tmp_path: Path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _write_page(tmp_path / "wiki" / "concepts" / "demo.md", "Demo Concept")

    rendered = catalog.render_catalog()

    assert "[Demo Concept](wiki/concepts/demo.md)" in rendered
    assert "[相关页]" not in rendered
    assert "相关页" in rendered
    assert "[核心导航](index.md)" in rendered


def test_write_catalog_creates_standalone_file(tmp_path: Path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _write_page(tmp_path / "roadmap" / "demo.md", "Demo Roadmap")

    output = catalog.write_catalog()

    assert output == tmp_path / "catalog.md"
    assert output.exists()
    assert "[Demo Roadmap](roadmap/demo.md)" in output.read_text(encoding="utf-8")


def test_live_catalog_matches_generator() -> None:
    """Committed catalog.md should match the current generator output."""
    rendered = catalog.render_catalog()
    on_disk = catalog.CATALOG.read_text(encoding="utf-8")
    assert rendered.count("\n- [") == on_disk.count("\n- [")
