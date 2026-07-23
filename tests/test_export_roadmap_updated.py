"""roadmap 页「更新时间」：无 frontmatter 时由 git 最近提交日回填。"""

from __future__ import annotations

from pathlib import Path

import export_minimal as em

ROOT = Path(__file__).resolve().parents[1]


def test_resolve_page_updated_roadmap_from_git() -> None:
    path = ROOT / "roadmap" / "depth-real2sim.md"
    assert path.is_file()
    updated = em.resolve_page_updated(path, {})
    assert updated is not None
    assert len(updated) == 10  # YYYY-MM-DD


def test_build_item_roadmap_includes_updated() -> None:
    path = ROOT / "roadmap" / "depth-real2sim.md"
    item = em.build_item(path)
    assert item["type"] == "roadmap_page"
    assert item.get("updated")
    assert len(str(item["updated"])) == 10


def test_resolve_page_updated_frontmatter_wins(tmp_path: Path, monkeypatch) -> None:
    roadmap_dir = tmp_path / "roadmap"
    roadmap_dir.mkdir()
    page = roadmap_dir / "depth-demo.md"
    page.write_text("# demo\n", encoding="utf-8")
    monkeypatch.setattr(em, "ROOT", tmp_path)
    monkeypatch.setattr(em, "_ROADMAP_GIT_UPDATED_CACHE", {"roadmap/depth-demo.md": "2020-01-01"})
    assert em.resolve_page_updated(page, {"updated": "2026-07-23"}) == "2026-07-23"
    assert em.resolve_page_updated(page, {}) == "2020-01-01"
