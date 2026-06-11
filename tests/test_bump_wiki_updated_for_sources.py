"""bump_wiki_updated_for_sources.py 单元测试。"""

from __future__ import annotations

from pathlib import Path

import bump_wiki_updated_for_sources as bump


def test_bump_updated_sets_frontmatter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
    (tmp_path / "sources" / "papers").mkdir(parents=True)
    (tmp_path / "wiki" / "concepts").mkdir(parents=True)
    wiki = tmp_path / "wiki" / "concepts" / "bar.md"
    wiki.write_text("---\ntype: concept\nupdated: 2020-01-01\n---\n\n# bar\n", encoding="utf-8")
    src = tmp_path / "sources" / "papers" / "x.md"
    src.write_text("map [bar](../../wiki/concepts/bar.md)\n", encoding="utf-8")

    targets = bump.collect_wiki_targets([src])
    assert targets == [wiki.resolve()]
    assert bump.bump_updated(wiki, "2026-06-11")
    assert "updated: 2026-06-11" in wiki.read_text(encoding="utf-8")
