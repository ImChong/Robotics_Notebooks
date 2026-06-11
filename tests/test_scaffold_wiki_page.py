"""Unit tests for scripts/scaffold_wiki_page.py."""

from __future__ import annotations

from pathlib import Path

import scaffold_wiki_page as scaf


def test_standard_skeleton_passes_self_check() -> None:
    content = scaf.build_skeleton("concept", "视觉伺服")
    assert scaf.self_check(content, "concept") == []
    # 关键结构锚点
    assert "## 一句话定义" in content
    assert "## 英文缩写速查" in content
    assert "## 关联页面" in content
    assert "## 参考来源" in content


def test_abbrev_after_definition_before_why() -> None:
    content = scaf.build_skeleton("method", "示例方法")
    pos_def = content.find("## 一句话定义")
    pos_abbrev = content.find("## 英文缩写速查")
    pos_why = content.find("## 为什么重要")
    assert pos_def < pos_abbrev < pos_why


def test_query_skeleton_has_query_markers() -> None:
    content = scaf.build_skeleton("query", "感知选型")
    assert scaf.self_check(content, "query") == []
    assert "**Query 产物**" in content
    assert content.startswith("---")
    assert "# Query：感知选型" in content


def test_frontmatter_has_required_keys() -> None:
    content = scaf.build_skeleton("concept", "X")
    for key in ("type:", "updated:", "summary:", "related:", "sources:"):
        assert key in content


def test_slugify_derives_from_ascii_title() -> None:
    assert scaf.slugify("Visual Servoing Intro", None) == "visual-servoing-intro"
    assert scaf.slugify("anything", "explicit-slug") == "explicit-slug"
    # 纯中文标题无 override → 空 slug，交由 CLI 报错
    assert scaf.slugify("视觉伺服", None) == ""


def test_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scaf, "REPO_ROOT", tmp_path)
    rc = scaf.main(["concept", "Demo Page", "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "wiki" / "concepts" / "demo-page.md").exists()


def test_write_creates_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scaf, "REPO_ROOT", tmp_path)
    rc = scaf.main(["concept", "Demo Page", "--slug", "demo-page"])
    assert rc == 0
    out = tmp_path / "wiki" / "concepts" / "demo-page.md"
    assert out.exists()
    # 不允许无 --force 覆盖
    assert scaf.main(["concept", "Demo Page", "--slug", "demo-page"]) == 1
