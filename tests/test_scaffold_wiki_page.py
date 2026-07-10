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
    assert "## 核心原理" in content
    assert "## 工程实践" in content
    assert "## 局限与风险" in content
    assert "## 关联页面" in content
    assert "## 参考来源" in content


def test_standard_skeleton_uses_unified_reading_order() -> None:
    content = scaf.build_skeleton("method", "示例方法")
    positions = [content.find(heading) for heading in scaf.STANDARD_SECTION_ORDER]
    assert all(pos >= 0 for pos in positions)
    assert positions == sorted(positions)


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


def test_dataset_skeleton_has_quick_block_and_tag() -> None:
    content = scaf.build_skeleton("entity", "AMASS", dataset=True)
    assert scaf.self_check(content, "entity") == []
    assert "## 数据集速查" in content
    # frontmatter 含 dataset tag，使 lint 数据集巡检生效
    assert "  - dataset\n" in content
    # 五维度速查行齐全
    for dim in ("规模", "模态", "许可证", "适配形态", "重定向就绪度"):
        assert dim in content
    # 数据集速查是统一骨架中的类型专属补充，不改变主标题顺序
    assert (
        content.find("## 英文缩写速查")
        < content.find("## 数据集速查")
        < content.find("## 为什么重要")
        < content.find("## 核心原理")
        < content.find("## 工程实践")
        < content.find("## 局限与风险")
    )


def test_dataset_skeleton_passes_lint_metadata_check(tmp_path: Path, monkeypatch) -> None:
    """生成的数据集页应直接通过 lint 的数据集元数据巡检（0 缺失维度）。"""
    import lint_wiki

    monkeypatch.setattr(lint_wiki, "REPO_ROOT", tmp_path)
    page = tmp_path / "wiki" / "entities" / "demo-dataset.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(scaf.build_skeleton("entity", "Demo Dataset", dataset=True), encoding="utf-8")
    results: dict = {"dataset_missing_metadata": []}
    lint_wiki._check_dataset_entity_metadata([page], results)
    assert results["dataset_missing_metadata"] == []


def test_dataset_flag_rejected_for_non_entity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scaf, "REPO_ROOT", tmp_path)
    assert scaf.main(["concept", "X", "--slug", "x", "--dataset"]) == 2


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
