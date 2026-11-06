"""健康度评分：frontmatter 规则与 roadmap/ 正文信号规则。

回归背景：roadmap/ 路线页按约定不带 frontmatter，旧逻辑统一走
``compute_health_score`` 导致 14 个路线节点健康度恒为 0。
"""

from __future__ import annotations

from pathlib import Path

import generate_link_graph as glg
import pytest
from scaffold_wiki_page import build_skeleton


def test_compute_health_score_requires_frontmatter() -> None:
    """无 frontmatter 页面在 wiki 规则下仍为 0（既有行为不变）。"""
    assert glg.compute_health_score("# 标题\n\n正文。\n") == 0


def test_roadmap_full_signals_score_three() -> None:
    content = "# 路线（纵深）：示例\n\n**摘要**：一句话摘要。\n\n## Stage 0 全景\n\n正文。\n"
    assert glg.compute_roadmap_health_score(content, internal_link_count=10) == 3


def test_roadmap_main_route_headings_count_as_stage_structure() -> None:
    """主路线的 ``## L−1``…``## L7`` 分层标题同样算阶段化结构。"""
    content = "# 主路线\n\n**首屏导读**：\n\n## L−1 序言\n\n## L0 数学\n"
    assert glg.compute_roadmap_health_score(content, internal_link_count=0) == 2


def test_roadmap_missing_all_signals_scores_zero() -> None:
    assert glg.compute_roadmap_health_score("# 标题\n\n正文。\n", internal_link_count=0) == 0


def test_roadmap_link_threshold_boundary() -> None:
    content = "# 标题\n\n正文。\n"
    threshold = glg.ROADMAP_HEALTH_MIN_INTERNAL_LINKS
    assert glg.compute_roadmap_health_score(content, internal_link_count=threshold - 1) == 0
    assert glg.compute_roadmap_health_score(content, internal_link_count=threshold) == 1


def test_all_repo_roadmap_pages_score_full() -> None:
    """仓库现有路线页应全部拿满 3 分（曾全部为 0 的回归防线）。"""
    pages = [p for p in sorted(glg.ROADMAP_DIR.glob("*.md")) if p.name != "README.md"]
    assert pages, "roadmap/ 下应存在路线页"
    for page in pages:
        content = page.read_text(encoding="utf-8")
        links = glg.extract_internal_links(content, page)
        score = glg.compute_roadmap_health_score(content, len(links))
        assert score == 3, f"{page.name} 健康度 {score} != 3"


def _skeleton(status: str = "complete", source: str = "TODO/source.md") -> str:
    """scaffold 生成的标准骨架（必然通过 self_check），可替换 status 与 sources 链接。"""
    content = build_skeleton("concept", "示例")
    content = content.replace("status: draft", f"status: {status}")
    return content.replace("sources/TODO/source.md", f"sources/{source}")


@pytest.fixture
def sources_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离的 sources/ 目录：full.md 已读完，partial.md 声明未读完，bare.md 无核对记录。"""
    (tmp_path / "full.md").write_text("- **核对范围：** 已通读 PDF 全文。\n", encoding="utf-8")
    (tmp_path / "partial.md").write_text(
        "## 核对范围与待办\n\n已核对摘要与方法；全文尚未逐节通读。\n\n## 摘录\n", encoding="utf-8"
    )
    (tmp_path / "bare.md").write_text("# 无核对记录\n", encoding="utf-8")
    monkeypatch.setattr(glg, "SOURCES_DIR", tmp_path)
    return tmp_path


def test_source_reading_state(sources_dir: Path) -> None:
    assert glg.source_reading_state(sources_dir / "full.md") == "full"
    assert glg.source_reading_state(sources_dir / "partial.md") == "partial"
    assert glg.source_reading_state(sources_dir / "bare.md") is None
    assert glg.source_reading_state(sources_dir / "missing.md") is None


@pytest.mark.parametrize(
    "record",
    [
        "官网首页及仓库 README；完整技术文档尚未通读。",
        "论文引言、方法、实验与部分附录；其余附录尚待逐项核对。",
        "仅 arXiv 摘要页与仓库 README；论文全文未通读。",
        "已核对训练仓库中 JAX 引擎入口；尚未逐项验证 PyTorch 与 CUDA 内核。",
        "README 固定版本；不代表本库已完成每个模型的全文及源码核查。",
    ],
)
def test_partial_reading_phrases(tmp_path: Path, record: str) -> None:
    """仓库内真实「未读完」措辞均判为 partial。"""
    page = tmp_path / "s.md"
    page.write_text(f"- **核对范围：** {record}\n", encoding="utf-8")
    assert glg.source_reading_state(page) == "partial"


def test_full_reading_record_not_misread_as_partial(tmp_path: Path) -> None:
    """「未标注版本号」等与阅读无关的否定不影响判定。"""
    page = tmp_path / "s.md"
    page.write_text("- **核对范围：** 通读文章正文；页面未标注版本号。\n", encoding="utf-8")
    assert glg.source_reading_state(page) == "full"


def test_health_score_signals(sources_dir: Path) -> None:
    """元数据 / 骨架达标 / 原文读完 各 +1。"""
    assert glg.compute_health_score(_skeleton(source="full.md")) == 3
    assert glg.compute_health_score(_skeleton(source="bare.md")) == 2
    assert glg.compute_health_score(_skeleton(source="partial.md")) == 2
    minimal = "---\nsummary: x\nsources:\n  - ../../sources/full.md\n---\n\n正文。\n"
    assert glg.compute_health_score(minimal) == 2
    assert glg.compute_health_score(minimal.replace("full.md", "bare.md")) == 1


def test_any_partial_source_blocks_fully_read(sources_dir: Path) -> None:
    """链接多份归档时，只要有一份声明未读完就不算读完。"""
    content = _skeleton(source="full.md") + "\n- [b](../../sources/partial.md)\n"
    assert glg.compute_health_score(content) == 2


def test_status_caps_health_by_ingest_completion(sources_dir: Path) -> None:
    """作者自报未完成的页按 status 封顶：stub ≤1，draft ≤2，complete 不封顶。"""
    assert glg.compute_health_score(_skeleton("stub", "full.md")) == 1
    assert glg.compute_health_score(_skeleton("draft", "full.md")) == 2
    assert glg.compute_health_score(_skeleton("complete", "full.md")) == 3


def test_status_cap_does_not_raise_low_score() -> None:
    """封顶只降不升。"""
    assert glg.compute_health_score("---\nstatus: complete\n---\n\n正文。\n") == 0


def test_reading_skeleton_accepts_lint_synonyms() -> None:
    """论文页常用的「核心机制」「局限与工程实践」视为覆盖核心原理 / 工程实践 / 局限与风险。"""
    heads = ["一句话定义", "英文缩写速查", "为什么重要", "核心机制", "局限与工程实践"]
    heads += ["关联页面", "参考来源", "推荐继续阅读"]
    content = "\n\n".join(f"## {h}\n\n正文" for h in heads)
    assert glg.has_reading_skeleton(content, "entity")
    assert not glg.has_reading_skeleton(content.replace("## 为什么重要", "## 核心信息"), "entity")


def test_query_reading_skeleton_uses_query_sections() -> None:
    content = "## 英文缩写速查\n\n## TL;DR 决策结论\n\n## 关联页面\n\n## 参考来源\n"
    assert glg.has_reading_skeleton(content, "query")
    assert not glg.has_reading_skeleton(content, "concept")
