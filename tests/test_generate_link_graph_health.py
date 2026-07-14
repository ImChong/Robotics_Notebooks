"""健康度评分：frontmatter 规则与 roadmap/ 正文信号规则。

回归背景：roadmap/ 路线页按约定不带 frontmatter，旧逻辑统一走
``compute_health_score`` 导致 14 个路线节点健康度恒为 0。
"""

from __future__ import annotations

import generate_link_graph as glg


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
