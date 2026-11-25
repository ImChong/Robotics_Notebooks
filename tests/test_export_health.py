"""export_minimal：wiki / 路线 / 参考索引 / 技术地图节点条目写出健康度，与 link-graph 同口径（详情页、路线页的健康度小条）。"""

from __future__ import annotations

from pathlib import Path

import generate_link_graph as glg
from export_minimal import build_item
from report_health import health_rows

ROOT = Path(__file__).resolve().parents[1]


def test_wiki_item_health_matches_link_graph() -> None:
    path = ROOT / "wiki" / "concepts" / "sim2real.md"
    item = build_item(path)
    text = path.read_text(encoding="utf-8")
    assert (item["health_score"], item["health_gaps"]) == glg.compute_health(text, path)


def test_roadmap_item_uses_roadmap_health_rule() -> None:
    path = ROOT / "roadmap" / "motion-control.md"
    item = build_item(path)
    text = path.read_text(encoding="utf-8")
    links = glg.extract_internal_links(text, path)
    assert (item["health_score"], item["health_gaps"]) == glg.compute_roadmap_health(
        text, len(links)
    )


def test_reference_page_health_uses_index_rubric() -> None:
    """参考索引页不套用 wiki 的单篇原文通读规则。"""
    path = ROOT / "references" / "papers" / "survey-papers.md"
    item = build_item(path)
    text = path.read_text(encoding="utf-8")
    links = glg.extract_internal_links(text, path)
    assert (item["health_score"], item["health_gaps"]) == glg.compute_reference_health(
        text, len(links)
    )
    assert (item["health_score"], item["health_gaps"]) == (3, 0)
    assert "```mermaid" in text
    assert health_rows(["references/papers/survey-papers.md"]) == [
        ("references/papers/survey-papers.md", 3, [])
    ]


def test_tech_map_node_health_uses_tech_map_rubric() -> None:
    """技术地图节点使用独立健康度口径，且输出 health_score / health_gaps。"""
    for rel in ("tech-map/overview.md", "tech-map/modules/control/mpc.md"):
        path = ROOT / rel
        item = build_item(path)
        text = path.read_text(encoding="utf-8")
        links = glg.extract_internal_links(text, path)
        assert (item["health_score"], item["health_gaps"]) == glg.compute_tech_map_health(
            text, len(links)
        )
        assert (item["health_score"], item["health_gaps"]) == (3, 0)
        assert "```mermaid" in text
        assert health_rows([rel]) == [(rel, 3, [])]


def test_non_detail_pages_not_exported_with_health() -> None:
    """非详情页面（如 README）不属于可访问详情页导出范围。"""
    readme = ROOT / "tech-map" / "README.md"
    assert readme.is_file()
    assert "health_score" not in build_item(ROOT / "README.md")
