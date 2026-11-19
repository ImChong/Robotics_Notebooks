"""export_minimal：wiki / 路线 / 参考索引页条目写出健康度，与 link-graph 同口径（详情页、路线页的健康度小条）。"""

from __future__ import annotations

from pathlib import Path

import generate_link_graph as glg
from export_minimal import build_item

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


def test_other_non_graph_pages_have_no_health() -> None:
    """非 wiki / roadmap / references 的技术地图页仍不写健康度。"""
    tech_pages = sorted((ROOT / "tech-map").rglob("*.md"))
    if tech_pages:
        assert "health_score" not in build_item(tech_pages[0])
