"""export_minimal：wiki / 路线页条目写出健康度，与 link-graph 同口径（详情页、路线页的健康度小条）。"""

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
    assert (item["health_score"], item["health_gaps"]) == glg.compute_roadmap_health(text, len(links))


def test_non_graph_pages_have_no_health() -> None:
    """references/ 等非图谱节点不写健康度。"""
    refs = sorted((ROOT / "references" / "papers").glob("*.md"))
    if refs:
        assert "health_score" not in build_item(refs[0])
