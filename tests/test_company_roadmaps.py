"""首页「具身公司 · 技术路线」数据：docs/company-roadmaps.json 的节点必须指向站内真实页面。"""

from __future__ import annotations

import json
from pathlib import Path

from utils.paths import path_to_id

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((REPO_ROOT / "docs" / "company-roadmaps.json").read_text(encoding="utf-8"))


def _site_ids() -> set[str]:
    pages = list((REPO_ROOT / "wiki").rglob("*.md")) + list((REPO_ROOT / "roadmap").glob("*.md"))
    return {path_to_id(p, REPO_ROOT) for p in pages}


def test_company_keys_unique_and_lenses_known() -> None:
    keys = [c["key"] for c in DATA["companies"]]
    assert len(keys) == len(set(keys))
    for company in DATA["companies"]:
        assert company["lenses"], company["key"]
        assert set(company["lenses"]) <= set(DATA["lenses"]), company["key"]


def test_every_node_links_site_page_and_original() -> None:
    site_ids = _site_ids()
    assert DATA["comparison_id"] in site_ids
    for company in DATA["companies"]:
        assert company["nodes"], company["key"]
        assert company["official"].startswith("https://"), company["key"]
        for node in company["nodes"]:
            label = f"{company['key']}/{node['title']}"
            assert node["id"] in site_ids, label
            assert node["url"].startswith("https://"), label
            assert node["date"] == "" or len(node["date"]) == 7, label


def test_dated_nodes_are_chronological() -> None:
    for company in DATA["companies"]:
        dates = [n["date"] for n in company["nodes"] if n["date"]]
        assert dates == sorted(dates), company["key"]
