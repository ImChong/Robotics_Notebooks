"""公司技术路线（company.html）：数据节点须指向站内真实页面，首页入口卡须与数据同序。"""

from __future__ import annotations

import json
import re
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


def test_home_entry_links_match_data_order() -> None:
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    card = html[html.index('id="homeCompanyLinks"') : html.index('id="homeCompanyToggle"')]
    links = re.findall(r'<a [^>]*href="company\.html\?id=([^"]+)"([^>]*)>', card)
    assert [key for key, _ in links] == [c["key"] for c in DATA["companies"]]
    # 折叠态只露出前 4 家，其余带 data-company-extra hidden
    assert [("hidden" in attrs) for _, attrs in links] == [i >= 4 for i in range(len(links))]


def test_hero_company_count_fallback_matches_data() -> None:
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'id="heroCompanyCount"[^>]*>(\d+)<', html)
    assert match and int(match.group(1)) == len(DATA["companies"])
