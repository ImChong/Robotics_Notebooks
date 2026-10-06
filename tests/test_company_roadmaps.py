"""公司技术路线（company.html）：数据节点须指向站内真实页面，首页入口卡须与数据同序。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml
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
            assert node["id"] != DATA["comparison_id"], label
            if node["date"]:
                assert re.fullmatch(r"\d{4}-\d{2}", node["date"]), label
                datetime.strptime(node["date"], "%Y-%m")
            else:
                assert node.get("date_note", "").strip(), label


def test_route_detail_pages_have_resolvable_sources_and_core_sections() -> None:
    """详情必须有可读归纳与本地溯源，避免有效 ID 指向空壳或断开的 source。"""
    pages = {path_to_id(p, REPO_ROOT): p for p in (REPO_ROOT / "wiki").rglob("*.md")}
    for node_id in {n["id"] for c in DATA["companies"] for n in c["nodes"]}:
        page = pages[node_id]
        text = page.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        for heading in ("## 英文缩写速查", "## 参考来源"):
            assert heading in text, str(page)
        assert frontmatter.get("sources"), str(page)
        for source in frontmatter["sources"]:
            assert (page.parent / source.split("#")[0]).is_file(), f"{page}: {source}"
        assert "本页是 **清单索引**" not in text, str(page)


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


def test_hero_company_count_uses_home_links_at_runtime() -> None:
    """Hero 数字属于部署统计；新增入口时不要求 PR 重写历史 HTML 数字。"""
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'id="heroCompanyCount"[^>]*>(\d+)<', html)
    assert match and int(match.group(1)) > 0
    script = (REPO_ROOT / "docs" / "main.js").read_text(encoding="utf-8")
    assert "company: homeCompanyLinks.length || readHeroStatFallback(companyEl," in script


def test_companies_are_ordered_by_founding_year() -> None:
    years = [company["founded_year"] for company in DATA["companies"]]
    assert all(type(year) is int and 1900 <= year <= datetime.now().year for year in years)
    assert years == sorted(years)
    for company in DATA["companies"]:
        assert company["founded_source"].startswith("https://"), company["key"]
        assert company["founded_note"].strip(), company["key"]
