"""实验室技术路线（lab.html）：数据节点须指向站内真实页面，首页入口卡须与数据同序。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from utils.paths import path_to_id

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((REPO_ROOT / "docs" / "lab-roadmaps.json").read_text(encoding="utf-8"))
COMPANY_DATA = json.loads(
    (REPO_ROOT / "docs" / "company-roadmaps.json").read_text(encoding="utf-8")
)


def _site_pages() -> dict[str, Path]:
    pages = list((REPO_ROOT / "wiki").rglob("*.md")) + list((REPO_ROOT / "roadmap").glob("*.md"))
    return {path_to_id(p, REPO_ROOT): p for p in pages}


def test_lab_keys_unique_and_not_listed_as_company() -> None:
    keys = [lab["key"] for lab in DATA["labs"]]
    assert len(keys) == len(set(keys))
    assert not set(keys) & {c["key"] for c in COMPANY_DATA["companies"]}
    for lab in DATA["labs"]:
        assert lab["lenses"], lab["key"]
        assert set(lab["lenses"]) <= set(DATA["lenses"]), lab["key"]


def test_every_node_links_site_page_with_sources() -> None:
    pages = _site_pages()
    assert DATA["comparison_id"] in pages
    for lab in DATA["labs"]:
        assert lab["nodes"], lab["key"]
        assert lab["official"].startswith("https://"), lab["key"]
        dates = []
        for node in lab["nodes"]:
            label = f"{lab['key']}/{node['title']}"
            assert node["id"] in pages, label
            assert node["url"].startswith("https://"), label
            if node["date"]:
                assert re.fullmatch(r"\d{4}-\d{2}", node["date"]), label
                datetime.strptime(node["date"], "%Y-%m")
                dates.append(node["date"])
            else:
                assert node.get("date_note", "").strip(), label
            page = pages[node["id"]]
            text = page.read_text(encoding="utf-8")
            for heading in ("## 英文缩写速查", "## 参考来源"):
                assert heading in text, str(page)
            assert "本页是 **清单索引**" not in text, str(page)
            sources = yaml.safe_load(text.split("---", 2)[1]).get("sources")
            assert sources, str(page)
            for source in sources:
                assert (page.parent / source.split("#")[0]).is_file(), f"{page}: {source}"
        assert dates == sorted(dates), lab["key"]


def test_labs_are_ordered_by_founding_year() -> None:
    years = [lab["founded_year"] for lab in DATA["labs"]]
    assert all(type(year) is int and 1900 <= year <= datetime.now().year for year in years)
    assert years == sorted(years)
    for lab in DATA["labs"]:
        assert lab["founded_source"].startswith("https://"), lab["key"]
        assert lab["founded_note"].strip(), lab["key"]


def test_home_entry_links_match_data_order() -> None:
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    start = html.index('id="homeLabLinks"')
    card = html[start : html.index("</div>", start)]
    keys = re.findall(r'<a [^>]*href="lab\.html\?id=([^"]+)"', card)
    assert keys == [lab["key"] for lab in DATA["labs"]]
    # 独立卡片默认展示四个，后续入口可展开，避免新增路线撑长首页。
    tags = re.findall(r'<a [^>]*href="lab\.html\?id=[^"]+"[^>]*>', card)
    assert len(tags) > 4
    assert all("hidden" not in tag for tag in tags[:4])
    assert all("data-lab-extra hidden" in tag for tag in tags[4:])
    toggle = re.search(r'id="homeLabToggle"[^>]*>展开全部 (\d+) 个实验室', html)
    assert toggle and int(toggle.group(1)) == len(keys)
    assert 'id="home-lab-routes" hidden' not in html
    assert 'id="homeOrgTabLab"' not in html


def test_hero_lab_count_uses_home_links_at_runtime() -> None:
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'id="heroLabCount"[^>]*>(\d+)<', html)
    assert match and int(match.group(1)) > 0
    script = (REPO_ROOT / "docs" / "main.js").read_text(encoding="utf-8")
    assert "lab: homeLabLinks.length || readHeroStatFallback(labEl," in script
