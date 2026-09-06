"""Sitemap URLs must resolve against the deployed docs/ directory."""

from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import export_minimal as em


def locations(items, base=em.BASE_URL):
    root = ElementTree.fromstring(em.generate_sitemap(items, base))
    return [node.text for node in root.findall("{*}url/{*}loc")]


def test_sitemap_routes_resolve_to_deployed_files():
    items = [
        {"type": "wiki_page", "id": "wiki-concepts-ppo"},
        {"type": "entity_page", "id": "entity-robot"},
        {"type": "roadmap_page", "id": "roadmap-control"},
        {"type": "source", "id": "not-a-detail-page"},
    ]
    urls = locations(items, em.BASE_URL + "/")
    assert len(urls) == 7
    assert len(set(urls)) == len(urls)
    queries = {}
    for url in urls:
        parsed = urlsplit(url)
        assert parsed.netloc == "imchong.github.io"
        relative = parsed.path.removeprefix("/Robotics_Notebooks/")
        assert not relative.startswith("docs/")
        assert (em.ROOT / "docs" / (relative or "index.html")).is_file()
        if parsed.query:
            queries[parse_qs(parsed.query)["id"][0]] = relative
    assert queries == {
        "wiki-concepts-ppo": "detail.html",
        "entity-robot": "detail.html",
        "roadmap-control": "roadmap.html",
    }


def test_sitemap_preserves_ids_and_escapes_xml():
    item_id = '中文 &/#?"<node>'
    urls = locations([{"type": "wiki_page", "id": item_id}], "https://example.org/a&b/")
    assert urls[0] == "https://example.org/a&b/"
    assert parse_qs(urlsplit(urls[-1]).query) == {"id": [item_id]}


def test_sitemap_empty_catalog_keeps_static_routes():
    assert len(locations([])) == 4
