from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TECH_MAP_HTML = ROOT / "docs" / "tech-map.html"
MAIN_JS = ROOT / "docs" / "main.js"


class TechMapFilterTests(unittest.TestCase):
    def test_tech_map_has_filter_mount_points(self):
        content = TECH_MAP_HTML.read_text(encoding="utf-8")
        self.assertIn('id="techMapFilterList"', content)
        self.assertIn('id="techMapFilterState"', content)

    def test_main_js_contains_filter_renderer(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected = [
            'function renderTechMapNodes',
            'function renderTechMapFilters',
            "document.getElementById('techMapFilterList')",
            'data-layer',
        ]
        for snippet in expected:
            self.assertIn(snippet, content)

    def test_main_js_contains_url_synced_layer_filter_logic(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected = [
            "params.get('layer')",
            'history.replaceState',
            'url.searchParams.set(\'layer\'',
            'url.searchParams.delete(\'layer\'',
            'currentLayer = initialLayer',
        ]
        for snippet in expected:
            self.assertIn(snippet, content)

    def test_main_js_contains_grouped_collapsible_rendering(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected = [
            'function renderTechMapNodeCard(node, detailPages)',
            'function renderTechMapGroupedNodes(nodes, detailPages)',
            '<details class="tech-map-group"',
            '<summary class="tech-map-group-summary">',
            'renderTechMapGroupedNodes(visibleNodes, detailPages)',
        ]
        for snippet in expected:
            self.assertIn(snippet, content)


if __name__ == '__main__':
    unittest.main()
