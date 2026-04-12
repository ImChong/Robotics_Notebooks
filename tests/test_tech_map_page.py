from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TECH_MAP_HTML = ROOT / "docs" / "tech-map.html"
MAIN_JS = ROOT / "docs" / "main.js"


class TechMapPageTests(unittest.TestCase):
    def test_tech_map_html_contains_data_driven_mount_points(self):
        content = TECH_MAP_HTML.read_text(encoding="utf-8")
        required_ids = [
            'id="techMapHeroSummary"',
            'id="techMapGraphMeta"',
            'id="techMapLayerList"',
            'id="techMapNodeGrid"',
        ]
        for marker in required_ids:
            self.assertIn(marker, content)

    def test_main_js_contains_tech_map_renderer(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function renderTechMapPage',
            "document.getElementById('techMapNodeGrid')",
            'detail.html?id=',
            'tech_map_page',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)


if __name__ == '__main__':
    unittest.main()
