from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROADMAP_HTML = ROOT / "docs" / "roadmap.html"
MAIN_JS = ROOT / "docs" / "main.js"


class RoadmapPageTests(unittest.TestCase):
    def test_roadmap_page_contains_required_mount_points(self):
        content = ROADMAP_HTML.read_text(encoding="utf-8")
        required_ids = [
            'id="roadmapTitle"',
            'id="roadmapSummary"',
            'id="roadmapMeta"',
            'id="roadmapStageList"',
            'id="roadmapRelatedList"',
            'id="roadmapSourceList"',
        ]
        for marker in required_ids:
            self.assertIn(marker, content)

    def test_main_js_contains_roadmap_page_renderer(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function renderRoadmapPage',
            "document.getElementById('roadmapStageList')",
            'roadmap.html?id=',
            'roadmap_pages',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)


if __name__ == '__main__':
    unittest.main()
