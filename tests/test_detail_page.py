import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETAIL_HTML = ROOT / "docs" / "detail.html"
MAIN_JS = ROOT / "docs" / "main.js"


class DetailPageScaffoldTests(unittest.TestCase):
    def test_detail_page_html_exists_with_required_mount_points(self):
        self.assertTrue(DETAIL_HTML.exists(), "docs/detail.html should exist")
        content = DETAIL_HTML.read_text(encoding="utf-8")
        required_ids = [
            'id="detailBreadcrumb"',
            'id="detailTitle"',
            'id="detailSummary"',
            'id="detailMeta"',
            'id="detailTagList"',
            'id="detailRelatedList"',
            'id="detailSourceList"',
            'id="detailEmptyState"',
        ]
        for marker in required_ids:
            self.assertIn(marker, content)

    def test_main_js_contains_detail_page_renderer(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function renderDetailPage",
            "new URLSearchParams(window.location.search)",
            "document.getElementById('detailTitle')",
            "document.getElementById('detailTagList')",
            "document.getElementById('detailRelatedList')",
            "document.getElementById('detailSourceList')",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)


if __name__ == "__main__":
    unittest.main()
