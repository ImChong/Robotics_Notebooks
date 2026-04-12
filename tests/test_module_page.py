from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_HTML = ROOT / "docs" / "module.html"
MAIN_JS = ROOT / "docs" / "main.js"


class ModulePageTests(unittest.TestCase):
    def test_module_page_html_exists_with_required_mount_points(self):
        self.assertTrue(MODULE_HTML.exists(), "docs/module.html should exist")
        content = MODULE_HTML.read_text(encoding="utf-8")
        required_ids = [
            'id="moduleTitle"',
            'id="moduleSummary"',
            'id="moduleMeta"',
            'id="moduleEntryList"',
            'id="moduleReferenceList"',
            'id="moduleRoadmapList"',
            'id="moduleRelatedModules"',
        ]
        for marker in required_ids:
            self.assertIn(marker, content)

    def test_main_js_contains_module_page_renderer(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function renderModulePage',
            "document.getElementById('moduleEntryList')",
            'module.html?id=',
            'module_pages',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)


if __name__ == '__main__':
    unittest.main()
