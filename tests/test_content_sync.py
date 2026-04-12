from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "export_minimal.py"
DETAIL_HTML = ROOT / "docs" / "detail.html"
MAIN_JS = ROOT / "docs" / "main.js"


class DetailContentSyncTests(unittest.TestCase):
    def test_export_script_mentions_content_markdown(self):
        content = EXPORT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('content_markdown', content)

    def test_detail_page_contains_content_mount_points(self):
        content = DETAIL_HTML.read_text(encoding="utf-8")
        self.assertIn('id="detailContentSection"', content)
        self.assertIn('id="detailContent"', content)

    def test_main_js_reads_content_markdown(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("document.getElementById('detailContent')", content)
        self.assertIn('content_markdown', content)

    def test_detail_page_uses_markdown_container_instead_of_preformatted_box(self):
        content = DETAIL_HTML.read_text(encoding="utf-8")
        self.assertIn('class="detail-markdown-body data-loading"', content)
        self.assertNotIn('<pre id="detailContent"', content)

    def test_main_js_contains_markdown_renderer_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function renderMarkdownContent(markdown)',
            'contentEl.innerHTML = contentMarkdown ? renderMarkdownContent(contentMarkdown)',
            "return '<pre><code>'",
            "return '<blockquote>'",
            "return '<ul>'",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)


if __name__ == '__main__':
    unittest.main()
