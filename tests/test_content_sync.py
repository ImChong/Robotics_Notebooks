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

    def test_detail_page_contains_toc_mount_points(self):
        content = DETAIL_HTML.read_text(encoding="utf-8")
        self.assertIn('id="detailTocSection"', content)
        self.assertIn('id="detailTocList"', content)

    def test_main_js_contains_markdown_renderer_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function stripYamlFrontmatter(markdown)',
            'function renderMarkdownContent(markdown, headings, markdownContext)',
            'contentEl.innerHTML = contentMarkdown ? renderMarkdownContent(contentMarkdown, detailHeadings, {',
            "blocks.push('<hr>');",
            'function renderCodeBlock(code, lang)',
            "return '<blockquote>'",
            "return '<ul>'",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_toc_renderer_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function slugifyHeading(text)',
            'function collectMarkdownHeadings(markdown)',
            'function renderDetailToc(container, headings)',
            "document.getElementById('detailTocList')",
            'renderDetailToc(tocEl, collectMarkdownHeadings(contentMarkdown));',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_math_rendering_hooks_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function renderMathBlocks(text)',
            'class="math-block"',
            'class="math-inline"',
            'expr.trim() +',
            'renderMathBlocks(renderInlineMarkdown(paragraphLines.join(',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_detail_page_loads_katex_assets(self):
        content = DETAIL_HTML.read_text(encoding="utf-8")
        expected_snippets = [
            'katex.min.css',
            'katex.min.js',
            'auto-render.min.js',
            'integrity=',
            'crossorigin="anonymous"',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_invokes_katex_auto_render_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function renderDetailMath(container)',
            "typeof window.renderMathInElement !== 'function'",
            'window.renderMathInElement(container, {',
            "left: '$$', right: '$$', display: true",
            "left: '\\\\(', right: '\\\\)', display: false",
            'renderDetailMath(contentEl);',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_toc_active_state_and_anchor_copy_hooks(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function enhanceDetailHeadings(container)',
            'heading-anchor-link',
            'navigator.clipboard.writeText',
            'function bindDetailTocSpy(container, tocContainer)',
            "tocContainer.querySelectorAll('a[href^=\"#\"]')",
            "link.classList.toggle('active',",
            'enhanceDetailHeadings(contentEl);',
            'bindDetailTocSpy(contentEl, tocEl);',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_internal_markdown_link_routing_hooks(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function buildMarkdownRouteIndex(siteData)',
            'function normalizeInternalMarkdownTarget(target, currentPath)',
            'function resolveInternalMarkdownHref(target, currentPath, routeIndex)',
            'function renderInlineMarkdown(text, markdownContext)',
            'resolveInternalMarkdownHref(target, markdownContext.currentPath, markdownContext.routeIndex)',
            'renderMarkdownContent(contentMarkdown, detailHeadings, {',
            'currentPath: detailPage.path || \'\'',
            'routeIndex: markdownRouteIndex',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_hash_navigation_hooks_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            'function scrollToDetailHashTarget(container)',
            "window.location.hash.replace(/^#/, '')",
            'decodeURIComponent(rawHash)',
            "container.querySelector('#' +",
            'target.scrollIntoView({ behavior: \'smooth\', block: \'start\' });',
            "window.addEventListener('hashchange', function () { scrollToDetailHashTarget(contentEl); });",
            'scrollToDetailHashTarget(contentEl);',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_style_css_contains_heading_anchor_active_toc_and_hash_target_styles(self):
        style_content = (ROOT / 'docs' / 'style.css').read_text(encoding='utf-8')
        expected_snippets = [
            '.heading-anchor-link',
            '.detail-markdown-body h2:hover .heading-anchor-link',
            '.detail-toc-list a.active',
            '.detail-hash-target',
            '.detail-markdown-body hr',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, style_content)


if __name__ == '__main__':
    unittest.main()
