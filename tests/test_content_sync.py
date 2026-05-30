import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "export_minimal.py"
DETAIL_HTML = ROOT / "docs" / "detail.html"
MAIN_JS = ROOT / "docs" / "main.js"


class DetailContentSyncTests(unittest.TestCase):
    def test_export_script_mentions_content_markdown(self):
        content = EXPORT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("content_markdown", content)

    def test_detail_page_contains_content_mount_points(self):
        content = DETAIL_HTML.read_text(encoding="utf-8")
        self.assertIn('id="detailContentSection"', content)
        self.assertIn('id="detailContent"', content)

    def test_main_js_reads_content_markdown(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("document.getElementById('detailContent')", content)
        self.assertIn("content_markdown", content)

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
            "function stripYamlFrontmatter(markdown)",
            "function renderMarkdownContent(markdown, headings, markdownContext)",
            "contentEl.innerHTML = contentMarkdown ? renderMarkdownContent(contentMarkdown, detailHeadings, {",
            "blocks.push('<hr>');",
            "function renderCodeBlock(code, lang)",
            "function escapeMermaidForInnerHtml(text)",
            "return '<div class=\"mermaid\">' + escapeMermaidForInnerHtml(String(code || '').trim()) + '</div>';",
            "return '<blockquote>'",
            "return '<ul>'",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_mermaid_click_zoom_lightbox(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "MERMAID_FONT_SIZE_PX",
            "MERMAID_LIGHTBOX_FONT_SCALE",
            "function getMermaidSvgLayoutSize(svg)",
            "function renderMermaidSvgForLightbox(host)",
            "flowchart: {",
            "useMaxWidth: false",
            "function fitMermaidLightboxToView(stage, body)",
            "function cloneMermaidSvgForLightbox(svg)",
            "function openMermaidLightbox(host)",
            "function bindMermaidZoom(container)",
            "function bindMermaidLightboxWheel(body)",
            "function bindMermaidLightboxGestures(body)",
            "function applyMermaidLightboxPinchZoom(stage, body)",
            "mermaidLightboxPinchState",
            "mermaidLightboxPanX",
            "MERMAID_LIGHTBOX_ZOOM_MIN",
            "mermaid-lightbox-stage",
            "mermaid-zoomable",
            "mermaid-lightbox",
            "mermaid-lightbox-hint",
            "拖拽平移 · 滚轮/双指缩放 · Esc 关闭",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_toc_renderer_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function slugifyHeading(text)",
            "function collectMarkdownHeadings(markdown)",
            "function renderTocHeadingLabel(text, markdownContext)",
            "function renderDetailToc(container, headings, markdownContext)",
            "function bindDetailTocEntryNavigation(tocContainer)",
            "renderTocHeadingLabel(heading.text, context)",
            'class="toc-entry"',
            "document.getElementById('detailTocList')",
            "renderDetailToc(tocEl, detailHeadings, detailMarkdownContext)",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)
        render_detail_toc = content[
            content.find("function renderDetailToc") : content.find("function bindDetailTocEntryNavigation")
        ]
        self.assertNotIn("escapeHtml(heading.text)", render_detail_toc)

    def test_main_js_contains_math_rendering_hooks_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function renderMathBlocks(text)",
            'class="math-block"',
            'class="math-inline"',
            "expr.trim() +",
            "renderMathBlocks(renderInlineMarkdown(paragraphLines.join(",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_splits_table_cells_inside_inline_math(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function splitMarkdownTableCells(row)", content)
        self.assertIn("const cells = splitMarkdownTableCells(row);", content)

        node_script = r"""
function splitMarkdownTableCells(row) {
  const cells = [];
  let current = '';
  let inInlineMath = false;
  let inDisplayMath = false;
  let inParenMath = false;
  let inCode = false;
  const source = String(row || '');
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    const next = source[i + 1];
    if (!inCode && !inInlineMath && !inDisplayMath && !inParenMath && ch === '\\' && next === '|') {
      current += '\\|'; i += 2; continue;
    }
    if (!inInlineMath && !inDisplayMath && !inParenMath && ch === '`') {
      inCode = !inCode; current += ch; i++; continue;
    }
    if (!inCode && !inInlineMath && !inParenMath && ch === '$' && next === '$') {
      inDisplayMath = !inDisplayMath; current += '$$'; i += 2; continue;
    }
    if (!inCode && !inDisplayMath && ch === '$' && !inParenMath) {
      inInlineMath = !inInlineMath; current += ch; i++; continue;
    }
    if (!inCode && !inInlineMath && !inDisplayMath && ch === '\\' && next === '(') {
      inParenMath = true; current += '\\('; i += 2; continue;
    }
    if (inParenMath && ch === '\\' && next === ')') {
      inParenMath = false; current += '\\)'; i += 2; continue;
    }
    if (!inCode && !inInlineMath && !inDisplayMath && !inParenMath && ch === '|') {
      cells.push(current); current = ''; i++; continue;
    }
    current += ch; i++;
  }
  cells.push(current);
  const trimmed = cells.map(function (c) { return c.trim(); });
  if (trimmed.length > 0 && trimmed[0] === '') trimmed.shift();
  if (trimmed.length > 0 && trimmed[trimmed.length - 1] === '') trimmed.pop();
  return trimmed;
}
const cases = [
  [
    '| M3 | 3 | + 负载相关 $K_l\\|\\tau_m-\\tau_e\\|$ | eRob80:50 |',
    ['M3', '3', '+ 负载相关 $K_l\\|\\tau_m-\\tau_e\\|$', 'eRob80:50']
  ],
  [
    '| 摩擦锥 | $|f_x|, |f_y| \\leq \\mu f_z$，$f_z \\geq 0$ |',
    ['摩擦锥', '$|f_x|, |f_y| \\leq \\mu f_z$，$f_z \\geq 0$']
  ]
];
for (const [row, expected] of cases) {
  const got = splitMarkdownTableCells(row);
  if (JSON.stringify(got) !== JSON.stringify(expected)) {
    console.error(JSON.stringify({ row, expected, got }));
    process.exit(1);
  }
}
console.log('ok');
"""
        result = subprocess.run(
            ["node", "-e", node_script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("ok", result.stdout)

    def test_detail_page_loads_katex_assets(self):
        content = DETAIL_HTML.read_text(encoding="utf-8")
        expected_snippets = [
            "katex.min.css",
            "katex.min.js",
            "auto-render.min.js",
            "integrity=",
            'crossorigin="anonymous"',
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_invokes_katex_auto_render_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function renderDetailMath(container)",
            "typeof window.renderMathInElement !== 'function'",
            "window.renderMathInElement(container, {",
            "left: '$$', right: '$$', display: true",
            "left: '\\\\(', right: '\\\\)', display: false",
            "renderDetailMath(contentEl);",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_toc_active_state_and_anchor_copy_hooks(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function enhanceDetailHeadings(container)",
            "heading-anchor-link",
            "navigator.clipboard.writeText",
            "function bindDetailTocSpy(container, tocContainer)",
            "tocContainer.querySelectorAll('a[href^=\"#\"]')",
            "link.classList.toggle('active',",
            "enhanceDetailHeadings(contentEl);",
            "bindDetailTocSpy(contentEl, tocEl);",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_internal_markdown_link_routing_hooks(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function buildMarkdownRouteIndex(siteData)",
            "function normalizeInternalMarkdownTarget(target, currentPath)",
            "function resolveInternalMarkdownHref(target, currentPath, routeIndex)",
            "function renderInlineMarkdown(text, markdownContext)",
            "resolveInternalMarkdownHref(target, markdownContext.currentPath, markdownContext.routeIndex)",
            "renderMarkdownContent(contentMarkdown, detailHeadings, {",
            "currentPath: detailPage.path || ''",
            "routeIndex: markdownRouteIndex",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_contains_hash_navigation_hooks_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function scrollToDetailHashTarget(container)",
            "window.location.hash.replace(/^#/, '')",
            "decodeURIComponent(rawHash)",
            "container.querySelector('#' +",
            "target.scrollIntoView({ behavior: 'smooth', block: 'start' });",
            "function notifyTocSpyScrollSync()",
            "window.addEventListener('hashchange', function () {",
            "scrollToDetailHashTarget(contentEl);",
            "scrollDetailPageLayoutHashIntoView(contentEl);",
            "notifyTocSpyScrollSync();",
            "scrollToDetailHashTarget(contentEl);",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_search_match_explanation_escapes_user_facing_strings(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("return escapeHtml('核心标签命中: ' + tags[k]);", content)
        self.assertIn("return escapeHtml('标题命中: ' + t);", content)

    def test_main_js_defines_escape_html_once(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertEqual(content.count("function escapeHtml"), 1)

    def test_style_css_contains_heading_anchor_active_toc_and_hash_target_styles(self):
        style_content = (ROOT / "docs" / "style.css").read_text(encoding="utf-8")
        expected_snippets = [
            ".heading-anchor-link",
            ".detail-markdown-body h2:hover .heading-anchor-link",
            ".detail-toc-list a.active",
            ".detail-toc-list .toc-entry.active",
            ".detail-toc-list .toc-entry a",
            ".detail-hash-target",
            ".detail-markdown-body hr",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, style_content)


if __name__ == "__main__":
    unittest.main()
