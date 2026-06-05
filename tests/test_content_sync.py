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
            "contentEl.innerHTML = contentMarkdown ? renderMarkdownContent(contentMarkdown, detailHeadings, detailMarkdownContext)",
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
            "htmlLabels: true",
            "useMaxWidth: false",
            "bindRoadmapSectionMermaidRerender",
            "function fitMermaidLightboxToView(stage, body)",
            "function cloneMermaidSvgForLightbox(svg)",
            "function openMermaidLightbox(host)",
            "function bindMermaidZoom(container)",
            "function fixMermaidForeignObjectOverflow(svg)",
            "function patchMermaidSvgLabelOverflow(container)",
            "MERMAID_LABEL_OVERFLOW_PAD",
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
            "function buildDetailTocTree(headings)",
            "function renderDetailTocList(nodes, markdownContext)",
            "function stripTocHeadingNumberPrefix(text, level)",
            "function renderTocHeadingLabel(text, markdownContext)",
            "function renderDetailToc(container, headings, markdownContext)",
            "function bindDetailTocEntryNavigation(tocContainer)",
            "renderTocHeadingLabel(",
            'class="toc-entry"',
            "document.getElementById('detailTocList')",
            "renderDetailToc(tocEl, detailHeadings, detailMarkdownContext)",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)
        render_detail_toc = content[
            content.find("function renderDetailToc") : content.find(
                "function bindDetailTocEntryNavigation"
            )
        ]
        self.assertNotIn("escapeHtml(heading.text)", render_detail_toc)

    def test_detail_toc_nested_tree_and_strip_number_prefix(self):
        """TOC 应按标题层级嵌套，并去掉 h3/h4 自带的小节序号前缀。"""
        node = r"""
const fs = require('fs');
function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function renderTocHeadingLabel(text) {
  return escapeHtml(text);
}
function tocHeadingLabelHasInnerLink() {
  return false;
}
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function stripTocHeadingNumberPrefix');
const labelStart = content.indexOf('function renderTocHeadingLabel', start);
const listStart = content.indexOf('function renderDetailTocList', start);
const listEnd = content.indexOf('function renderDetailToc(container', listStart);
const block = content.slice(start, labelStart) + content.slice(listStart, listEnd);
eval(block);
const headings = [
  { level: 2, text: '主要方法', slug: 'main-methods' },
  { level: 3, text: '1. Domain Randomization', slug: 'dr' },
  { level: 3, text: '2. System Identification', slug: 'sysid' },
  { level: 2, text: '常见误区', slug: 'pitfalls' },
];
const html = renderDetailTocList(buildDetailTocTree(headings));
if (!html.includes('<ol><li class="toc-level-2">')) throw new Error('missing root ol');
if (!html.includes('<ol><li class="toc-level-3">')) throw new Error('missing nested ol');
if (html.includes('1. Domain Randomization')) throw new Error('number prefix not stripped');
if (!html.includes('Domain Randomization')) throw new Error('expected stripped label');
if ((html.match(/<ol>/g) || []).length < 2) throw new Error('expected nested ol');
console.log('ok');
"""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(node)
            tmp_path = tmp.name
        result = subprocess.run(
            ["node", tmp_path, str(MAIN_JS)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("ok", result.stdout)

    def test_main_js_contains_math_rendering_hooks_for_detail_content(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function renderMathBlocks(text)",
            "function applyMathBlocksInHtmlFragment(html)",
            "function convertMermaidFencesInHtmlFragment(html)",
            'class="math-block"',
            'class="math-inline"',
            "expr.trim() +",
            "renderMathBlocks(renderInlineMarkdown(paragraphLines.join(",
            "convertMermaidFencesInHtmlFragment(htmlBlockLines.join",
            "applyMathBlocksInHtmlFragment(htmlFragment)",
            "if (htmlBlockOpenTag) {",
            "htmlBlockLines.push(line);",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_html_fragment_math_gets_inline_wrapper(self):
        """<details> 等原样 HTML 块内的 \\(...\\) 也应包 math-inline，与 detail 正文一致。"""
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function renderMathBlocks(text)');
const end = content.indexOf('function splitMarkdownTableCells', start);
eval(content.slice(start, end));
const sample = [
  '<details class="selftest-answers">',
  '<summary>参考答案</summary>',
  '<ol><li>矩阵（\\(R^\\top R=I\\)）</li></ol>',
  '</details>',
].join('\n');
const out = applyMathBlocksInHtmlFragment(sample);
if (!out.includes('class="math-inline"')) throw new Error('missing math-inline wrapper');
if (out.includes('矩阵（\\(R')) throw new Error('unwrapped delimiter remains');
console.log('ok');
"""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(node)
            tmp_path = tmp.name
        result = subprocess.run(
            ["node", tmp_path, str(MAIN_JS)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("ok", result.stdout)

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
            "tocContainer.querySelectorAll('a[href^=\"#\"], .toc-entry[data-href]')",
            "item.classList.toggle('active',",
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
            "Angle-bracket autolinks: <https://...>",
            "/<(https?:\\/\\/[^>\\s]+)>/gi",
            "function renderLinkLabel(label)",
            "renderLinkLabel(label)",
            "resolveInternalMarkdownHref(target, markdownContext.currentPath, markdownContext.routeIndex)",
            "const detailMarkdownContext = {",
            "currentPath: detailPage.path || ''",
            "routeIndex: markdownRouteIndex",
            "renderMarkdownContent(contentMarkdown, detailHeadings, detailMarkdownContext)",
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
            ".detail-toc-list ol ol > li",
            ".detail-hash-target",
            ".detail-markdown-body hr",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, style_content)


if __name__ == "__main__":
    unittest.main()
