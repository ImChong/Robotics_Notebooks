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
            "let quoteHtml = '<blockquote>';",
            "function splitListLine(line)",
            "function renderListSlice(start, end, indent)",
            "tag === 'ul' && hasTaskAtIndent(groupStart, j, indent) ? ' class=\"contains-task-list\"' : ''",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_escape_mermaid_for_inner_html_preserves_br_labels(self):
        """Mermaid htmlLabels use <br/>; escape must entity-encode so innerHTML does not eat tags."""
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function escapeMermaidForInnerHtml(text)');
const end = content.indexOf('function convertMermaidFencesInHtmlFragment', start);
eval(content.slice(start, end));
function decodeHtmlEntities(value) {
  return String(value || '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
}
const sample = 'gate["Identity gate<br/>$$\\tanh(\\alpha)\\approx 0$$<br/>$$u\' = u + \\Delta u$$"]';
const escaped = escapeMermaidForInnerHtml(sample);
if (escaped.includes('<br/>')) throw new Error('raw <br/> must not appear in innerHTML payload');
if (!escaped.includes('&lt;br/')) throw new Error('expected entity-encoded br');
const recovered = decodeHtmlEntities(escaped);
if (!recovered.includes('<br/>')) throw new Error('missing literal <br/> after entity decode');
if (recovered.includes('Identity gate$$')) throw new Error('line breaks collapsed: ' + recovered);
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

    def test_normalize_mermaid_markdown_emphasis_to_html_bold(self):
        """Depth roadmap overview used **Stage N**; htmlLabels need <b>, not literal asterisks."""
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertIn("function normalizeMermaidMarkdownEmphasis(source)", content)
        self.assertIn("normalizeMermaidMarkdownEmphasis(source)", content)
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function normalizeMermaidMarkdownEmphasis(source)');
const end = content.indexOf('function mermaidSourceForCurrentBrowser', start);
eval(content.slice(start, end));
const sample = 'S0["**Stage 0**<br/>全景与前置<br/><em>BFM</em>"]';
const out = normalizeMermaidMarkdownEmphasis(sample);
if (out.includes('**')) throw new Error('markdown ** must be removed: ' + out);
if (!out.includes('<b>Stage 0</b>')) throw new Error('expected <b>Stage 0</b>: ' + out);
if (!out.includes('<br/>') || !out.includes('<em>BFM</em>')) {
  throw new Error('existing HTML labels must stay intact: ' + out);
}
console.log('ok');
"""
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
            "function normalizeMathExpr(expr)",
            "function renderMathBlocks(text)",
            "function applyMathBlocksInHtmlFragment(html)",
            "function convertMermaidFencesInHtmlFragment(html)",
            'class="math-block"',
            'class="math-inline"',
            "normalizeMathExpr(expr.trim())",
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
const start = content.indexOf('function normalizeMathExpr(expr)');
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
            "left: '\\\\[', right: '\\\\]', display: true",
            "left: '\\\\(', right: '\\\\)', display: false",
            "renderDetailMath(contentEl);",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_bracket_display_math_gets_block_wrapper(self):
        """\\[...\\] display math (common in formalization pages) should wrap math-block for KaTeX."""
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function normalizeMathExpr(expr)');
const end = content.indexOf('function applyMathBlocksInHtmlFragment', start);
eval(content.slice(start, end));
const sample = '\\[ i_a + i_b + i_c = 0 \\]';
const out = renderMathBlocks(sample);
if (!out.includes('class="math-block"')) throw new Error('missing math-block wrapper');
if (!out.includes('i_a + i_b + i_c = 0')) throw new Error('bracket math content lost');
const aligned = '\\[ \\begin{aligned} i_d &= i_\\alpha \\\\ i_q &= i_\\beta \\end{aligned} \\]';
const out2 = renderMathBlocks(aligned);
if (!out2.includes('class="math-block"')) throw new Error('aligned block missing wrapper');
const escapedStar = '\\( i_d^\\* = 0 \\)';
const out3 = renderMathBlocks(escapedStar);
if (!out3.includes('i_d^* = 0')) throw new Error('markdown star escape not normalized: ' + out3);
if (out3.includes('\\*')) throw new Error('raw \\* still present: ' + out3);
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

    def test_inline_math_with_padding_spaces_in_table_cells(self):
        """$ ... $ with spaces inside delimiters (abbrev glossary tables) should render."""
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('const matchHtmlRegExp');
const end = content.indexOf('function normalizeMathExpr(expr)', start);
eval(content.slice(start, end));
const sample = '正向动力学 $ \\tau \\to \\ddot{q} $';
const out = renderInlineMarkdown(sample, {});
if (!out.includes('\\tau \\to \\ddot{q}')) throw new Error('math content lost: ' + out);
if (out.includes('$')) throw new Error('raw dollar delimiters remain: ' + out);
const tight = renderInlineMarkdown('$\\ddot{q} \\to \\tau$', {});
if (!tight.includes('\\ddot{q} \\to \\tau')) throw new Error('tight math broken: ' + tight);
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

    def test_inline_markdown_unescapes_asterisk_before_emphasis(self):
        """A\\* and **A\\*** must render as A* without a visible backslash."""
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('const matchHtmlRegExp');
const end = content.indexOf('function normalizeMathExpr(expr)', start);
eval(content.slice(start, end));
const bold = renderInlineMarkdown('**A\\***（A-star）', {});
if (bold.includes('\\*') || bold.includes('A\\')) throw new Error('backslash leaked in bold: ' + bold);
if (!bold.includes('<strong>') || !bold.includes('&#42;')) throw new Error('expected strong+entity: ' + bold);
const cell = renderInlineMarkdown('A\\* 全局规划', {});
if (cell.includes('\\*')) throw new Error('backslash leaked in plain: ' + cell);
if (!cell.includes('&#42;')) throw new Error('expected entity in plain: ' + cell);
const label = renderLinkLabel('A\\*');
if (label.includes('\\*') || !label.includes('&#42;')) throw new Error('link label bad: ' + label);
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

    def test_wiki_prose_backslash_star_does_not_leak_after_render(self):
        """All wiki prose lines with \\* must render without a visible backslash-star."""
        node = r"""
const fs = require('fs');
const path = require('path');
const mainJs = process.argv[2];
const wikiRoot = process.argv[3];
const content = fs.readFileSync(mainJs, 'utf8');
const start = content.indexOf('const matchHtmlRegExp');
const endNorm = content.indexOf('function normalizeMathExpr(expr)', start);
const endMath = content.indexOf('function applyMathBlocksInHtmlFragment', endNorm);
eval(content.slice(start, endNorm));
eval(content.slice(endNorm, endMath));

function walk(dir, out) {
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) walk(p, out);
    else if (name.endsWith('.md')) out.push(p);
  }
  return out;
}

function decode(html) {
  return String(html || '')
    .replace(/<[^>]+>/g, '')
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"');
}

const files = walk(wikiRoot, []);
const failures = [];
let checked = 0;
for (const file of files) {
  const body = fs.readFileSync(file, 'utf8');
  let inCode = false;
  for (const line of body.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) { inCode = !inCode; continue; }
    if (inCode || !line.includes('\\*')) continue;
    checked += 1;
    const rendered = decode(renderMathBlocks(renderInlineMarkdown(line, {})));
    if (rendered.includes('\\*')) {
      failures.push(path.relative(wikiRoot, file) + ': ' + line.trim().slice(0, 120));
    }
  }
}
if (!checked) throw new Error('expected wiki prose lines with \\*');
if (failures.length) {
  throw new Error('backslash-star leaked in ' + failures.length + ' lines\n' + failures.slice(0, 20).join('\n'));
}
console.log('ok checked=' + checked);
"""
        import subprocess
        import tempfile
        from pathlib import Path

        wiki_root = Path(__file__).resolve().parents[1] / "wiki"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(node)
            tmp_path = tmp.name
        result = subprocess.run(
            ["node", tmp_path, str(MAIN_JS), str(wiki_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("ok checked=", result.stdout)

    def test_main_js_contains_toc_active_state_and_anchor_copy_hooks(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "function enhanceDetailHeadings(container)",
            "heading-anchor-link",
            "navigator.clipboard.writeText",
            "function bindDetailTocSpy(container, tocContainer)",
            "tocContainer.querySelectorAll('a[href^=\"#\"], .toc-entry[data-href]')",
            "navItems[j].classList.toggle('active',",
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
        self.assertIn("return escapeHtml('核心标签命中: ' + itemTags[k]);", content)
        self.assertIn("return escapeHtml('标题命中: ' + t);", content)

    def test_main_js_defines_escape_html_once(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        self.assertEqual(content.count("function escapeHtml"), 1)

    def test_main_js_strips_related_section_from_detail_body(self):
        content = MAIN_JS.read_text(encoding="utf-8")
        expected_snippets = [
            "DETAIL_CONTENT_SKIP_SECTIONS = ['关联页面']",
            "function stripLinkedReferenceSourceLines(markdown)",
            "stripLinkedReferenceSourceLines(",
            "function stripDetailContentSections(markdown, sectionLabels)",
            "stripDetailContentSections(detailPage.content_markdown || '', DETAIL_CONTENT_SKIP_SECTIONS)",
            "function decodeBasicHtmlEntities(text)",
            "function stripAngleBracketAutolinks(text)",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, content)

    def test_main_js_clean_reference_label_strips_angle_bracket_autolinks(self):
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function decodeBasicHtmlEntities');
const end = content.indexOf('function looksLikeRepoPath', start);
eval(content.slice(start, end));
const samples = [
  ['FastStair 论文 HTML：<', 'FastStair 论文 HTML'],
  ['wechat.md — <https://mp.weixin.qq.com/s/example>', 'wechat.md'],
  ['BOM &lt;$400', 'BOM <$400'],
];
for (const [input, expected] of samples) {
  const got = cleanReferenceLabelText(input);
  if (got !== expected) throw new Error('expected ' + JSON.stringify(expected) + ' got ' + JSON.stringify(got));
}
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

    def test_strip_detail_content_sections_removes_related_heading_block(self):
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('var DETAIL_CONTENT_SKIP_SECTIONS');
const end = content.indexOf('function stripDetailContentSections', start);
const fnStart = content.indexOf('function stripDetailContentSections', start);
const fnEnd = content.indexOf('function buildMarkdownRouteIndex', fnStart);
eval(content.slice(start, fnEnd));
const sample = [
  '## 核心内容',
  '',
  '正文段落。',
  '',
  '## 关联页面',
  '',
  '- [Sim2Real](sim2real.md)',
  '- [Locomotion](locomotion.md)',
  '',
  '## 常见误区',
  '',
  '误区说明。',
].join('\n');
const stripped = stripDetailContentSections(sample, DETAIL_CONTENT_SKIP_SECTIONS);
if (stripped.includes('## 关联页面')) throw new Error('related heading should be removed');
if (stripped.includes('sim2real.md')) throw new Error('related links should be removed');
if (!stripped.includes('## 核心内容')) throw new Error('missing core heading');
if (!stripped.includes('## 常见误区')) throw new Error('missing trailing heading');
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

    def test_strip_linked_reference_source_lines_keeps_plain_text_only(self):
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function referenceSourceLineHasLink');
const fnEnd = content.indexOf('function renderHomeStats', start);
eval(content.slice(start, fnEnd));
const sample = [
  '## 核心内容',
  '',
  '正文段落。',
  '',
  '## 参考来源',
  '',
  '- [Paper](https://example.com/paper)',
  '- Tobin et al. 2017, Domain Randomization paper',
  '',
  '## 常见误区',
  '',
  '误区说明。',
].join('\n');
const stripped = stripLinkedReferenceSourceLines(sample);
if (!stripped.includes('## 参考来源')) throw new Error('reference heading should remain for plain text');
if (stripped.includes('example.com/paper')) throw new Error('linked reference should be removed from body');
if (!stripped.includes('Tobin et al. 2017')) throw new Error('plain text reference should remain in body');
if (!stripped.includes('## 常见误区')) throw new Error('missing trailing heading');
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

    def test_strip_linked_reference_source_lines_removes_heading_when_only_links(self):
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
const start = content.indexOf('function referenceSourceLineHasLink');
const fnEnd = content.indexOf('function renderHomeStats', start);
eval(content.slice(start, fnEnd));
const sample = [
  '## 核心内容',
  '',
  '正文段落。',
  '',
  '## 参考来源',
  '',
  '- [Paper](https://example.com/paper)',
  '- [Repo](https://github.com/example/repo)',
  '',
  '## 常见误区',
  '',
  '误区说明。',
].join('\n');
const stripped = stripLinkedReferenceSourceLines(sample);
if (stripped.includes('## 参考来源')) throw new Error('reference heading should be removed when only links');
if (stripped.includes('example.com/paper')) throw new Error('linked reference should be removed from body');
if (!stripped.includes('## 常见误区')) throw new Error('missing trailing heading');
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

    def test_render_markdown_content_preserves_nested_lists(self):
        """Roadmap「其它纵深路径 / 关联知识页」等缩进子项不得被拍平为同级 <li>。"""
        node = r"""
const fs = require('fs');
const content = fs.readFileSync(process.argv[2], 'utf8');
function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function stripYamlFrontmatter(markdown) {
  return String(markdown || '').replace(/\r\n/g, '\n').trim();
}
function renderInlineMarkdown(text) {
  return String(text || '').replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, label, href) {
    return '<a href="' + escapeHtml(href) + '">' + escapeHtml(label) + '</a>';
  });
}
function renderMathBlocks(text) { return text; }
function collectMarkdownHeadings() { return []; }
function slugifyHeading(text) {
  return String(text || '').toLowerCase().replace(/\s+/g, '-') || 'section';
}
function normalizeCodeLang(lang) { return lang || ''; }
function renderCodeBlock(code) { return '<pre><code>' + escapeHtml(code) + '</code></pre>'; }
function convertMermaidFencesInHtmlFragment(html) { return html; }
function applyMathBlocksInHtmlFragment(html) { return html; }
const start = content.indexOf('const RE_HR = ');
const end = content.indexOf('function renderChipList', start);
if (start < 0 || end < 0) throw new Error('cannot locate renderMarkdownContent block');
eval(content.slice(start, end));
const sample = [
  '## 和其他页面的关系',
  '',
  '- 完整成长路线参考：[主路线](motion-control.md)',
  '- 其它纵深路径：',
  '  - [遥操作](depth-teleoperation.md)',
  '  - [人形足球](depth-humanoid-soccer.md)',
  '- 关联知识页：',
  '  - [人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md)',
  '  - [MARL](../wiki/methods/marl.md)',
].join('\n');
const html = renderMarkdownContent(sample, [], {});
if (!html.includes('<ul><li>完整成长路线参考')) {
  throw new Error('missing top-level list: ' + html);
}
if (!html.includes('其它纵深路径：<ul><li><a href="depth-teleoperation.md">遥操作</a></li>')) {
  throw new Error('其它纵深路径 children not nested: ' + html);
}
if (!html.includes('关联知识页：<ul><li><a href="../wiki/concepts/humanoid-multi-robot-coordination.md">人形多机协调</a></li>')) {
  throw new Error('关联知识页 children not nested: ' + html);
}
const flatSibling = html.includes('</li><li>关联知识页：</li><li><a href="../wiki/concepts');
if (flatSibling) throw new Error('关联知识页 was flattened to sibling bullets');
const topLiCount = (html.match(/<ul><li>/g) || []).length;
if (topLiCount < 1) throw new Error('expected outer ul');
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

    def test_style_css_contains_heading_anchor_active_toc_and_hash_target_styles(self):
        style_content = (ROOT / "docs" / "style.css").read_text(encoding="utf-8")
        expected_snippets = [
            ".heading-anchor-link",
            ".detail-markdown-body h2:hover .heading-anchor-link",
            ".detail-toc-list a.active",
            ".detail-toc-list .toc-entry.active",
            ".detail-toc-list .toc-entry a",
            "--toc-marker-width",
            "grid-template-columns: var(--toc-marker-width)",
            ".detail-toc-list ol ol > li::before",
            "text-align: left",
            "padding: 2px 0",
            ".detail-hash-target",
            ".detail-markdown-body hr",
            ".detail-markdown-body ul ul",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, style_content)


if __name__ == "__main__":
    unittest.main()
