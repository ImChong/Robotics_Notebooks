// Regression: METATRON formulas must use blue KaTeX wrappers instead of inline code.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const root = resolve(__dirname, '..');
const markdown = readFileSync(resolve(root, 'wiki/entities/paper-metatron-promptable-whole-body-execution.md'), 'utf8');
const js = readFileSync(resolve(root, 'docs/main.js'), 'utf8');
const css = readFileSync(resolve(root, 'docs/style.css'), 'utf8');

test('METATRON math expressions produce blue KaTeX inline wrappers', () => {
  const start = js.indexOf('  function unescapeMarkdownEscapes(');
  const end = js.indexOf('  /** 对原样透传的 HTML 片段', start);
  assert.ok(start >= 0 && end > start, 'shared Markdown and math rendering functions exist');
  const context = {
    escapeHtml(value) {
      return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    },
  };
  vm.runInNewContext(js.slice(start, end), context);

  const samples = [
    ['**五元组 Prompt', 1],
    ['**Prior Transformer**', 4],
    ['**Posterior**', 2],
    ['隐变量经 L2', 5],
    ['**步态转向：**', 1],
  ];
  let total = 0;
  for (const [anchor, minimum] of samples) {
    const line = markdown.split('\n').find((item) => item.includes(anchor));
    assert.ok(line, 'missing METATRON formula line: ' + anchor);
    const formulas = [...line.matchAll(/\$([^$\n]+)\$/g)].map((match) => match[1]);
    assert.ok(formulas.length >= minimum, 'missing TeX delimiters: ' + anchor);
    const rendered = context.renderMathBlocks(context.renderInlineMarkdown(line, {}));
    assert.equal((rendered.match(/class="math-inline"/g) || []).length, formulas.length,
      'some formulas did not enter the KaTeX wrapper: ' + anchor);
    for (const formula of formulas) {
      assert.ok(rendered.includes('\\(' + formula + '\\)'), 'lost TeX expression: ' + formula);
    }
    assert.doesNotMatch(rendered, /@@MDMATHTOKEN/, 'leaked Markdown math token');
    total += formulas.length;
  }
  assert.ok(total >= 13, 'representative METATRON formulas were not all checked');
  assert.match(markdown, /\$z_t = \\frac\{\\tilde z_t\}\{\\lVert\\tilde z_t\\rVert_2\}\$/);
  assert.match(markdown, /\$F\(h_t\^q\) \\to \\operatorname\{sg\}\(d_\{t\+1\}\)\$/);
  const tick = String.fromCharCode(96);
  assert.ok(markdown.includes(tick + 'VAE_NOISE_KEY = 0' + tick),
    'literal configuration should remain inline code');
});

test('detail math-inline CSS retains a colored border and fill', () => {
  const style = css.match(/\.detail-markdown-body \.math-inline,\s*\.detail-summary \.math-inline\s*\{([^}]+)\}/);
  assert.ok(style, 'shared formula highlight rule exists');
  assert.match(style[1], /background:\s*rgba\(45,116,218,/);
  assert.match(style[1], /border:\s*1px solid rgba\(45,116,218,/);
});
