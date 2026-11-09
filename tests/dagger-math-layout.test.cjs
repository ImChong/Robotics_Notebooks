// Regression: DAgger's long LaTeX should remain renderable and scroll within narrow cards.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const root = resolve(__dirname, '..');
const css = readFileSync(resolve(root, 'docs/style.css'), 'utf8');
const js = readFileSync(resolve(root, 'docs/main.js'), 'utf8');
const dagger = readFileSync(resolve(root, 'wiki/methods/dagger.md'), 'utf8');

test('inline math respects its containing column on narrow screens', () => {
  const rule = css.match(/\.detail-markdown-body \.math-inline,\s*\.detail-summary \.math-inline\s*\{([^}]+)\}/);
  assert.ok(rule, 'detail math style is present');
  assert.match(rule[1], /box-sizing:\s*border-box\s*;/);
  assert.match(rule[1], /max-width:\s*100%\s*;/);
  assert.match(rule[1], /overflow-x:\s*auto\s*;/);
});

test('DAgger aggregation and error-bound formulas keep valid KaTeX delimiters', () => {
  const start = js.indexOf('  function unescapeMarkdownEscapes(');
  const end = js.indexOf('  /** 对原样透传的 HTML 片段', start);
  assert.ok(start >= 0 && end > start, 'inline Markdown + math functions found');
  const context = {
    escapeHtml(text) {
      return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    },
  };
  vm.runInNewContext(js.slice(start, end), context);

  const lines = dagger.split('\n');
  const samples = [
    lines.find((line) => line.startsWith('5. 将新样本并入数据集：')),
    lines.find((line) => line.includes('**BC 的二次误差界**')),
    lines.find((line) => line.includes('**DAgger 的线性界**')),
    lines.find((line) => line.includes('**β 的选择**')),
  ];
  assert.ok(samples.every(Boolean), 'all original DAgger examples are available');
  let formulaCount = 0;
  for (const line of samples) {
    const expressions = [...line.matchAll(/\$([^$]+)\$/g)].map((match) => match[1]);
    const rendered = context.renderMathBlocks(context.renderInlineMarkdown(line, {}));
    assert.equal((rendered.match(/class="math-inline"/g) || []).length, expressions.length);
    for (const expr of expressions) {
      assert.ok(rendered.includes('\\(' + expr + '\\)'), 'math survived Markdown processing: ' + expr);
    }
    assert.doesNotMatch(rendered, /@@MDMATHTOKEN/);
    formulaCount += expressions.length;
  }
  assert.equal(formulaCount, 9, 'all nine representative DAgger formulas were checked');
});

test('DAgger beta choice keeps a valid TeX backslash rather than a backspace control character', () => {
  const line = dagger.split('\n').find((item) => item.includes('**β 的选择**'));
  assert.ok(line, 'DAgger beta-choice bullet exists');
  assert.ok(line.includes("\\beta_i=\\mathbb{I}(i=1)"), 'beta choice retains the complete TeX expression');
  assert.equal(dagger.includes(String.fromCharCode(8)), false, 'markdown must not contain a backspace control character');
});
