// 节点浮窗内的 LaTeX / KaTeX 公式：$...$ 归一为 \(...\)，货币写法保持原样，
// 内容注入后由 RNMath.render 按内容懒加载 KaTeX 渲染。
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const mainSource = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const tooltipSource = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');

function slice(startMarker, endMarker) {
  const start = mainSource.indexOf(startMarker);
  const end = mainSource.indexOf(endMarker);
  assert.ok(start >= 0 && end > start, `located ${startMarker.trim()}`);
  return mainSource.slice(start, end);
}

function harness() {
  const rendered = [];
  const context = vm.createContext({ window: {}, document: { createElement: () => ({}) } });
  vm.runInContext(slice('  function isCurrencyDollarPair(', '  function renderInlineMarkdown('), context);
  vm.runInContext(slice('  function normalizeDollarMath(', '  /** Strip markdown-only escapes'), context);
  vm.runInContext(slice('  function normalizeMathExpr(', '  function renderMathBlocks('), context);
  vm.runInContext(tooltipSource, context);
  context.window.RNMath = {
    normalizeDollarMath: context.normalizeDollarMath,
    render: (el) => rendered.push(el)
  };
  return { api: context.window.RNGraphTooltip, rendered };
}

test('inline $...$ in summary and title becomes KaTeX \\(...\\)', () => {
  const { api } = harness();
  const html = api.buildNodeTooltipHtml({
    type: 'formalization',
    title: '刚体动力学 $O(n)$ 算法',
    summary: 'ABA 与 RNEA 求解正向动力学，复杂度 $O(n)$，并计算 $M(q)$。'
  });
  assert.match(html, /刚体动力学 \\\(O\(n\)\\\) 算法/);
  assert.match(html, /复杂度 \\\(O\(n\)\\\)/);
  assert.match(html, /\\\(M\(q\)\\\)/);
  assert.ok(!html.includes('$O(n)$'), 'no raw $...$ left for KaTeX delimiters');
});

test('currency dollars in summary stay literal', () => {
  const { api } = harness();
  const html = api.buildNodeTooltipHtml({
    type: 'entity',
    title: 'Open Duck Mini',
    summary: 'BOM 目标 <$400，整机 $2000 起。'
  });
  assert.match(html, /&lt;\$400，整机 \$2000 起。/);
  assert.ok(!html.includes('\\('), 'currency pair is not treated as math');
});

test('display delimiters pass through untouched', () => {
  const { api } = harness();
  const html = api.buildNodeTooltipHtml({ title: 'T', summary: '$$a^2 + b^2$$ 与 \\[c^2\\]' });
  assert.match(html, /\$\$a\^2 \+ b\^2\$\$/);
  assert.match(html, /\\\[c\^2\\\]/);
});

test('renderMath delegates to the shared on-demand KaTeX renderer', () => {
  const { api, rendered } = harness();
  const el = { textContent: '\\(O(n)\\)' };
  api.renderMath(el);
  assert.deepEqual(rendered, [el]);
});

test('renderMath is a no-op without an element or without main.js loaded', () => {
  const { api, rendered } = harness();
  api.renderMath(null);
  assert.deepEqual(rendered, []);
});
