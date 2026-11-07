// 节点浮窗摘要的行内 Markdown：按 CommonMark 强调规则渲染，A* / RRT* 等字面星号保持原样。
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const context = vm.createContext({ window: {}, document: { createElement: () => ({}) } });
vm.runInContext(readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8'), context);
const api = context.window.RNGraphTooltip;
const md = api.renderInlineMarkdown;

test('single asterisks are emphasis, double asterisks are strong', () => {
  assert.equal(md('BeyondMimic（*Science Robotics* 2026）'), 'BeyondMimic（<em>Science Robotics</em> 2026）');
  assert.equal(md('这是**重点**内容，_斜体_'), '这是<strong>重点</strong>内容，<em>斜体</em>');
  assert.equal(md('*foo **bar***'), '<em>foo <strong>bar</strong></em>');
});

test('literal stars and underscores that cannot open emphasis stay text', () => {
  assert.equal(md('全局 A* 搜折线，RRT* 求最优'), '全局 A* 搜折线，RRT* 求最优');
  assert.equal(md('DexMachina*/HERMES*；rmw_* 与 foo_bar_'), 'DexMachina*/HERMES*；rmw_* 与 foo_bar_');
  assert.equal(md('\\*不是强调\\*'), '*不是强调*');
});

test('code spans, html escaping and math segments are not touched by emphasis', () => {
  assert.equal(md('`a*b*c` 与 <i>'), '<code>a*b*c</code> 与 &lt;i&gt;');
  assert.equal(md('\\(x_1 * y_2\\) 与 *z*'), '\\(x_1 * y_2\\) 与 <em>z</em>');
});

test('tooltip summary is rendered as markdown, title stays plain text', () => {
  const html = api.buildNodeTooltipHtml({ title: 'RRT*', summary: '*Science Robotics* 论文' });
  assert.match(html, /<div class="tt-title">RRT\*<\/div>/);
  assert.match(html, /<div class="tt-summary"><em>Science Robotics<\/em> 论文<\/div>/);
});
