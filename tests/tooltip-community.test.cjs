// 节点浮窗社区标签：独占一行放在摘要下方，不进顶部类型徽章行；无社区时不渲染。
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const tooltipSource = readFileSync(resolve(__dirname, '../docs/graph-tooltip.js'), 'utf8');

function api() {
  const context = vm.createContext({ window: {} });
  vm.runInContext(tooltipSource, context);
  return context.window.RNGraphTooltip;
}

test('community tag renders on its own row below summary, outside the type badge row', () => {
  const html = api().buildNodeTooltipHtml({
    type: 'method',
    title: 'PPO',
    summary: '近端策略优化',
    communityLabel: '强化学习（Reinforcement Learning, RL）',
    communityColor: '#4e79a7',
    linkHtml: '<a class="tt-link" href="#">打开详情页 →</a>'
  });
  const badgeRow = html.slice(0, html.indexOf('</div>') + 6);
  assert.match(badgeRow, /class="tt-meta-badges"/);
  assert.ok(!badgeRow.includes('tt-community'), 'community tag is not in the type badge row');
  const order = ['tt-meta-badges', 'tt-title', 'tt-summary', 'tt-community', 'tt-link'].map((c) => html.indexOf(c));
  assert.deepEqual(order, order.slice().sort((a, b) => a - b), 'badges → title → summary → community → link');
  assert.match(html, /<div class="tt-community" style="--community-r:78;--community-g:121;--community-b:167">/);
  assert.match(html, /<span>强化学习<\/span>/);
});

test('no community label → no community tag', () => {
  const html = api().buildNodeTooltipHtml({ type: 'concept', title: 'T', communityColor: '#4e79a7' });
  assert.ok(!html.includes('tt-community'));
});
