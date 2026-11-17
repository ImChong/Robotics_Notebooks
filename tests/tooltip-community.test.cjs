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
  const divEnd = html.indexOf('</div>');
  assert.ok(divEnd >= 0, 'expected closing badgeRow </div>');
  const badgeRow = html.slice(0, divEnd + 6);
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


test('page-path metadata resolves communities and all matching routes without a mini graph', () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(tooltipSource, context);
  vm.runInContext(readFileSync(resolve(__dirname, '../docs/depth-filters.js'), 'utf8'), context);
  const api = context.window.RNGraphTooltip;
  const node = { id: 'wiki/entities/paper-gr00t.md', detail_id: 'entity-gr00t',
    community: 'community-vla', community_secondary: 'community-imitation-learning' };
  api.setGraphData({ nodes: [node], communities: [
    { id: node.community, label: '视觉语言动作（VLA）社区' },
    { id: node.community_secondary, label: '模仿学习（IL）社区' }
  ] });
  const html = api.buildNodeTooltipHtml({ health: { path: node.id }, title: 'GR00T' });
  assert.match(html, /所属社区/);
  assert.match(html, /视觉语言动作/);
  assert.match(html, /模仿学习/);
  const routes = context.window.RNDepthFilters.depthsForNode(node);
  assert.ok(routes.length > 0);
  for (const key of routes) assert.ok(html.includes('graph.html?depth=' + encodeURIComponent(key)));
  assert.equal(api.resolveMetadataNode({ id: 'entity-gr00t' }), node);
  assert.ok(!api.buildNodeTooltipHtml({ health: { path: 'unknown.md' } }).includes('tt-routes'));
});


test('long bilingual tooltip badges place Chinese above English and keep short labels inline', () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(tooltipSource, context);
  context.window.RNDepthFilters = {
    depthsForNode: () => ['long', 'short'],
    DEPTH_META: {
      long: { emoji: '🗺️', label: '扩散与流匹配策略（Diffusion & Flow Matching）' },
      short: { emoji: '🗺️', label: '视觉动作（VLA）' }
    }
  };
  const html = context.window.RNGraphTooltip.buildNodeTooltipHtml({
    communityLabel: '机器人基础模型（Robot Foundation Models）社区'
  });
  assert.match(html, /tt-label-bilingual"><span>机器人基础模型<\/span><span class="tt-label-en" lang="en">Robot Foundation Models/);
  assert.match(html, /tt-label-bilingual"><span>扩散与流匹配策略<\/span><span class="tt-label-en" lang="en">Diffusion &amp; Flow Matching/);
  assert.match(html, /<span>视觉动作（VLA）<\/span>/);
});
