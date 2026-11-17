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


test('tooltip community and route labels show Chinese only for either parenthesis style', () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(tooltipSource, context);
  const api = context.window.RNGraphTooltip;
  for (const [open, close] of [['（', '）'], ['(', ')']]) {
    const label = '机器人基础模型' + open + 'Robot Foundation Models' + close;
    context.window.RNDepthFilters = {
      depthsForNode: () => ['bfm', 'vla'],
      DEPTH_META: {
        bfm: { emoji: '🗺️', label },
        vla: { emoji: '🗺️', label: '视觉动作' + open + 'VLA' + close }
      }
    };
    const fallback = api.buildNodeTooltipHtml({ communityLabel: label + ' 社区' });
    api.setGraphData({
      nodes: [{ id: 'wiki/example.md', community: 'example' }],
      communities: [{ id: 'example', label: label + ' 社区' }]
    });
    const indexed = api.buildNodeTooltipHtml({ health: { path: 'wiki/example.md' } });
    for (const html of [fallback, indexed]) {
      assert.match(html, /<span>机器人基础模型<\/span>/);
      assert.match(html, /<span>视觉动作<\/span>/);
      assert.ok(!html.includes('Robot Foundation Models'));
      assert.ok(!html.includes('VLA'));
      assert.ok(!html.includes('tt-label-en'));
      assert.match(html, /graph.html\?depth=bfm/);
    }
  }
});
