// 前端加载与渲染性能优化回归测试：
// 1. star-history.html 头部不阻塞 vendor/plotly.min.js 且延后加载 wiki-activity.json
// 2. company-roadmap.js 不等待全量图谱元数据即先渲染时间线，到达后渐进补全徽标
// 3. graph.html 命中预计算无重叠布局时跳过首屏重叠求解，且社区标签两趟读写不交错重排
// 4. change-log 长日只挂载首屏可见条数并在展开时增量插入
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const readDoc = (file) => readFileSync(resolve(__dirname, '../docs', file), 'utf8');

test('star-history.html does not block head on plotly.min.js and keeps wiki-activity lazy', () => {
  const html = readDoc('star-history.html');
  const headMatch = html.match(/<head>([\s\S]*?)<\/head>/i);
  assert.ok(headMatch, 'star-history.html contains <head>');
  const head = headMatch[1];
  assert.ok(!/<script[^>]+vendor\/plotly\.min\.js/i.test(head), '<head> 不应包含阻塞的 plotly.min.js <script>');
  assert.ok(/<link[^>]+rel="preload"[^>]+href="vendor\/plotly\.min\.js"/i.test(head), '<head> 应预加载 plotly.min.js');
  assert.ok(html.includes('function ensurePlotly()'), '按需动态加载 Plotly');
  assert.ok(html.includes('Promise.all([view.load(), ensurePlotly()])'), '视图数据与 Plotly 并行加载');
  assert.ok(
    html.includes('function addedNodes()') && html.includes('addedNodesPromise || (addedNodesPromise ='),
    'wiki-activity.json 仅在访问节点视图时惰性请求',
  );
});

test('company-roadmap.js renders timeline before RNLinkGraph settles and hydrates badges later', async () => {
  const dom = new JSDOM('<div id="companyRoadmapPage"></div>', {
    url: 'https://example.test/company.html?id=example',
    runScripts: 'outside-only',
  });
  const w = dom.window;
  w.eval(readDoc('graph-tooltip.js'));
  w.eval(readDoc('depth-filters.js'));
  w.eval(readDoc('org-route-order.js'));

  let resolveGraph;
  const deferredGraph = new Promise((r) => { resolveGraph = r; });
  w.RNLinkGraph = { ensure: () => deferredGraph };
  w.RNNodeHealth = { renderMeta: () => '3/3' };
  w.fetch = async () => ({
    ok: true,
    json: async () => ({
      companies: [{
        key: 'example',
        name: 'Example',
        nodes: [{ id: 'example-node', title: 'Project', date: '2025-03' }],
      }],
    }),
  });

  w.eval(readDoc('company-roadmap.js'));
  await new Promise(setImmediate);

  const cardBefore = w.document.querySelector('.co-node-card');
  assert.ok(cardBefore, 'RNLinkGraph 未返回前已完成首屏时间线渲染');
  assert.equal(cardBefore.querySelector('.co-node-metadata'), null, '图谱到达前先不渲染元数据徽标');

  const node = {
    id: 'wiki/example.md',
    detail_id: 'example-node',
    updated: '2026-10-12',
    community: 'community-vla',
    health_score: 3,
  };
  w.RNGraphTooltip.setGraphData({
    nodes: [node],
    communities: [{ id: node.community, label: '视觉动作（VLA）社区' }],
  });
  resolveGraph({});
  await new Promise(setImmediate);
  await new Promise(setImmediate);

  const metadataAfter = w.document.querySelector('.co-node-card .co-node-metadata');
  assert.ok(metadataAfter, 'RNLinkGraph 到达后渐进补全节点徽标');
  assert.equal(metadataAfter.querySelector('time').dateTime, node.updated);
  dom.window.close();
});

test('graph.html skips startup overlap pass when layout_resolved and batches community bbox reads', () => {
  const html = readDoc('graph.html');
  assert.ok(
    html.includes('const precomputedResolved2D = !!(precomputedLayout && graphData.layout_resolved && nodeScale === 1);'),
    '检测预计算已消重叠标记',
  );
  assert.ok(html.includes('if (!precomputedResolved2D) {'), '预计算未消重叠时才在前端重算');
  const bboxIdx = html.indexOf('const measuredBoxes = new Array(pendingBBoxes.length);');
  const readIdx = html.indexOf('pendingBBoxes[i].textEl.getBBox()', bboxIdx);
  const writeIdx = html.indexOf("item.rectEl.setAttribute('x'", readIdx);
  assert.ok(bboxIdx > 0 && readIdx > bboxIdx && writeIdx > readIdx, '社区标签 getBBox 读取集中在属性写入之前');
  assert.ok(html.includes('function ensure2DNodeLabel('), '2D 节点标签按单节点懒挂载，避免点击或缩放时一次创建 8k+ <text>');
  assert.ok(html.includes('incidentEdges2D') && html.includes('is-dim-sidebar'), '悬停与侧栏聚焦采用 O(deg) CSS 类与邻接边索引');
  assert.ok(html.includes('lastAppVh') && html.includes('tooltipMoveRaf'), '视口 CSS 变量写入与 hover tooltip 定位均做缓存与 rAF 节流');
});

test('change-log mounts only initial fold window and appends rows incrementally', () => {
  const mainJs = readDoc('main.js');
  assert.ok(mainJs.includes('var dayMetasByDate = Object.create(null);'), '缓存超量日原始条目');
  assert.ok(mainJs.includes('for (var ii = 0; ii < showCount; ii++)'), '首屏仅挂载折叠阈值条数');
  assert.ok(
    mainJs.includes("listEl.insertAdjacentHTML('beforeend', addHtml)"),
    '点展开时按需增量插入剩余条目',
  );
});
