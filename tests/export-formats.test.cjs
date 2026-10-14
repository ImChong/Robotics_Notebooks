// 紧凑导出格式的还原：link-graph 的下标边、wiki-activity 的 node_index + [detail_id, action]
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');

function fn(name, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `${name} is located`);
  const context = vm.createContext({});
  vm.runInContext(source.slice(start, end), context);
  return context[name];
}

const expandLinkGraph = fn('expandLinkGraph', '  function expandLinkGraph(', '  function ensureLinkGraphData(');
const expandWikiActivity = fn('expandWikiActivity', '  function expandWikiActivity(', '  function renderUpdatesItemSuffix(');

test('index edges are expanded to {source, target} by node order', () => {
  const data = {
    nodes: [{ id: 'wiki/a.md' }, { id: 'wiki/b.md' }, { id: 'wiki/c.md' }],
    edge_format: 'index',
    edges: [[0, 1], [2, 0], [1, 9]],
  };
  const out = expandLinkGraph(data);
  // 结果对象来自 vm realm（原型不同），经 JSON 规整后按值比较
  assert.deepEqual(JSON.parse(JSON.stringify(out.edges)), [
    { source: 'wiki/a.md', target: 'wiki/b.md' },
    { source: 'wiki/c.md', target: 'wiki/a.md' },
  ], 'an out-of-range endpoint is dropped rather than left undefined');
  assert.equal(out.nodes, data.nodes);
});

test('object edges and malformed payloads pass through untouched', () => {
  const legacy = { nodes: [{ id: 'wiki/a.md' }], edges: [{ source: 'wiki/a.md', target: 'wiki/a.md' }] };
  assert.equal(expandLinkGraph(legacy), legacy);
  assert.equal(legacy.edges[0].source, 'wiki/a.md');
  const empty = { nodes: [], edges: [] };
  assert.equal(expandLinkGraph(empty).edges.length, 0);
  assert.equal(expandLinkGraph(null), null);
});

test('compact activity days get their node metadata back from node_index', () => {
  const data = {
    format: 'compact-v1',
    node_index: {
      'entity-a': { label: 'A', type: 'entity', path: 'wiki/entities/a.md', has_repo: true, community_label: '控制 社区' },
      'wiki-concepts-b': { label: 'B', type: 'concept', path: 'wiki/concepts/b.md' },
    },
    days: [
      { date: '2026-10-01', count: 2, added_count: 1, nodes: [['entity-a', 'added'], ['wiki-concepts-b']] },
      { date: '2026-10-02', count: 1, nodes: [['entity-a']] },
    ],
  };
  const out = expandWikiActivity(data);
  assert.deepEqual(JSON.parse(JSON.stringify(out.days[0].nodes[0])), {
    detail_id: 'entity-a', label: 'A', type: 'entity', path: 'wiki/entities/a.md', has_repo: true, community_label: '控制 社区', action: 'added',
  });
  assert.deepEqual(JSON.parse(JSON.stringify(out.days[0].nodes[1])), { detail_id: 'wiki-concepts-b', label: 'B', type: 'concept', path: 'wiki/concepts/b.md' });
  assert.equal('action' in out.days[0].nodes[1], false, 'a reference without action stays without the key');
  assert.equal(out.days[1].nodes[0].label, 'A', 'the same node reused across days');
  assert.equal(out.days[0].added_count, 1);
});

test('legacy activity payloads pass through untouched', () => {
  const legacy = { days: [{ date: '2026-10-01', nodes: [{ detail_id: 'x', action: 'added' }] }] };
  assert.equal(expandWikiActivity(legacy), legacy);
  assert.equal(legacy.days[0].nodes[0].detail_id, 'x');
  assert.equal(expandWikiActivity(null), null);
});

test('the pages that fetch these files on their own run the same expansion', () => {
  const graphPage = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
  assert.ok(graphPage.includes('window.RNExportFormats.expandLinkGraph(graphData)'));
  const trendPage = readFileSync(resolve(__dirname, '../docs/star-history.html'), 'utf8');
  assert.ok(trendPage.includes("expandExport('expandLinkGraph'") && trendPage.includes("expandExport('expandWikiActivity'"));
  assert.ok(source.includes("window.RNExportFormats = { expandLinkGraph: expandLinkGraph, expandWikiActivity: expandWikiActivity };"));
});
