'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');

function extractBlock(source, signature) {
  const start = source.indexOf(signature);
  assert.notEqual(start, -1, `Expected to find "${signature}"`);
  const braceStart = source.indexOf('{', start);
  assert.notEqual(braceStart, -1, `Expected opening brace for "${signature}"`);
  let depth = 0;
  let inSingle = false;
  let inDouble = false;
  let inTemplate = false;
  let escaped = false;
  for (let i = braceStart; i < source.length; i += 1) {
    const ch = source[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === '\\') {
      escaped = true;
      continue;
    }
    if (!inDouble && !inTemplate && ch === "'") {
      inSingle = !inSingle;
      continue;
    }
    if (!inSingle && !inTemplate && ch === '"') {
      inDouble = !inDouble;
      continue;
    }
    if (!inSingle && !inDouble && ch === '`') {
      inTemplate = !inTemplate;
      continue;
    }
    if (inSingle || inDouble || inTemplate) {
      continue;
    }
    if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(start, i + 1);
      }
    }
  }
  throw new Error(`Unclosed brace block for "${signature}"`);
}

function extractStatement(source, signature) {
  const start = source.indexOf(signature);
  assert.notEqual(start, -1, `Expected to find "${signature}"`);
  const end = source.indexOf(';', start);
  assert.notEqual(end, -1, `Expected terminating semicolon for "${signature}"`);
  return source.slice(start, end + 1);
}

test('wiki-type-labels and graph-tooltip expose reference_page and tech_map_node', () => {
  const labelsSrc = fs.readFileSync(path.join(ROOT, 'docs/wiki-type-labels.js'), 'utf8');
  const tooltipSrc = fs.readFileSync(path.join(ROOT, 'docs/graph-tooltip.js'), 'utf8');

  const ctx = { window: {}, globalThis: {} };
  ctx.globalThis = ctx;
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(labelsSrc, ctx);
  vm.runInContext(tooltipSrc, ctx);

  assert.equal(ctx.RNWikiTypeLabels.ZH.reference_page, '参考');
  assert.equal(ctx.RNWikiTypeLabels.ZH.tech_map_node, '技术节点');
  assert.equal(ctx.RNWikiTypeLabels.EN.reference_page, 'Reference');
  assert.equal(ctx.RNWikiTypeLabels.EN.tech_map_node, 'Tech Node');

  const colors = ctx.RNGraphTooltip.GRAPH_NODE_TYPE_COLOR;
  assert.equal(colors.reference_page, '#818cf8');
  assert.equal(colors.tech_map_node, '#a3e635');
});

test('graph.html supports reference_page and tech_map_node colors, legend order, and detail URLs', () => {
  const html = fs.readFileSync(path.join(ROOT, 'docs/graph.html'), 'utf8');
  const snippet = [
    'const indexItemByKey = new Map();',
    extractBlock(html, 'const TYPE_COLOR ='),
    extractStatement(html, 'const TYPE_LEGEND_ORDER ='),
    extractBlock(html, 'function toDetailId(path)'),
    extractBlock(html, 'function pageUrlForNode(d)'),
    'this.__exports = { TYPE_COLOR, TYPE_LEGEND_ORDER, pageUrlForNode };',
  ].join('\n');

  const ctx = {};
  vm.createContext(ctx);
  vm.runInContext(snippet, ctx);

  const { TYPE_COLOR, TYPE_LEGEND_ORDER, pageUrlForNode } = ctx.__exports;
  assert.equal(TYPE_COLOR.reference_page, '#818cf8');
  assert.equal(TYPE_COLOR.tech_map_node, '#a3e635');
  assert.ok(TYPE_LEGEND_ORDER.includes('reference_page'));
  assert.ok(TYPE_LEGEND_ORDER.includes('tech_map_node'));

  assert.equal(
    pageUrlForNode({ id: 'references/papers/survey-papers.md', detail_id: 'reference-papers-survey-papers', type: 'reference_page' }),
    'detail.html?id=reference-papers-survey-papers',
  );
  assert.equal(
    pageUrlForNode({ id: 'tech-map/modules/control/mpc.md', detail_id: 'tech-node-control-mpc', type: 'tech_map_node' }),
    'detail.html?id=tech-node-control-mpc',
  );
  assert.equal(
    pageUrlForNode({ id: 'roadmap/locomotion-rl-depth.md', detail_id: 'roadmap-locomotion-rl-depth', type: 'roadmap_page' }),
    'roadmap.html?id=roadmap-locomotion-rl-depth',
  );
  assert.equal(
    pageUrlForNode({ id: 'wiki/concepts/reward-shaping.md', type: 'concept' }),
    'detail.html?id=wiki-concepts-reward-shaping',
  );
});

test('main.js handles typed health labels and kmap node types for references and tech-map', () => {
  const mainSrc = fs.readFileSync(path.join(ROOT, 'docs/main.js'), 'utf8');
  // Test the rendered row metadata rather than binding the test to a private helper name.
  const compactRowMeta = extractBlock(mainSrc, 'function buildCompactPageRowMeta(page, detailId, communityIndex)');
  const rowTypeResolver = compactRowMeta.match(/\btype:\s*([\w$]+)\(safePage, detailId, communityIndex\)/)?.[1];
  assert.ok(rowTypeResolver, 'Expected the compact row type to come from the display-type resolver');
  const snippet = [
    'var COMMUNITY_COLORS = [];',
    'var COMMUNITY_META = {};',
    'var CATALOG_META_BY_ID = {};',
    'function cleanTitle(t) { return String(t || ""); }',
    extractStatement(mainSrc, 'var HEALTH_ITEM_LABELS ='),
    extractStatement(mainSrc, 'var ROADMAP_HEALTH_ITEM_LABELS ='),
    extractStatement(mainSrc, 'var REFERENCE_HEALTH_ITEM_LABELS ='),
    extractStatement(mainSrc, 'var TECH_MAP_HEALTH_ITEM_LABELS ='),
    extractStatement(mainSrc, 'var HEALTH_GAP_STATUS ='),
    extractBlock(mainSrc, 'function healthTipLines(meta)'),
    extractBlock(mainSrc, 'var ROADMAP_KMAP_TYPE_COLOR ='),
    extractStatement(mainSrc, 'var ROADMAP_KMAP_PATH_TYPE ='),
    extractBlock(mainSrc, 'function roadmapKmapNodeType(page, id)'),
    extractBlock(mainSrc, `function ${rowTypeResolver}(page, detailId, communityIndex)`),
    'function buildInternalLinkTitle(id) { return id; }',
    compactRowMeta,
    'this.__exports = { healthTipLines, ROADMAP_KMAP_TYPE_COLOR, roadmapKmapNodeType, buildCompactPageRowMeta };',
  ].join('\n');

  const ctx = {
    Math,
    String,
    Object,
    Array,
    Number,
    window: {},
  };
  vm.createContext(ctx);
  vm.runInContext(snippet, ctx);

  const { healthTipLines, ROADMAP_KMAP_TYPE_COLOR, roadmapKmapNodeType, buildCompactPageRowMeta } = ctx.__exports;

  assert.equal(ROADMAP_KMAP_TYPE_COLOR.reference_page, '#818cf8');
  assert.equal(ROADMAP_KMAP_TYPE_COLOR.tech_map_node, '#a3e635');

  const refTip = healthTipLines({ type: 'reference_page', path: 'references/papers/mpc.md', health_score: 2, health_gaps: 4 });
  assert.ok(refTip.some((line) => line.includes('分类覆盖与交叉链接')));

  const techTip = healthTipLines({ type: 'tech_map_node', path: 'tech-map/overview.md', health_score: 2, health_gaps: 4 });
  assert.ok(techTip.some((line) => line.includes('站内入口 ≥3')));

  const wikiTip = healthTipLines({ type: 'concept', path: 'wiki/concepts/a.md', health_score: 2, health_gaps: 4 });
  assert.ok(wikiTip.some((line) => line.includes('原文通读')));

  assert.equal(roadmapKmapNodeType({ path: 'references/papers/mpc.md' }, 'reference-papers-mpc'), 'reference_page');
  assert.equal(roadmapKmapNodeType({ path: 'tech-map/overview.md' }, 'tech-node-overview'), 'tech_map_node');
  assert.equal(
    buildCompactPageRowMeta({ path: 'tech-map/modules/control/mpc.md' }, 'tech-node-control-mpc').type,
    'tech_map_node',
  );
  assert.equal(
    buildCompactPageRowMeta({ path: 'references/repos/simulation.md' }, 'reference-repos-simulation').type,
    'reference_page',
  );
});
