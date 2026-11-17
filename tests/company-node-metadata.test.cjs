const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');
const read = file => readFileSync(resolve(__dirname, '../docs', file), 'utf8');

for (const [kind, list] of [['company', 'companies'], ['lab', 'labs']]) {
  test(`${kind} cards show page update dates and Chinese route links independently of release dates`, async () => {
    const dom = new JSDOM(`<div id="${kind}RoadmapPage"></div>`, {
      url: `https://example.test/${kind}.html?id=example`, runScripts: 'outside-only'
    });
    const w = dom.window;
    w.eval(read('graph-tooltip.js'));
    w.eval(read('depth-filters.js'));
    w.eval(read('org-route-order.js'));
    const node = { id: 'wiki/example.md', detail_id: 'example-node', updated: '2026-10-12',
      community: 'community-vla', health_score: 3 };
    w.RNGraphTooltip.setGraphData({ nodes: [node], communities: [{ id: node.community, label: '视觉动作（VLA）社区' }] });
    w.RNLinkGraph = { ensure: async () => ({}) };
    w.RNNodeHealth = { renderMeta: () => '3/3' };
    w.fetch = async () => ({ ok: true, json: async () => ({ [list]: [{
      key: 'example', name: 'Example', nodes: [{ id: 'example-node', title: 'Project', date: '2025-03' }]
    }] }) });
    w.eval(read('company-roadmap.js'));
    await new Promise(setImmediate);
    const card = w.document.querySelector('.co-node-card');
    assert.equal(card.querySelector('time').dateTime, node.updated);
    assert.equal(w.document.querySelector('.co-node-date').textContent, '2025-03');
    const routes = [...card.querySelectorAll('.tt-routes a')];
    assert.ok(routes.length > 0);
    assert.ok(routes.some(a => a.getAttribute('href') === 'graph.html?depth=vla'));
    assert.ok(routes.every(a => !a.textContent.includes('VLA')));
    dom.window.close();
  });
}
