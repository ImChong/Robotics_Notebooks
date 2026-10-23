const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const source = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const start = source.indexOf('  // ── 首页知识节点活跃度热力图');
const end = source.indexOf('  function moduleHref(', start);

for (const reduced of [false, true]) {
  test(`timeline back-to-top reaches document origin (reduced motion: ${reduced})`, () => {
    const dom = new JSDOM('<h1 id="change-log-heading">更新记录</h1><div id="homeLatestWikiModule"></div>', { runScripts: 'outside-only' });
    const { window } = dom;
    const calls = [];
    window.scrollTo = (options) => calls.push(options);
    window.matchMedia = () => ({ matches: reduced });
    window.document.querySelector('h1').scrollIntoView = () => assert.fail('must scroll the document, not the heading');
    window.escapeHtml = (s) => String(s);
    window.latestNodeHref = () => 'detail.html?id=sample';
    window.wikiTypeLabel = () => '实体';
    window.renderActionBadgeCell = () => '';
    window.renderUpdatesItemSuffix = () => '';
    window.homeLatestItemsOf = (stats) => stats.latest_wiki_nodes;
    window.eval(source.slice(start, end));
    window.renderLatestWikiNode({ latest_wiki_nodes: [{ detail_id: 'sample', title: 'Sample', date: '2026-10-23', action: 'added' }] }, null);
    const button = window.document.querySelector('.updates-timeline-back-top');
    assert.ok(button, 'production timeline renders the button');
    button.click();
    assert.equal(calls.length, 1);
    assert.equal(calls[0].top, 0);
    assert.equal(calls[0].behavior, reduced ? 'auto' : 'smooth');
    dom.window.close();
  });
}
