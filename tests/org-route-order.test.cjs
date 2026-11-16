const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const read = path => readFileSync(resolve(__dirname, '..', path), 'utf8');
const helper = read('docs/org-route-order.js');
const orgs = [
  { key: 'late', founded_year: 2022, nodes: [{ date: '2001-02' }] },
  { key: 'undated', nodes: [{ date: '' }] },
  { key: 'fallback', nodes: [{ date: '2024-07' }, { date: '' }, { date: '2015-03' }] },
  { key: 'early', founded_year: 2010, nodes: [{ date: '2026-01' }] },
  { key: 'same', founded_year: 2010, nodes: [{ date: '2024-01' }] },
  { key: 'fallback-first', founded_year: null, nodes: [{ date: '2005-06' }] },
].map(org => ({ name: org.key, ...org }));
const expected = ['fallback-first', 'early', 'same', 'fallback', 'late', 'undated'];

test('founding takes precedence, missing years use the earliest dated node, and ties are stable', () => {
  const dom = new JSDOM('', { runScripts: 'outside-only' });
  dom.window.eval(helper);
  const original = JSON.stringify(orgs);
  assert.deepEqual(dom.window.OrgRouteOrder.sort(orgs).map(org => org.key), expected);
  assert.equal(JSON.stringify(orgs), original, 'shared source objects must not be mutated');
  const date = dom.window.OrgRouteOrder.date;
  assert.equal(date({ founded_year: '2019', nodes: [{ date: '2000-01' }] }), '2019-01-01');
  assert.equal(date({ founded_year: '', nodes: [{ date: '2020-02-29' }, { date: '2020-03' }] }), '2020-02-29');
  for (const founded_year of [null, '', 0, 'unknown', '2020-02', Infinity]) {
    assert.equal(date({ founded_year, nodes: [{ date: '2020-02-30' }, { date: '2014-13' }, { date: '2016' }] }), '2016-01-01');
  }
  assert.equal(date({ nodes: [{ date: 'invalid' }] }), '');
  assert.equal(date({}), '');
  dom.window.close();
});

for (const kind of ['company', 'lab']) {
  test(`${kind} homepage preview sorts by the same founding / node fallback before choosing visible links`, () => {
    const dom = new JSDOM(read('docs/index.html'), { runScripts: 'outside-only' });
    const w = dom.window;
    const links = w.document.getElementById(kind === 'company' ? 'homeCompanyLinks' : 'homeLabLinks');
    links.replaceChildren(...orgs.map(org => {
      const link = w.document.createElement('a');
      link.href = `${kind}.html?id=${org.key}`;
      link.textContent = org.name;
      if (org.founded_year) link.dataset.foundedYear = org.founded_year;
      link.dataset.firstNodeDate = org.nodes.map(n => n.date).filter(Boolean).sort()[0] || '';
      return link;
    }));
    w.matchMedia = () => ({ matches: true });
    const source = read('docs/main.js');
    w.eval(helper);
    w.eval(source.slice(source.indexOf('  // ── 首页路线预览'), source.indexOf('  // ── 首页入口卡指针光斑')));
    assert.deepEqual([...links.children].map(link => link.textContent), expected);
    assert.deepEqual([...links.children].filter(link => !link.hidden).map(link => link.textContent), expected.slice(0, 4));
    dom.window.close();
  });

  test(`${kind} default selection, explicit selection and wraparound pager use the same mixed-date order`, async () => {
    for (const query of ['', '?id=fallback']) {
      const dom = new JSDOM(`<div id="${kind === 'company' ? 'company' : 'lab'}RoadmapPage"></div>`, {
        url: 'https://example.test/' + kind + '.html' + query, runScripts: 'outside-only',
      });
      dom.window.fetch = async () => ({ ok: true, json: async () => ({ [kind === 'company' ? 'companies' : 'labs']: orgs }) });
      dom.window.eval(helper);
      dom.window.eval(read('docs/company-roadmap.js'));
      await new Promise(setImmediate);
      const d = dom.window.document;
      assert.deepEqual([...d.querySelectorAll('.co-tab')].map(link => link.textContent.replace(/\d+$/, '')), expected);
      const current = query ? 'fallback' : expected[0];
      const index = expected.indexOf(current);
      assert.equal(d.getElementById('coName').textContent, current);
      const pager = d.querySelectorAll('.co-pager a');
      assert.equal(pager[0].getAttribute('href'), `${kind}.html?id=${expected[(index - 1 + expected.length) % expected.length]}`);
      assert.equal(pager[2].getAttribute('href'), `${kind}.html?id=${expected[(index + 1) % expected.length]}`);
      dom.window.close();
    }
  });
}

test('homepage static dates match the route JSON so sorting does not need extra requests', () => {
  const dom = new JSDOM(read('docs/index.html'));
  for (const [kind, list] of [['company', 'companies'], ['lab', 'labs']]) {
    const data = JSON.parse(read(`docs/${kind}-roadmaps.json`))[list];
    for (const org of data) {
      const link = dom.window.document.querySelector(`a[href="${kind}.html?id=${org.key}"]`);
      assert.ok(link, org.key);
      assert.equal(link.dataset.foundedYear || '', String(org.founded_year || ''), org.key);
      assert.equal(link.dataset.firstNodeDate, org.nodes.map(n => n.date).filter(Boolean).sort()[0] || '', org.key);
    }
  }
  dom.window.close();
});

test('every route surface loads the shared ordering helper and the offline shell includes it', () => {
  for (const page of ['index.html', 'company.html', 'lab.html', 'star-history.html']) {
    const html = read('docs/' + page);
    assert.ok(html.indexOf('src="org-route-order.js"') < html.indexOf('src="main.js"'));
    assert.ok(html.includes('src="org-route-order.js"'));
  }
  assert.match(read('docs/sw.js'), /'org-route-order\.js'/);
});
