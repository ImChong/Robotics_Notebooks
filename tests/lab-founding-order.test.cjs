const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const root = resolve(__dirname, '..');
const read = path => readFileSync(resolve(root, path), 'utf8');
const data = JSON.parse(read('docs/lab-roadmaps.json'));
const sorted = [...data.labs].sort((a, b) => a.founded_year - b.founded_year);

test('homepage sorts shuffled labs before choosing the four visible entries', () => {
  const dom = new JSDOM(read('docs/index.html'), { runScripts: 'outside-only' });
  const w = dom.window;
  const links = w.document.getElementById('homeLabLinks');
  [...links.children].reverse().forEach((link, index) => {
    link.hidden = index >= 4;
    link.toggleAttribute('data-lab-extra', index >= 4);
    links.appendChild(link);
  });
  const source = read('docs/main.js');
  const start = source.indexOf("  var labToggle = document.getElementById('homeLabToggle');");
  const end = source.indexOf('  // ── 首页入口卡指针光斑', start);
  w.syncHomeLinksScrollbar = () => {};
  w.animateHomeLinksToggle = (_card, update) => update();
  w.labRoutesCard = w.document.getElementById('home-lab-routes');
  w.eval(source.slice(start, end));
  const keys = () => [...links.children].map(link => link.getAttribute('href').split('=')[1]);
  const visible = () => [...links.children].filter(link => !link.hidden);
  assert.deepEqual(keys(), sorted.map(lab => lab.key));
  assert.deepEqual(visible().map(link => link.getAttribute('href').split('=')[1]), sorted.slice(0, 4).map(lab => lab.key));
  w.document.getElementById('homeLabToggle').click();
  assert.equal(visible().length, sorted.length);
  w.document.getElementById('homeLabToggle').click();
  assert.equal(visible().length, 4);
  assert.deepEqual(keys(), sorted.map(lab => lab.key));
  dom.window.close();
});

async function renderPage(kind, data, query = '') {
  const dom = new JSDOM(`<div id="${kind}"></div>`, {
    url: 'https://example.test/' + query, runScripts: 'outside-only',
  });
  dom.window.fetch = async () => ({ ok: true, json: async () => data });
  dom.window.eval(read('docs/company-roadmap.js'));
  await new Promise(setImmediate);
  return dom;
}

test('lab switcher, default selection and pager use founding order with shuffled data', async () => {
  const shuffled = { ...data, labs: [...data.labs].reverse() };
  const dom = await renderPage('labRoadmapPage', shuffled);
  const d = dom.window.document;
  assert.equal(d.getElementById('coName').textContent, sorted[0].name);
  assert.deepEqual([...d.querySelectorAll('.co-tab')].map(link => link.getAttribute('href')), sorted.map(lab => 'lab.html?id=' + lab.key));
  const pager = d.querySelectorAll('.co-pager a');
  assert.equal(pager[0].getAttribute('href'), 'lab.html?id=' + sorted.at(-1).key);
  assert.equal(pager[2].getAttribute('href'), 'lab.html?id=' + sorted[1].key);
  assert.deepEqual(shuffled.labs.map(lab => lab.key), [...data.labs].reverse().map(lab => lab.key));
  dom.window.close();
});

test('company pages retain their curated order', async () => {
  const companies = [{ key: 'later', name: 'Later', founded_year: 2024 }, { key: 'earlier', name: 'Earlier', founded_year: 2010 }];
  const dom = await renderPage('companyRoadmapPage', { companies });
  assert.equal(dom.window.document.getElementById('coName').textContent, 'Later');
  dom.window.close();
});
