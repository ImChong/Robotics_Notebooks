const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const read = path => readFileSync(resolve(__dirname, '..', path), 'utf8');
const source = read('docs/main.js');
const start = source.indexOf('  // ── 首页路线预览');
const end = source.indexOf('  // ── 首页入口卡指针光斑', start);
assert.ok(start >= 0 && end > start);

function home(extraLabs = 0, width = 1280) {
  const dom = new JSDOM(read('docs/index.html'), {
    url: 'https://example.test/', runScripts: 'outside-only',
  });
  const w = dom.window;
  const listeners = [];
  const previewMedia = {
    matches: width <= 860,
    addEventListener(type, listener) { if (type === 'change') listeners.push(listener); },
  };
  w.matchMedia = query => query === '(max-width: 860px)' ? previewMedia : { matches: false };
  dom.resizePreview = nextWidth => {
    previewMedia.matches = nextWidth <= 860;
    listeners.forEach(listener => listener(previewMedia));
  };
  const labs = w.document.getElementById('homeLabLinks');
  for (let i = 0; i < extraLabs; i++) {
    const link = w.document.createElement('a');
    link.href = 'lab.html?id=future-' + i;
    link.textContent = 'Future lab ' + i;
    link.className = 'btn-secondary';
    link.setAttribute('data-founded-year', String(2030 + i));
    labs.prepend(link);
  }
  w.eval(read('docs/org-route-order.js'));
  w.eval(source.slice(start, end));
  return dom;
}

const visible = links => [...links.children].filter(link => !link.hidden);

for (const width of [1280, 860, 390, 320]) test(`route and company previews respect ${width}px and restore their order and scroll`, () => {
  const dom = home(0, width);
  const count = width <= 860 ? 4 : 8;
  for (const kind of ['Route', 'Company']) {
    const d = dom.window.document;
    const links = d.getElementById('home' + kind + 'Links');
    const toggle = d.getElementById('home' + kind + 'Toggle');
    const all = [...links.children];
    const preview = visible(links);
    assert.equal(preview.length, count);
    assert.deepEqual(preview, kind === 'Route' ? all.slice(-count) : all.slice(0, count));
    assert.equal(toggle.hidden, false);
    toggle.click();
    assert.equal(visible(links).length, all.length);
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(links.classList.contains('is-expanded'), true);
    links.scrollTop = 200;
    toggle.click();
    assert.deepEqual(visible(links), preview);
    assert.deepEqual([...links.children], all);
    assert.equal(links.scrollTop, 0);
    assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  }
  dom.window.close();
});

test('mobile labs preview four and can expand and collapse in English', async () => {
  const dom = home(0, 390);
  const d = dom.window.document;
  const labs = d.getElementById('homeLabLinks');
  const toggle = d.getElementById('homeLabToggle');
  const totalLabs = labs.children.length;
  assert.equal(visible(labs).length, 4);
  assert.equal(toggle.hidden, false);
  dom.window.eval(read('docs/home-i18n.js'));
  d.getElementById('homeOrgTabLab').click();
  d.getElementById('homeLanguageToggle').click();
  assert.equal(toggle.textContent, `Show all ${totalLabs} labs ↓`);
  toggle.click();
  await new Promise(setImmediate);
  assert.equal(visible(labs).length, totalLabs);
  assert.equal(toggle.textContent, 'Collapse labs ↑');
  toggle.click();
  await new Promise(setImmediate);
  assert.equal(visible(labs).length, 4);
  assert.equal(toggle.textContent, `Show all ${totalLabs} labs ↓`);
  dom.window.close();
});

test('crossing the mobile breakpoint updates previews while preserving expanded lists', () => {
  const dom = home();
  const d = dom.window.document;
  const routes = d.getElementById('homeRouteLinks');
  const companies = d.getElementById('homeCompanyLinks');
  const labs = d.getElementById('homeLabLinks');
  const labToggle = d.getElementById('homeLabToggle');
  const routePreview = visible(routes);
  const companyPreview = visible(companies);
  dom.resizePreview(860);
  assert.deepEqual(visible(routes), routePreview.slice(-4));
  assert.deepEqual(visible(companies), companyPreview.slice(0, 4));
  assert.equal(visible(labs).length, 4);
  assert.equal(labToggle.hidden, false);
  for (const kind of ['Route', 'Company', 'Lab']) d.getElementById('home' + kind + 'Toggle').click();
  for (const width of [861, 390]) {
    dom.resizePreview(width);
    for (const links of [routes, companies, labs]) {
      assert.equal(visible(links).length, links.children.length);
    }
    assert.equal(labToggle.hidden, width > 860);
  }
  for (const kind of ['Route', 'Company', 'Lab']) d.getElementById('home' + kind + 'Toggle').click();
  assert.equal(visible(routes).length, 4);
  assert.equal(visible(companies).length, 4);
  assert.equal(visible(labs).length, 4);
  dom.resizePreview(861);
  assert.deepEqual(visible(routes), routePreview);
  assert.deepEqual(visible(companies), companyPreview);
  assert.equal(visible(labs).length, Math.min(labs.children.length, 8));
  assert.equal(labToggle.hidden, labs.children.length <= 8);
  dom.window.close();
});

test('short lab lists are fully visible and tab changes retain the company expansion', () => {
  const dom = home();
  const d = dom.window.document;
  const labs = d.getElementById('homeLabLinks');
  assert.equal(visible(labs).length, labs.children.length);
  assert.equal(d.getElementById('homeLabToggle').hidden, true);
  d.getElementById('homeCompanyToggle').click();
  const companyTab = d.getElementById('homeOrgTabCompany');
  companyTab.focus();
  companyTab.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  assert.equal(d.getElementById('home-company-routes').hidden, true);
  assert.equal(d.getElementById('home-lab-routes').hidden, false);
  assert.equal(d.getElementById('homeOrgTabLab').getAttribute('aria-selected'), 'true');
  assert.equal(d.activeElement.id, 'homeOrgTabLab');
  d.activeElement.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
  assert.equal(d.activeElement, companyTab);
  assert.equal(d.getElementById('home-lab-routes').hidden, true);
  assert.equal(d.getElementById('homeCompanyToggle').getAttribute('aria-expanded'), 'true');
  assert.equal(visible(d.getElementById('homeCompanyLinks')).length,
    d.getElementById('homeCompanyLinks').children.length);
  dom.window.close();
});

test('labs beyond eight remain expandable and English controls update after toggling', async () => {
  const dom = home(3);
  const w = dom.window;
  const d = w.document;
  const labs = d.getElementById('homeLabLinks');
  const toggle = d.getElementById('homeLabToggle');
  const totalLabs = labs.children.length;
  assert.ok(totalLabs > 8);
  assert.equal(visible(labs).length, 8);
  assert.equal(labs.lastElementChild.textContent, 'Future lab 2');
  assert.equal(toggle.hidden, false);
  w.eval(read('docs/home-i18n.js'));
  d.getElementById('homeOrgTabLab').click();
  d.getElementById('homeLanguageToggle').click();
  assert.equal(toggle.textContent, `Show all ${totalLabs} labs ↓`);
  toggle.click();
  await new Promise(setImmediate);
  assert.equal(visible(labs).length, totalLabs);
  assert.equal(toggle.textContent, 'Collapse labs ↑');
  toggle.click();
  await new Promise(setImmediate);
  assert.equal(visible(labs).length, 8);
  assert.equal(toggle.textContent, `Show all ${totalLabs} labs ↓`);
  d.getElementById('homeLanguageToggle').click();
  assert.equal(toggle.textContent, `展开全部 ${totalLabs} 个实验室 ↓`);
  dom.window.close();
});
