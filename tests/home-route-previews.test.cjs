const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const read = path => readFileSync(resolve(__dirname, '..', path), 'utf8');
const source = read('docs/main.js');
const start = source.indexOf('  // ── 首页「更多路线」折叠');
const end = source.indexOf('  // ── 首页入口卡指针光斑', start);
assert.ok(start >= 0 && end > start);

function home(extraLabs = 0) {
  const dom = new JSDOM(read('docs/index.html'), {
    url: 'https://example.test/', runScripts: 'outside-only',
  });
  const w = dom.window;
  const labs = w.document.getElementById('homeLabLinks');
  for (let i = 0; i < extraLabs; i++) {
    const link = w.document.createElement('a');
    link.href = 'lab.html?id=future-' + i;
    link.textContent = 'Future lab ' + i;
    link.className = 'btn-secondary';
    link.setAttribute('data-founded-year', String(2030 + i));
    labs.prepend(link);
  }
  w.eval(source.slice(start, end));
  return dom;
}

const visible = links => [...links.children].filter(link => !link.hidden);

test('route and company previews show eight, expand in place and restore their order and scroll', () => {
  const dom = home();
  for (const kind of ['Route', 'Company']) {
    const d = dom.window.document;
    const links = d.getElementById('home' + kind + 'Links');
    const toggle = d.getElementById('home' + kind + 'Toggle');
    const all = [...links.children];
    const preview = visible(links);
    assert.equal(preview.length, 8);
    assert.deepEqual(preview, kind === 'Route' ? all.slice(-8) : all.slice(0, 8));
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
  assert.equal(labs.children.length, 10);
  assert.equal(visible(labs).length, 8);
  assert.equal(labs.lastElementChild.textContent, 'Future lab 2');
  assert.equal(toggle.hidden, false);
  w.eval(read('docs/home-i18n.js'));
  d.getElementById('homeOrgTabLab').click();
  d.getElementById('homeLanguageToggle').click();
  assert.equal(toggle.textContent, 'Show all 10 labs ↓');
  toggle.click();
  await new Promise(setImmediate);
  assert.equal(visible(labs).length, 10);
  assert.equal(toggle.textContent, 'Collapse labs ↑');
  toggle.click();
  await new Promise(setImmediate);
  assert.equal(visible(labs).length, 8);
  assert.equal(toggle.textContent, 'Show all 10 labs ↓');
  d.getElementById('homeLanguageToggle').click();
  assert.equal(toggle.textContent, '展开全部 10 个实验室 ↓');
  dom.window.close();
});
