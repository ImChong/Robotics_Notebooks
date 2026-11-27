const assert = require('node:assert/strict');
const { readdirSync, readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const docsDir = resolve(__dirname, '..', 'docs');

test('all docs pages with a site footer default to the Chinese footer', () => {
  const htmlFiles = readdirSync(docsDir).filter((name) => name.endsWith('.html')).sort();
  const checked = [];

  for (const name of htmlFiles) {
    const html = readFileSync(resolve(docsDir, name), 'utf8');
    const dom = new JSDOM(html);
    const { document } = dom.window;
    const footer = document.querySelector('body > footer.footer');
    if (!footer) {
      dom.window.close();
      continue;
    }

    const hasLangToggle = Boolean(document.getElementById('homeLanguageToggle'));
    const text = footer.textContent.replace(/\s+/g, ' ').trim();
    const authorLink = footer.querySelector('a');
    assert.ok(authorLink, `${name} footer should link to the author page`);
    assert.equal(authorLink.getAttribute('href'), 'https://imchong.github.io/index.html');
    assert.equal(authorLink.textContent.trim(), '刘冲', `${name} footer author should default to Chinese`);
    assert.equal(
      text,
      '© 刘冲 2026 · 机器人技术栈地图',
      `${name} (${hasLangToggle ? 'bilingual' : 'zh-only'}) footer should default to Chinese`,
    );
    checked.push(name);
    dom.window.close();
  }

  assert.deepEqual(checked, [
    '404.html',
    'change-log.html',
    'company.html',
    'detail.html',
    'discovery.html',
    'hubs.html',
    'index.html',
    'lab.html',
    'roadmap.html',
    'star-history.html',
  ]);
});
