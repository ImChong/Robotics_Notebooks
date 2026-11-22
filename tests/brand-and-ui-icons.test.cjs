// 品牌主 Logo（方案 B）、F2 分页同源 Favicon 与 E3 全量 SVG 图标回归测试。
const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const DOCS_DIR = resolve(__dirname, '../docs');
const readDoc = (file) => readFileSync(resolve(DOCS_DIR, file), 'utf8');

const PAGE_FAVICONS = {
  'index.html': 'assets/icons/favicon-home.svg',
  '404.html': 'assets/icons/favicon-home.svg',
  'change-log.html': 'assets/icons/favicon-home.svg',
  'company.html': 'assets/icons/favicon-home.svg',
  'lab.html': 'assets/icons/favicon-home.svg',
  'hubs.html': 'assets/icons/favicon-home.svg',
  'star-history.html': 'assets/icons/favicon-home.svg',
  'graph.html': 'assets/icons/favicon-graph.svg',
  'detail.html': 'assets/icons/favicon-detail.svg',
  'roadmap.html': 'assets/icons/favicon-roadmap.svg',
  'references.html': 'assets/icons/favicon-refs.svg',
};

test('all HTML pages use F2 family SVG favicons, apple-touch-icon, and no emoji data-URI favicons', () => {
  for (const [page, favPath] of Object.entries(PAGE_FAVICONS)) {
    const html = readDoc(page);
    assert.ok(
      html.includes(`href="${favPath}"`),
      `${page} should link to ${favPath}`,
    );
    assert.ok(
      html.includes('href="assets/icons/apple-touch-icon.png"'),
      `${page} should link to apple-touch-icon.png`,
    );
    assert.ok(
      !html.includes('href="data:image/svg+xml,'),
      `${page} should not use legacy inline emoji favicon`,
    );
    assert.ok(
      existsSync(resolve(DOCS_DIR, favPath)),
      `${favPath} must exist on disk`,
    );
  }
  assert.ok(existsSync(resolve(DOCS_DIR, 'assets/icons/apple-touch-icon.png')));
  assert.ok(existsSync(resolve(DOCS_DIR, 'assets/icons/brand-logo.svg')));
});

test('manifest.json uses local Scheme B icon assets without external icons8 links', () => {
  const manifest = JSON.parse(readDoc('manifest.json'));
  assert.ok(Array.isArray(manifest.icons) && manifest.icons.length >= 2);
  for (const entry of manifest.icons) {
    assert.ok(!/^https?:\/\//i.test(entry.src), `manifest icon must be local: ${entry.src}`);
    assert.ok(
      existsSync(resolve(DOCS_DIR, entry.src)),
      `manifest icon must exist on disk: ${entry.src}`,
    );
  }
});

test('all header pages render Scheme B inline brand SVG and monochrome theme SVGs', () => {
  for (const page of Object.keys(PAGE_FAVICONS)) {
    if (page === 'references.html') continue;
    const html = readDoc(page);
    assert.match(html, /<svg class="site-brand-mark"/, `${page} renders .site-brand-mark`);
    assert.match(html, /<span class="icon-sun"><svg class="ui-icon"/, `${page} uses SVG sun icon`);
    assert.match(html, /<span class="icon-moon"><svg class="ui-icon"/, `${page} uses SVG moon icon`);
  }
});

test('RNIcons covers every depth roadmap key and renders star SVG for open-source badge', () => {
  const ctx = vm.createContext({ window: {} });
  ctx.window = ctx;
  vm.runInContext(readDoc('ui-icons.js'), ctx);
  vm.runInContext(readDoc('depth-filters.js'), ctx);

  const { RNIcons, RNDepthFilters } = ctx;
  assert.ok(RNIcons && typeof RNIcons.svg === 'function');
  for (const key of RNDepthFilters.DEPTH_ORDER) {
    const meta = RNDepthFilters.DEPTH_META[key];
    assert.ok(meta && meta.icon, `${key} has icon key in DEPTH_META`);
    assert.equal(meta.emoji, undefined, `${key} no longer stores emoji`);
    assert.ok(RNIcons.has(meta.icon), `RNIcons defines icon "${meta.icon}" for ${key}`);
    assert.match(RNIcons.depthSvg(key, meta.icon), /^<svg class="ui-icon"/);
  }
  const starHtml = RNIcons.repoStarHtml();
  assert.match(starHtml, /class="updates-item-opensource"/);
  assert.match(starHtml, /class="ui-icon ui-icon--star"/);
  assert.ok(!starHtml.includes('⭐'), 'open-source badge uses star SVG rather than emoji');
});

test('zero legacy UI emoji characters remain in updated docs pages and icon modules', () => {
  const uiEmoji = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/u;
  const targets = [
    ...Object.keys(PAGE_FAVICONS),
    'depth-filters.js',
    'graph-tooltip.js',
    'sponsor.js',
    'home-i18n.js',
    'ui-icons.js',
    'manifest.json',
  ];
  for (const file of targets) {
    const lines = readDoc(file).split('\n');
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*')) continue;
      // Exclude non-emoji typographic arrows / close x / play triangle
      const cleaned = lines[i].replace(/[✕▶⌘→←↑↓]/g, '');
      assert.ok(!uiEmoji.test(cleaned), `unexpected UI emoji in ${file}:${i + 1}: ${trimmed.slice(0, 100)}`);
    }
  }
});
