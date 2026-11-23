// Regression checks for Taste Skill Swiss Engineering default UI + Classic fallback switcher.
const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');

const DOCS_DIR = resolve(__dirname, '../docs');
const readDoc = (file) => readFileSync(resolve(DOCS_DIR, file), 'utf8');

const HEADER_PAGES = [
  'index.html',
  'detail.html',
  'roadmap.html',
  'hubs.html',
  'change-log.html',
  'company.html',
  'lab.html',
  'star-history.html',
  'graph.html',
  '404.html',
];

test('all 10 header pages expose #uiStyleToggle, Geist font preloads, and ^⇧L keyboard hint', () => {
  for (const page of HEADER_PAGES) {
    const html = readDoc(page);
    assert.ok(html.includes('id="uiStyleToggle"'), `${page} includes #uiStyleToggle`);
    assert.ok(
      html.includes('href="fonts/geist-latin-wght-normal.woff2"'),
      `${page} preloads Geist sans woff2`,
    );
    assert.ok(
      html.includes('href="fonts/geist-mono-latin-wght-normal.woff2"'),
      `${page} preloads Geist Mono woff2`,
    );
    assert.ok(html.includes('class="kbd-hint"'), `${page} displays ^⇧L shortcut hint`);
  }
});

test('theme-init.js defaults to data-ui="tech" and restores classic/light theme-color accurately', () => {
  const initSrc = readDoc('theme-init.js');

  function runInit({ theme = null, ui = null } = {}) {
    const attrs = {};
    const meta = {
      content: '',
      setAttribute(k, v) {
        if (k === 'content') this.content = String(v);
      },
    };
    const storage = new Map();
    if (theme) storage.set('robotics-notebooks-theme', theme);
    if (ui) storage.set('robotics-notebooks-ui', ui);
    const ctx = vm.createContext({
      document: {
        documentElement: {
          setAttribute(k, v) { attrs[k] = String(v); },
          getAttribute(k) { return attrs[k] || null; },
        },
        querySelector(sel) {
          return sel === 'meta[name="theme-color"]' ? meta : null;
        },
      },
      localStorage: {
        getItem(k) { return storage.has(k) ? storage.get(k) : null; },
      },
    });
    vm.runInContext(initSrc, ctx);
    return { attrs, metaContent: meta.content };
  }

  const defaultDark = runInit();
  assert.equal(defaultDark.attrs['data-theme'], 'dark');
  assert.equal(defaultDark.attrs['data-ui'], 'tech');
  assert.equal(defaultDark.metaContent, '#0a0b0d');

  const techLight = runInit({ theme: 'light', ui: 'tech' });
  assert.equal(techLight.attrs['data-theme'], 'light');
  assert.equal(techLight.attrs['data-ui'], 'tech');
  assert.equal(techLight.metaContent, '#f5f4f0');

  const classicDark = runInit({ theme: 'dark', ui: 'classic' });
  assert.equal(classicDark.attrs['data-theme'], 'dark');
  assert.equal(classicDark.attrs['data-ui'], 'classic');
  assert.equal(classicDark.metaContent, '#191919');

  const classicLight = runInit({ theme: 'light', ui: 'classic' });
  assert.equal(classicLight.attrs['data-theme'], 'light');
  assert.equal(classicLight.attrs['data-ui'], 'classic');
  assert.equal(classicLight.metaContent, '#ffffff');
});

test('index.html embeds #mini-graph-section in 7+5 Hero frame and classic mode relocates it above hubs', () => {
  const indexHtml = readDoc('index.html');
  const mainJs = readDoc('main.js');
  const dom = new JSDOM(indexHtml, {
    url: 'https://imchong.github.io/Robotics_Notebooks/index.html',
    runScripts: 'outside-only',
  });
  const doc = dom.window.document;
  const heroFrame = doc.querySelector('#hero .hero-frame');
  const heroPrimary = doc.querySelector('#hero .hero-primary');
  const miniSection = doc.getElementById('mini-graph-section');
  const hubSection = doc.getElementById('home-hub-section');

  assert.ok(heroFrame && heroPrimary && miniSection && hubSection, 'required hero and hub sections exist');
  assert.ok(heroFrame.contains(heroPrimary), 'heroFrame contains heroPrimary');
  assert.ok(heroFrame.contains(miniSection), 'heroFrame embeds #mini-graph-section in tech mode');
  assert.ok(miniSection.classList.contains('hero-viewport'), '#mini-graph-section carries .hero-viewport');

  const startFn = mainJs.indexOf('  function syncHomeUiLayout(ui)');
  const endFn = mainJs.indexOf('\n  updateThemeToggle();', startFn);
  assert.ok(startFn >= 0 && endFn > startFn, 'syncHomeUiLayout function found in docs/main.js');
  const ctx = vm.createContext({ document: doc });
  vm.runInContext(`${mainJs.slice(startFn, endFn)}\nglobalThis.syncHomeUiLayout = syncHomeUiLayout;`, ctx);

  ctx.syncHomeUiLayout('classic');
  assert.equal(heroFrame.contains(miniSection), false, 'classic mode removes #mini-graph-section from Hero');
  assert.equal(miniSection.nextElementSibling, hubSection, 'classic mode places #mini-graph-section before #home-hub-section');
  assert.equal(miniSection.classList.contains('hero-viewport'), false);

  ctx.syncHomeUiLayout('tech');
  assert.equal(heroFrame.contains(miniSection), true, 'switching back to tech restores #mini-graph-section into Hero');
  assert.equal(miniSection.classList.contains('hero-viewport'), true);
  dom.window.close();
});

test('self-hosted Geist and Geist Mono woff2 assets exist and are registered in style.css', () => {
  const styleCss = readDoc('style.css');
  const fontFiles = [
    'fonts/geist-latin-wght-normal.woff2',
    'fonts/geist-latin-ext-wght-normal.woff2',
    'fonts/geist-mono-latin-wght-normal.woff2',
    'fonts/geist-mono-latin-ext-wght-normal.woff2',
  ];
  for (const rel of fontFiles) {
    assert.ok(existsSync(resolve(DOCS_DIR, rel)), `${rel} exists on disk`);
    assert.ok(styleCss.includes(`url('${rel}')`), `style.css declares @font-face for ${rel}`);
  }
});

test('latest and hub rows keep .node-health inline-flex on mobile and reset negative margin on desktop', () => {
  const styleCss = readDoc('style.css');
  const bandStart = styleCss.indexOf('/* ── 6. Band 03 & 04：密集工程台账（Latest & Hubs）');
  const bandEnd = styleCss.indexOf('/* ── 7. 全站子页与共享组件对齐', bandStart);
  assert.ok(bandStart >= 0 && bandEnd > bandStart, 'Band 03 & 04 CSS block exists');
  const bandCss = styleCss.slice(bandStart, bandEnd);

  assert.equal(
    /display\s*:\s*initial/.test(bandCss),
    false,
    'Band 03 & 04 must not reset .home-latest-row > * / .home-hub-row > * to display: initial on mobile',
  );
  assert.match(
    bandCss,
    /:root:not\(\[data-ui="classic"\]\)\s+\.home-latest-row\s*>\s*\.node-health[\s\S]*?:root:not\(\[data-ui="classic"\]\)\s+\.home-hub-row\s*>\s*\.node-health[\s\S]*?margin:\s*0\s*;/,
    'Desktop .home-latest-row/.home-hub-row > .node-health resets negative margin to 0',
  );
});
