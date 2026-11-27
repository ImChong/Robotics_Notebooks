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
  'discovery.html',
  'graph.html',
  '404.html',
];

test('all 11 header pages expose #uiStyleToggle, Geist font preloads, and ^⇧L keyboard hint', () => {
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

test('homepage Band 03 renders a single visible "查看全部更新 →" link in both tech and classic modes', () => {
  const indexHtml = readDoc('index.html');
  const mainJs = readDoc('main.js');
  const styleCss = readDoc('style.css');

  const dom = new JSDOM(indexHtml, {
    url: 'https://imchong.github.io/Robotics_Notebooks/index.html',
    runScripts: 'outside-only',
  });
  const { window } = dom;
  const start = mainJs.indexOf('  // ── 首页知识节点活跃度热力图');
  const end = mainJs.indexOf('  function roadmapHref(', start);
  assert.ok(start >= 0 && end > start, 'renderLatestWikiNode slice located');

  window.escapeHtml = (s) => String(s);
  window.latestNodeHref = () => 'detail.html?id=sample';
  window.wikiTypeLabel = () => '实体';
  window.renderActionBadgeCell = () => '<span class="home-latest-row-badge"></span>';
  window.renderUpdatesItemSuffix = () => '';
  window.renderUpdatesItemRepoStar = () => '';
  window.renderUpdatesItemCommunityCat = () => '';
  window.renderHealthBar = () => '<span class="node-health"></span>';
  window.homeLatestItemsOf = (stats) => stats.latest_wiki_nodes;
  window.collectHomeCompactAddedNodes = (items) => items;
  window.eval(mainJs.slice(start, end));

  window.renderLatestWikiNode(
    { latest_wiki_nodes: [{ detail_id: 'sample', label: 'Sample', recency: '2026-11-24', action: 'added' }] },
    null,
  );

  const section = window.document.getElementById('home-latest-wiki-section');
  const headerLink = section.querySelector('.band-aside.tech-only > a.ledger-link[href="change-log.html"]');
  const compactMore = section.querySelector('#homeLatestWikiModule .home-latest-more');
  assert.ok(headerLink, 'tech mode header exposes .ledger-link inside .band-aside.tech-only');
  assert.ok(compactMore, 'compact module renders .home-latest-more');
  assert.ok(compactMore.classList.contains('classic-only'), 'compact .home-latest-more carries .classic-only');
  assert.equal(section.querySelectorAll('a[href="change-log.html"]').length, 2, 'only one tech-only header link and one classic-only footer link exist');

  assert.match(styleCss, /:root:not\(\[data-ui="classic"\]\)\s+\.classic-only\s*\{\s*display:\s*none\s*!important;\s*\}/);
  assert.match(styleCss, /\[data-ui="classic"\]\s+\.tech-only\s*\{\s*display:\s*none\s*!important;\s*\}/);
  assert.match(styleCss, /:root:not\(\[data-ui="classic"\]\)\s+\.home-latest-more\s*\{\s*display:\s*none;\s*\}/);
  dom.window.close();
});

test('clicking mini-graph entry card highlights full-graph buttons twice after border trace and keeps 3D href synced', () => {
  const indexHtml = readDoc('index.html');
  const mainJs = readDoc('main.js');
  const miniGraphJs = readDoc('mini-graph.js');
  const styleCss = readDoc('style.css');

  const dom = new JSDOM(indexHtml, {
    url: 'https://imchong.github.io/Robotics_Notebooks/index.html',
    runScripts: 'outside-only',
  });
  const { window } = dom;

  const traceVarsStart = mainJs.indexOf('  var BORDER_TRACE_MS =');
  const traceVarsEnd = mainJs.indexOf('  function syncHomeLinksScrollbar(', traceVarsStart);
  const traceFnStart = mainJs.indexOf('  function playCardBorderTrace(');
  const traceFnEnd = mainJs.indexOf('  /** 展开 / 收起路线按钮组', traceFnStart);
  const triggerStart = mainJs.indexOf('  // 入口卡「项目查询 / 知识图谱」');
  const triggerEnd = mainJs.indexOf('  if (window.location.hash ===', triggerStart);
  assert.ok(traceVarsStart >= 0 && traceVarsEnd > traceVarsStart);
  assert.ok(traceFnStart >= 0 && traceFnEnd > traceFnStart);
  assert.ok(triggerStart >= 0 && triggerEnd > triggerStart);

  const timers = [];
  window.matchMedia = () => ({ matches: false });
  window.requestAnimationFrame = (cb) => {
    cb();
    return 1;
  };
  window.cancelAnimationFrame = () => {};
  window.setTimeout = (cb, ms) => {
    timers.push({ cb, ms, cleared: false });
    return timers.length;
  };
  window.clearTimeout = (id) => {
    if (timers[id - 1]) timers[id - 1].cleared = true;
  };

  window.eval(`
    var routeToggle = null;
    var prefersReducedMotion = false;
    var prefetchWikiSearchIndex = function () {};
    var focusSearchFromJump = function () {};
    ${mainJs.slice(traceVarsStart, traceVarsEnd)}
    ${mainJs.slice(traceFnStart, traceFnEnd)}
    ${mainJs.slice(triggerStart, triggerEnd)}
    globalThis.__pulseMiniGraphExpandHint = pulseMiniGraphExpandHint;
  `);

  const wrap = window.document.getElementById('mini-graph-wrap');
  const expandBtn = window.document.getElementById('mini-graph-expand');
  const openBtn = window.document.getElementById('mini-graph-open-btn');
  const telemetryLink = window.document.getElementById('mini-graph-telemetry-link');
  const entryCard = window.document.querySelector('a.home-entry-card[data-trace-target="mini-graph-wrap"]');
  assert.ok(wrap && expandBtn && openBtn && telemetryLink && entryCard);

  wrap.scrollIntoView = () => {};
  window.document.getElementById('mini-graph-section').scrollIntoView = () => {};
  entryCard.dispatchEvent(new window.MouseEvent('click', { bubbles: true, cancelable: true }));

  const traceSvg = wrap.querySelector('.home-border-trace-svg');
  assert.ok(traceSvg, 'border trace SVG mounts on #mini-graph-wrap');
  const traceRect = traceSvg.querySelector('rect');
  const animEvent = new window.Event('animationend');
  animEvent.animationName = 'home-border-trace-dash';
  traceRect.dispatchEvent(animEvent);

  assert.equal(expandBtn.classList.contains('is-pulse-hint'), true, 'classic #mini-graph-expand gets .is-pulse-hint');
  assert.equal(openBtn.classList.contains('is-pulse-hint'), true, 'tech #mini-graph-open-btn gets .is-pulse-hint');
  assert.equal(telemetryLink.classList.contains('is-pulse-hint'), true, 'tech #mini-graph-telemetry-link gets .is-pulse-hint');

  const clearTimers = timers.filter((t) => !t.cleared && t.ms === 1800);
  assert.equal(clearTimers.length, 3, 'all three pulse targets schedule 1800ms cleanup');
  clearTimers.forEach((t) => t.cb());
  assert.equal(openBtn.classList.contains('is-pulse-hint'), false, '.is-pulse-hint clears after TOGGLE_HINT_MS');
  assert.equal(telemetryLink.classList.contains('is-pulse-hint'), false);

  window.__pulseMiniGraphExpandHint();
  assert.equal(openBtn.classList.contains('is-pulse-hint'), true, 'hash arrival helper also highlights tech buttons');
  assert.equal(telemetryLink.classList.contains('is-pulse-hint'), true);

  assert.match(styleCss, /:root:not\(\[data-ui="classic"\]\)\s+\.kin-icon-btn\.is-pulse-hint\s*\{[^}]*animation:\s*kin-icon-btn-pulse\s+0\.85s\s+ease\s+2/);
  assert.match(styleCss, /:root:not\(\[data-ui="classic"\]\)\s+\.kin-metric-link\.is-pulse-hint\s*\{[^}]*animation:\s*home-route-toggle-hint\s+0\.85s\s+ease\s+2/);
  assert.match(styleCss, /:root:not\(\[data-ui="classic"\]\)\s+\.kin-metric:has\(\.kin-metric-link\.is-pulse-hint\)\s*\{[^}]*animation:\s*kin-metric-cell-pulse\s+0\.85s\s+ease\s+2/);
  assert.ok(miniGraphJs.includes("document.getElementById('mini-graph-open-btn')"));
  assert.ok(miniGraphJs.includes("document.getElementById('mini-graph-telemetry-link')"));

  dom.window.close();
});

test('graph view floating panels and controls use unified 2px/0px Swiss radius under tech mode', () => {
  const graphHtml = readDoc('graph.html');
  const styleCss = readDoc('style.css');

  const graphTechStart = graphHtml.indexOf('/* ── Taste Skill 瑞士工程风（默认 tech，classic 回退原样式）── */');
  const graphTechEnd = graphHtml.indexOf('</style>', graphTechStart);
  assert.ok(graphTechStart >= 0 && graphTechEnd > graphTechStart, 'graph.html Swiss tech CSS block exists');
  const graphTechCss = graphHtml.slice(graphTechStart, graphTechEnd);

  const requiredGraphSelectors = [
    '.chip.active',
    '.filter-count',
    '.physics-close',
    '.filter-close',
    '.sb-close',
    '.perspective-tier-badge',
    '.physics-action-btn',
    '.filter-recency-mode-btn',
    '.filter-clear',
    '.filter-scroll-chrome',
    '.filter-scroll-thumb',
    '.filter-resize-se::before',
    '.depth-intro-hub-tag',
    '.depth-intro-step-btn',
    '#graph-sidebar .tt-type',
    '.tt-community',
    '.legend-row',
    '.sb-related-list li a',
    'input[type=range]::-webkit-slider-thumb',
    'input[type=range]::-moz-range-thumb',
  ];
  for (const sel of requiredGraphSelectors) {
    assert.ok(graphTechCss.includes(sel), `graph.html tech overrides style ${sel}`);
  }

  const styleTechStart = styleCss.indexOf('/* ── 7. 全站子页与共享组件对齐');
  const styleTechEnd = styleCss.indexOf('/* ── 8. 纵深与主路线正文折叠框', styleTechStart);
  assert.ok(styleTechStart >= 0 && styleTechEnd > styleTechStart);
  const styleTechCss = styleCss.slice(styleTechStart, styleTechEnd);

  const requiredSharedSelectors = [
    '.kb-chat-icon-btn',
    '.kb-chat-starter-btn',
    '.kb-chat-citations__toggle',
    '.kb-chat-mode-badge',
    '.kb-chat-welcome',
    '.kb-chat-msg--user',
    '.kb-chat-msg--assistant',
    '.kb-chat-cite-pill',
    '.kb-chat-md-quote',
    '.graph-hover-tooltip .tt-type',
    '.graph-hover-tooltip .tt-community',
    '.sponsor-qr',
  ];
  for (const sel of requiredSharedSelectors) {
    assert.ok(styleTechCss.includes(sel), `style.css tech Section 7 styles ${sel}`);
  }
});

test('homepage Hero tech actions follow ordered anchors and translate in English mode', () => {
  const indexHtml = readDoc('index.html');
  const homeI18nJs = readDoc('home-i18n.js');

  const dom = new JSDOM(indexHtml, {
    url: 'https://imchong.github.io/Robotics_Notebooks/index.html',
    runScripts: 'outside-only',
  });
  const doc = dom.window.document;
  const links = Array.from(doc.querySelectorAll('#hero .hero-actions.tech-only a'));

  assert.equal(links.length, 6, 'Hero .hero-actions.tech-only renders 6 action links');
  assert.ok(links[0].classList.contains('btn-primary') && links[0].classList.contains('hero-cta'));
  assert.equal(links[0].getAttribute('href'), '#home-start');
  assert.equal(links[0].querySelector('span[data-home-copy]')?.textContent.trim(), '探索入口');

  assert.ok(links[1].classList.contains('link-draw'));
  assert.equal(links[1].getAttribute('href'), '#wiki-search');
  assert.equal(links[1].hasAttribute('data-focus-search'), true);
  assert.equal(links[1].getAttribute('data-trace-target'), 'wiki-search-panel');
  assert.equal(links[1].textContent.trim(), '知识检索');

  assert.ok(links[2].classList.contains('link-draw'));
  assert.equal(links[2].getAttribute('href'), '#home-latest-wiki-section');
  assert.equal(links[2].textContent.trim(), '最新更新');

  assert.ok(links[3].classList.contains('link-draw'));
  assert.equal(links[3].getAttribute('href'), '#home-hub-section');
  assert.equal(links[3].textContent.trim(), '核心枢纽');

  assert.ok(links[4].classList.contains('link-draw'));
  assert.equal(links[4].getAttribute('href'), 'star-history.html');
  assert.equal(links[4].textContent.trim(), '增长趋势');

  assert.ok(links[5].classList.contains('link-draw'));
  assert.equal(links[5].getAttribute('href'), 'discovery.html');
  assert.equal(links[5].textContent.trim(), '研究发现');

  const kickers = [
    doc.querySelector('#home-start .band-kicker'),
    doc.querySelector('#wiki-search .band-kicker'),
    doc.querySelector('#home-latest-wiki-section .band-kicker'),
    doc.querySelector('#home-hub-section .band-kicker'),
  ];
  assert.deepEqual(
    kickers.map((el) => el?.textContent.trim()),
    ['01 / 探索入口', '02 / 知识检索', '03 / 最新更新', '04 / 核心枢纽'],
  );

  dom.window.localStorage.setItem('robotics-home-language', 'en');
  dom.window.eval(homeI18nJs);

  assert.equal(links[0].querySelector('span[data-home-copy]')?.textContent.trim(), 'Explore Gateways');
  assert.equal(links[1].textContent.trim(), 'Knowledge Search');
  assert.equal(links[2].textContent.trim(), 'Latest Updates');
  assert.equal(links[3].textContent.trim(), 'Core Hubs');
  assert.equal(links[4].textContent.trim(), 'Trends');
  assert.equal(links[5].textContent.trim(), 'Discovery');
  assert.deepEqual(
    kickers.map((el) => el?.textContent.trim()),
    ['01 / GATEWAYS', '02 / SEARCH', '03 / LATEST', '04 / HUBS'],
  );

  dom.window.close();
});
