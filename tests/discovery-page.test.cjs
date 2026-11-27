const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const ROOT_DIR = resolve(__dirname, '..');
const DOCS_DIR = resolve(ROOT_DIR, 'docs');
const readDoc = (file) => readFileSync(resolve(DOCS_DIR, file), 'utf8');

const EXPECTED_URLS = [
  'https://huggingface.co/papers',
  'https://www.visionforrobotics.com/robotics-digest/',
  'https://aimap.page/en/learn/papers/',
  'https://www.alphaxiv.org/',
  'https://arxiv.org/list/cs.RO/recent',
  'https://trendingai.today/papers/',
  'https://github.com/trending?since=daily',
  'https://github.com/topics/robotics',
  'https://huggingface.co/models?pipeline_tag=robotics&sort=trending',
  'https://huggingface.co/datasets?task_categories=robotics&sort=trending',
  'https://upstar.dev/t/robotics?p=day',
  'https://upstar.dev/t/robotics',
];

function createDiscoveryDom(options = {}) {
  const html = readDoc('discovery.html');
  const dom = new JSDOM(html, {
    url: 'https://imchong.github.io/Robotics_Notebooks/discovery.html',
    runScripts: 'outside-only',
  });
  const w = dom.window;
  const storage = new Map(Object.entries(options.storage || {}));
  Object.defineProperty(w, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k) => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, String(v)),
    },
  });

  const sourcesJson = options.sourcesJson ?? JSON.parse(readDoc('discovery-sources.json'));
  w.fetch = async (url) => {
    if (String(url).includes('discovery-sources.json')) {
      if (options.failSources) throw new Error('sources fetch failed');
      return { ok: true, status: 200, json: async () => sourcesJson };
    }
    throw new Error(`unexpected fetch: ${url}`);
  };

  w.eval(readDoc('ui-icons.js'));
  w.__rnDiscoveryManualInit = true;
  w.RNDiscovery = null;
  w.eval(readDoc('discovery.js'));

  return { dom, w, storage };
}

test('discovery-sources.json lists the 12 recommended websites across 4 categories without DailyArxiv', () => {
  const sources = JSON.parse(readDoc('discovery-sources.json'));
  assert.equal(sources.schema_version, 1);
  assert.deepEqual(
    sources.categories.map((c) => c.id),
    ['papers', 'projects', 'models', 'datasets'],
  );
  assert.equal(sources.sites.length, 12);

  const urls = sources.sites.map((s) => s.url);
  for (const expectedUrl of EXPECTED_URLS) {
    assert.ok(urls.includes(expectedUrl), `missing recommended site URL: ${expectedUrl}`);
  }
  assert.equal(
    urls.some((u) => /DailyArxiv/i.test(u)),
    false,
    'DailyArxiv should not be listed',
  );

  for (const site of sources.sites) {
    assert.ok(site.name_zh && site.name_en, `${site.id} has bilingual names`);
    assert.ok(site.summary_zh && site.summary_en, `${site.id} has bilingual summaries`);
    assert.ok(site.best_for_zh && site.best_for_en, `${site.id} has best_for descriptions`);
    assert.ok(site.cadence_zh && site.cadence_en, `${site.id} has cadence descriptions`);
  }
});

test('curated picks artifacts and DailyArxiv references are removed from discovery', () => {
  assert.equal(existsSync(resolve(DOCS_DIR, 'discovery-picks.json')), false);
  assert.equal(existsSync(resolve(ROOT_DIR, 'scripts/generate_discovery_picks.py')), false);
  assert.equal(existsSync(resolve(ROOT_DIR, '.github/workflows/discovery-daily.yml')), false);
  const html = readDoc('discovery.html');
  const js = readDoc('discovery.js');
  const swJs = readDoc('sw.js');
  assert.equal(html.includes('discovery-picks'), false);
  assert.equal(js.includes('discovery-picks.json'), false);
  assert.equal(swJs.includes('discovery-picks.json'), false);
});

test('homepage and discovery share unified 2-char header/mobile nav links, i18n, SW, and narrow-screen header CSS', () => {
  const indexDom = new JSDOM(readDoc('index.html'));
  const discoveryDom = new JSDOM(readDoc('discovery.html'));
  const indexDoc = indexDom.window.document;
  const discoveryDoc = discoveryDom.window.document;

  const expectedDesktopNav = ['入口', '搜索', '最新', '枢纽', '图谱', '趋势', '发现'];
  const expectedMobileNav = ['顶部', '入口', '搜索', '最新', '枢纽', '图谱', '趋势', '发现'];

  assert.deepEqual(
    Array.from(indexDoc.querySelectorAll('.site-header .main-nav a')).map((a) => a.textContent.trim()),
    expectedDesktopNav,
  );
  assert.deepEqual(
    Array.from(discoveryDoc.querySelectorAll('.site-header .main-nav a')).map((a) => a.textContent.trim()),
    expectedDesktopNav,
  );
  assert.deepEqual(
    Array.from(indexDoc.querySelectorAll('#mobileNavSheet a [data-nav-label]')).map((el) => el.textContent.trim()),
    expectedMobileNav,
  );
  assert.deepEqual(
    Array.from(discoveryDoc.querySelectorAll('#mobileNavSheet a [data-nav-label]')).map((el) => el.textContent.trim()),
    expectedMobileNav,
  );

  for (const [label, doc] of [['index.html', indexDoc], ['discovery.html', discoveryDoc]]) {
    const sheet = doc.getElementById('mobileNavSheet');
    assert.equal(doc.getElementById('mobileUiStyleToggle'), null, `${label} removes #mobileUiStyleToggle`);
    assert.equal(sheet.textContent.includes('旧版样式'), false, `${label} mobile drawer omits 旧版样式`);
    assert.equal(doc.querySelector('.header-right .github-link.classic-only')?.textContent.trim(), '发现');
  }

  assert.equal(
    indexDoc.querySelector('#hero .hero-social-row a[href="discovery.html"]')?.textContent.trim(),
    '研究发现',
  );
  assert.equal(discoveryDoc.getElementById('discovery-heading')?.textContent.trim(), '研究发现');

  indexDom.window.close();
  discoveryDom.window.close();

  const homeI18n = readDoc('home-i18n.js');
  assert.ok(homeI18n.includes("'研究发现': 'Discovery'"));
  assert.ok(homeI18n.includes("'发现': 'Discovery'"));
  assert.ok(homeI18n.includes("'枢纽': 'Hubs'"));
  assert.ok(homeI18n.includes("'趋势': 'Trends'"));
  assert.ok(homeI18n.includes("'图谱': 'Graph'"));

  const swJs = readDoc('sw.js');
  for (const asset of ['discovery.html', 'discovery.js', 'discovery-sources.json']) {
    assert.ok(swJs.includes(`'${asset}'`), `sw.js includes ${asset}`);
  }

  const styleCss = readDoc('style.css');
  assert.match(styleCss, /:root:not\(\[data-ui="classic"\]\)\s+\.header-start\s*\+\s*\.header-spacer\s*\{\s*display:\s*none;\s*\}/);
  assert.match(styleCss, /\[data-ui="classic"\]\s+\.header-start\s*\{[\s\S]*?min-width:\s*0;/);
  assert.match(styleCss, /\[data-ui="classic"\]\s+\.header-start\s*\+\s*\.header-spacer\s*\{\s*display:\s*none;\s*\}/);
  assert.match(styleCss, /\.discovery-row-anchor\s*\{[\s\S]*?grid-template-columns:\s*34px\s+minmax\(0,\s*1fr\)\s+\d+px\s+\d+px\s+18px;/);
  assert.equal(
    /body\.page-home\s+\.header-right\s+\.version-toggle[\s\S]{0,120}display:\s*none/.test(styleCss),
    false,
    'style.css keeps #uiStyleToggle visible at <= 480px',
  );
});

test('sources load failure displays graceful fallback banner', async () => {
  const { dom, w } = createDiscoveryDom({ failSources: true });
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  assert.equal(w.document.querySelectorAll('.discovery-row').length, 0);
  const banner = w.document.querySelector('#discoverySourcesMount .discovery-banner--error');
  assert.ok(banner);
  assert.match(banner.textContent, /加载失败/);
  dom.window.close();
});

test('ledger rows, hover/focus detail popover, inline drawer, category filters, and bilingual toggle work cleanly', async () => {
  const { dom, w, storage } = createDiscoveryDom();
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  assert.equal(w.document.getElementById('discoveryStatSources').textContent, '12');
  assert.equal(w.document.getElementById('discoveryStatPapers').textContent, '6');
  assert.equal(w.document.getElementById('discoveryStatProjects').textContent, '4');
  assert.equal(w.document.getElementById('discoveryStatHubs').textContent, '2');
  assert.equal(w.document.querySelectorAll('.discovery-row').length, 12);

  const popover = w.document.getElementById('discoveryDetailPopover');
  assert.equal(popover.hidden, true);
  assert.equal(popover.getAttribute('aria-hidden'), 'true');

  const firstAnchor = w.document.querySelector('.discovery-row-anchor[data-site-id="hf-daily-papers"]');
  assert.ok(firstAnchor);
  firstAnchor.dispatchEvent(new w.MouseEvent('mouseover', { bubbles: true, clientX: 260 }));
  assert.equal(popover.hidden, false);
  assert.equal(popover.getAttribute('aria-hidden'), 'false');
  assert.equal(popover.classList.contains('is-visible'), true);
  assert.equal(firstAnchor.getAttribute('aria-describedby'), 'discoveryDetailPopover');
  assert.equal(firstAnchor.closest('.discovery-row').classList.contains('is-hovered'), true);
  assert.ok(popover.textContent.includes('Hugging Face 社区每日论文聚集地'));
  assert.ok(popover.textContent.includes('https://huggingface.co/papers'));
  assert.ok(popover.textContent.includes('每日精选 · 社区投票'));

  w.document.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  assert.equal(popover.hidden, true);
  assert.equal(popover.getAttribute('aria-hidden'), 'true');
  assert.equal(firstAnchor.hasAttribute('aria-describedby'), false);

  firstAnchor.dispatchEvent(new w.FocusEvent('focusin', { bubbles: true }));
  assert.equal(popover.hidden, false);
  w.document.getElementById('discoverySourcesMount').dispatchEvent(
    new w.FocusEvent('focusout', { bubbles: true, relatedTarget: w.document.getElementById('discoveryLanguageToggle') }),
  );
  assert.equal(popover.hidden, true);

  firstAnchor.dispatchEvent(new w.MouseEvent('mouseover', { bubbles: true, clientX: 240 }));
  assert.equal(popover.hidden, false);
  w.document.getElementById('discoverySourcesMount').dispatchEvent(new w.MouseEvent('mouseleave', { bubbles: false }));
  assert.equal(popover.hidden, true);

  const firstInfoBtn = w.document.querySelector('.discovery-row-info-btn[data-site-id="hf-daily-papers"]');
  const firstInlineDrawer = w.document.getElementById('discovery-inline-hf-daily-papers');
  assert.equal(firstInlineDrawer.hidden, true);
  assert.equal(firstInfoBtn.getAttribute('aria-expanded'), 'false');
  firstInfoBtn.click();
  assert.equal(firstInlineDrawer.hidden, false);
  assert.equal(firstInfoBtn.getAttribute('aria-expanded'), 'true');
  firstInfoBtn.click();
  assert.equal(firstInlineDrawer.hidden, true);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="projects"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-row').length, 4);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="papers"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-row').length, 6);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="models"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-row').length, 1);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="all"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-row').length, 12);

  const langBtn = w.document.getElementById('discoveryLanguageToggle');
  langBtn.click();
  assert.equal(storage.get('robotics-home-language'), 'en');
  assert.equal(w.document.documentElement.lang, 'en');
  assert.equal(w.document.getElementById('discovery-sources-heading').textContent, 'Recommended Websites');
  assert.equal(w.document.getElementById('uiStyleToggle').textContent, 'Classic');
  assert.deepEqual(
    Array.from(w.document.querySelectorAll('.site-header .main-nav a')).slice(-4).map((a) => a.textContent.trim()),
    ['Hubs', 'Graph', 'Trends', 'Discovery'],
  );
  assert.match(w.document.querySelector('footer.footer').textContent, /Chong Liu/);
  dom.window.close();
});
