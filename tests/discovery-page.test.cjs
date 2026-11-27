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

test('homepage navigation, bilingual dictionary, and SW expose Discovery entry', () => {
  const indexHtml = readDoc('index.html');
  const homeI18n = readDoc('home-i18n.js');
  const swJs = readDoc('sw.js');

  assert.match(indexHtml, /<nav class="main-nav"[\s\S]*href="discovery\.html"[^>]*>研究发现<\/a>/);
  assert.match(indexHtml, /id="mobileNavSheet"[\s\S]*href="discovery\.html"/);
  assert.ok(homeI18n.includes("'研究发现': 'Discovery'"));
  for (const asset of ['discovery.html', 'discovery.js', 'discovery-sources.json']) {
    assert.ok(swJs.includes(`'${asset}'`), `sw.js includes ${asset}`);
  }
});

test('sources load failure displays graceful fallback banner', async () => {
  const { dom, w } = createDiscoveryDom({ failSources: true });
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 0);
  const banner = w.document.querySelector('#discoverySourcesMount .discovery-banner--error');
  assert.ok(banner);
  assert.match(banner.textContent, /加载失败/);
  dom.window.close();
});

test('category filters, telemetry counts, and bilingual toggle work cleanly', async () => {
  const { dom, w, storage } = createDiscoveryDom();
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  assert.equal(w.document.getElementById('discoveryStatSources').textContent, '12');
  assert.equal(w.document.getElementById('discoveryStatPapers').textContent, '6');
  assert.equal(w.document.getElementById('discoveryStatProjects').textContent, '4');
  assert.equal(w.document.getElementById('discoveryStatHubs').textContent, '2');
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 12);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="projects"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 4);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="papers"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 6);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="models"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 1);

  w.document.querySelector('#discoveryCategoryTabs button[data-category="all"]').click();
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 12);

  const langBtn = w.document.getElementById('discoveryLanguageToggle');
  langBtn.click();
  assert.equal(storage.get('robotics-home-language'), 'en');
  assert.equal(w.document.documentElement.lang, 'en');
  assert.equal(w.document.getElementById('discovery-sources-heading').textContent, 'Recommended Websites');
  assert.match(w.document.querySelector('footer.footer').textContent, /Chong Liu/);
  dom.window.close();
});
