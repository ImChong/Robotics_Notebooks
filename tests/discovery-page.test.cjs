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
  'https://github.com/Ed1sonChen/DailyArxiv',
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

  const clipboardCalls = [];
  Object.defineProperty(w.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: async (text) => {
        clipboardCalls.push(text);
      },
    },
  });

  const fetchCalls = [];
  const sourcesJson = options.sourcesJson ?? JSON.parse(readDoc('discovery-sources.json'));
  const picksJson = options.picksJson ?? JSON.parse(readDoc('discovery-picks.json'));
  w.fetch = async (url, init) => {
    fetchCalls.push({ url, method: (init && init.method) || 'GET' });
    if (String(url).includes('discovery-sources.json')) {
      if (options.failSources) throw new Error('sources fetch failed');
      return { ok: true, status: 200, json: async () => sourcesJson };
    }
    if (String(url).includes('discovery-picks.json')) {
      if (options.failPicks) throw new Error('picks fetch failed');
      return { ok: true, status: 200, json: async () => picksJson };
    }
    throw new Error(`unexpected fetch: ${url}`);
  };

  w.eval(readDoc('ui-icons.js'));
  w.__rnDiscoveryManualInit = true;
  w.RNDiscovery = null;
  w.eval(readDoc('discovery.js'));

  return { dom, w, clipboardCalls, fetchCalls, storage };
}

test('discovery-sources.json lists all 13 verified websites and marks DailyArxiv as non-ranking tool', () => {
  const sources = JSON.parse(readDoc('discovery-sources.json'));
  assert.equal(sources.schema_version, 1);
  assert.ok(Array.isArray(sources.categories) && sources.categories.length === 5);
  assert.ok(Array.isArray(sources.sites) && sources.sites.length === 13);

  const urls = sources.sites.map((s) => s.url);
  for (const expectedUrl of EXPECTED_URLS) {
    assert.ok(urls.includes(expectedUrl), `missing recommended site URL: ${expectedUrl}`);
  }

  for (const site of sources.sites) {
    assert.ok(site.name_zh && site.name_en, `${site.id} has bilingual names`);
    assert.ok(site.summary_zh && site.summary_en, `${site.id} has bilingual summaries`);
    assert.ok(site.best_for_zh && site.best_for_en, `${site.id} has best_for descriptions`);
    assert.ok(site.cadence_zh && site.cadence_en, `${site.id} has cadence descriptions`);
  }

  const dailyArxiv = sources.sites.find((s) => s.url === 'https://github.com/Ed1sonChen/DailyArxiv');
  assert.ok(dailyArxiv, 'DailyArxiv entry exists');
  assert.equal(dailyArxiv.is_supplement_tool, true);
  assert.match(dailyArxiv.summary_zh, /非统一实时榜单/);
  assert.match(dailyArxiv.summary_en, /not a unified real-time ranking/i);
});

test('discovery-picks.json has unique entries, valid dates, and valid KB entity links', () => {
  const picks = JSON.parse(readDoc('discovery-picks.json'));
  assert.equal(picks.schema_version, 1);
  assert.ok(['ok', 'partial', 'stale'].includes(picks.status.state));
  assert.ok(Array.isArray(picks.items) && picks.items.length >= 8 && picks.items.length <= 12);

  const ids = new Set();
  for (const item of picks.items) {
    assert.ok(!ids.has(item.id), `duplicate pick id: ${item.id}`);
    ids.add(item.id);
    assert.ok(['paper', 'project', 'model', 'dataset'].includes(item.kind));
    assert.ok(['published', 'updated'].includes(item.date_type));
    assert.match(item.date, /^\d{4}-\d{2}-\d{2}$/);
    assert.ok(item.title_zh && item.title_en);
    assert.ok(item.contribution_zh && item.contribution_en);
    assert.ok(Object.keys(item.links || {}).length >= 1, `${item.id} has primary links`);
    if (item.kb_node) {
      assert.match(item.kb_node.url, /^detail\.html\?id=entity-/);
      assert.ok(
        existsSync(resolve(ROOT_DIR, item.kb_node.path)),
        `KB node file exists on disk: ${item.kb_node.path}`,
      );
    }
  }
});

test('homepage navigation and bilingual dictionary expose Discovery entry', () => {
  const indexHtml = readDoc('index.html');
  const homeI18n = readDoc('home-i18n.js');
  const swJs = readDoc('sw.js');

  assert.match(indexHtml, /<nav class="main-nav"[\s\S]*href="discovery\.html"[^>]*>研究发现<\/a>/);
  assert.match(indexHtml, /id="mobileNavSheet"[\s\S]*href="discovery\.html"/);
  assert.ok(homeI18n.includes("'研究发现': 'Discovery'"));
  for (const asset of ['discovery.html', 'discovery.js', 'discovery-sources.json', 'discovery-picks.json']) {
    assert.ok(swJs.includes(`'${asset}'`), `sw.js includes ${asset}`);
  }
});

test('recommended websites render even when curated picks fetch fails (decoupled design)', async () => {
  const { dom, w } = createDiscoveryDom({ failPicks: true });
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  const siteCards = w.document.querySelectorAll('.discovery-site-card');
  assert.equal(siteCards.length, 13, 'all 13 websites render despite picks fetch failure');

  const statusText = w.document.getElementById('discoveryStatusBar').textContent;
  assert.match(statusText, /推荐网站导航/);
  assert.equal(w.document.querySelectorAll('.discovery-pick-card').length, 0);
  dom.window.close();
});

test('stale picks snapshot displays stale notice while keeping preserved items', async () => {
  const stalePicks = JSON.parse(readDoc('discovery-picks.json'));
  stalePicks.status = {
    state: 'stale',
    last_attempt_at: '2026-11-28T09:00:00+08:00',
    sources_checked: ['hf-daily-papers'],
    sources_failed: ['hf-daily-papers'],
    message_zh: '本次精选更新失败，已保留上次有效精选内容。',
    message_en: 'Refresh failed; showing last valid snapshot.',
  };

  const { dom, w } = createDiscoveryDom({ picksJson: stalePicks });
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  const statusBar = w.document.getElementById('discoveryStatusBar');
  assert.match(statusBar.textContent, /保留上次有效内容/);
  assert.match(statusBar.textContent, /2026-11-28 09:00 UTC\+8/);
  assert.equal(
    w.document.querySelectorAll('.discovery-pick-card').length,
    stalePicks.items.length,
    'preserved picks remain rendered',
  );
  dom.window.close();
});

test('category filters, bilingual toggle, and plain-text checklist copy work without creating PRs', async () => {
  const { dom, w, clipboardCalls, fetchCalls, storage } = createDiscoveryDom();
  const app = w.RNDiscovery.initDiscoveryPage(w.document, w);
  await app.ready;

  // Filter websites by category: projects has 4 sources
  const projectsTab = w.document.querySelector('#discoveryCategoryTabs button[data-category="projects"]');
  projectsTab.click();
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 4);

  const allTab = w.document.querySelector('#discoveryCategoryTabs button[data-category="all"]');
  allTab.click();
  assert.equal(w.document.querySelectorAll('.discovery-site-card').length, 13);

  // Copy plain-text checklist
  const copyBtn = w.document.getElementById('copyDiscoveryListBtn');
  copyBtn.click();
  await new Promise(setImmediate);
  assert.equal(clipboardCalls.length, 1);
  assert.match(clipboardCalls[0], /Robotics Discovery 最新精选清单/);
  assert.match(clipboardCalls[0], /https:\/\/arxiv\.org\/abs\/|https:\/\/github\.com\//);
  assert.match(w.document.getElementById('copyDiscoveryStatus').textContent, /已复制/);

  // Ensure no POST / PR creation calls occurred
  assert.ok(fetchCalls.every((c) => c.method === 'GET'));

  // Toggle English language & verify persistence
  const langBtn = w.document.getElementById('discoveryLanguageToggle');
  langBtn.click();
  assert.equal(storage.get('robotics-home-language'), 'en');
  assert.equal(w.document.documentElement.lang, 'en');
  assert.equal(w.document.getElementById('discovery-sources-heading').textContent, 'Recommended Websites');
  assert.match(w.document.querySelector('footer.footer').textContent, /Chong Liu/);

  // Copy again in English
  copyBtn.click();
  await new Promise(setImmediate);
  assert.equal(clipboardCalls.length, 2);
  assert.match(clipboardCalls[1], /Robotics Discovery Curated Picks/);
  dom.window.close();
});
