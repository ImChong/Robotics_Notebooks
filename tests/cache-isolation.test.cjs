const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function harness(networkResponse, options = {}) {
  const handlers = {}, removed = [], opened = [], installed = [], fetched = [];
  const ownResponse = { source: 'own cache' };
  const unavailable = options.unavailable || [];
  const cache = {
    match: async () => ownResponse,
    put: async () => {},
    // 逐个可选资源：单个失败只影响自身
    add: async (url) => {
      if (unavailable.includes(url)) throw new Error(`offline: ${url}`);
      installed.push(url);
    },
    // 外壳整批：任一缺失即整批失败，不留下半套外壳
    addAll: async (urls) => {
      for (const url of urls) {
        if (unavailable.includes(url)) throw new Error(`offline: ${url}`);
      }
      installed.push(...urls);
    },
  };
  const context = vm.createContext({
    URL, console,
    self: {
      registration: { scope: 'https://imchong.github.io/Robotics_Notebooks/' },
      location: { origin: 'https://imchong.github.io' },
      addEventListener: (name, fn) => { handlers[name] = fn; },
      clients: { claim() {} }, skipWaiting() {},
    },
    caches: {
      keys: async () => [vm.runInContext('CACHE_NAME', context), 'robotics-wiki-old', 'other-project-v1'],
      delete: async (key) => { removed.push(key); },
      open: async (key) => { opened.push(key); return cache; },
      match: () => { throw new Error('global cache lookup leaks between projects'); },
    },
    fetch: async (request) => {
      fetched.push(request.url);
      if (networkResponse) return networkResponse;
      throw new Error('offline');
    },
  });
  vm.runInContext(readFileSync(resolve(__dirname, '../docs/sw.js'), 'utf8'), context);
  return { handlers, removed, opened, ownResponse, installed, fetched };
}

test('SW upgrade removes only its old caches', async () => {
  const h = harness();
  let completion;
  h.handlers.activate({ waitUntil: (p) => { completion = p; } });
  await completion;
  assert.deepEqual(h.removed, ['robotics-wiki-old']);
});

test('SW ignores other projects, similar prefixes, external hosts and non-GET requests', () => {
  const h = harness();
  for (const [url, method] of [
    ['https://imchong.github.io/Robot_Description_Gallery_Online/main.js', 'GET'],
    ['https://imchong.github.io/Robotics_Notebooks-other/main.js', 'GET'],
    ['https://example.org/Robotics_Notebooks/main.js', 'GET'],
    ['https://imchong.github.io/Robotics_Notebooks/main.js', 'POST'],
  ]) {
    h.handlers.fetch({ request: { url, method }, respondWith() { assert.fail('must not intercept'); } });
  }
});

test('offline reads use the project cache for regular and network-first assets', async () => {
  for (const asset of ['main.js', 'sponsor.js', 'exports/site-catalog-v1.json']) {
    const h = harness();
    let result;
    h.handlers.fetch({
      request: { url: `https://imchong.github.io/Robotics_Notebooks/${asset}`, method: 'GET' },
      respondWith: (p) => { result = p; },
    });
    assert.equal(await result, h.ownResponse);
    assert.ok(h.opened.every((key) => key.startsWith('robotics-wiki-')));
  }
});


test('installation never downloads full-body exports or every page body', async () => {
  const h = harness();
  let completion;
  h.handlers.install({ waitUntil: (p) => { completion = p; } });
  await completion;
  assert.ok(h.installed.length > 0);
  assert.ok(h.installed.every((url) => !/site-data-v1|index-v1|page-content/.test(url)));
});

test('catalog prefers a fresh response while an immutable cached body needs no network', async () => {
  const fresh = { status: 200, clone() { return this; } };
  for (const asset of ['exports/site-catalog-v1.json', `exports/page-content/${'a'.repeat(64)}.json`]) {
    const h = harness(fresh);
    let result;
    h.handlers.fetch({
      request: { url: `https://imchong.github.io/Robotics_Notebooks/${asset}`, method: 'GET' },
      respondWith: (p) => { result = p; },
    });
    const isCatalog = asset.includes('site-catalog');
    assert.equal(await result, isCatalog ? fresh : h.ownResponse);
    assert.equal(h.fetched.length, isCatalog ? 1 : 0);
  }
});

test('installation caches the offline shell and skips the largest data files', async () => {
  const h = harness();
  let completion;
  h.handlers.install({ waitUntil: (p) => { completion = p; } });
  await completion;
  const asset = (name) => `/Robotics_Notebooks/${name}`;
  for (const shell of ['', 'index.html', 'style.css', 'theme-init.js', 'main.js']) {
    assert.ok(h.installed.includes(asset(shell)), `shell asset must stay precached: ${shell || '/'}`);
  }
  for (const heavy of [
    'search-index.json',
    'exports/link-graph.json',
    'exports/wiki-activity.json',
    'exports/hub-rankings.json',
  ]) {
    assert.ok(!h.installed.includes(asset(heavy)), `install must not download ${heavy}`);
  }
});

test('an optional asset failure does not block installation', async () => {
  const h = harness(null, {
    unavailable: ['/Robotics_Notebooks/vendor/d3.min.js', '/Robotics_Notebooks/exports/graph-stats.json'],
  });
  let completion;
  h.handlers.install({ waitUntil: (p) => { completion = p; } });
  await completion;
  assert.ok(h.installed.includes('/Robotics_Notebooks/style.css'));
  assert.ok(h.installed.includes('/Robotics_Notebooks/graph.html'));
  assert.ok(!h.installed.includes('/Robotics_Notebooks/vendor/d3.min.js'));
});

test('a missing shell asset fails installation instead of caching half a shell', async () => {
  const h = harness(null, { unavailable: ['/Robotics_Notebooks/style.css'] });
  let completion;
  h.handlers.install({ waitUntil: (p) => { completion = p; } });
  await assert.rejects(completion, /style\.css/);
  assert.deepEqual(h.installed, []);
});
