const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function harness() {
  const handlers = {}, removed = [], opened = [];
  const ownResponse = { source: 'own cache' };
  const cache = { match: async () => ownResponse, put: async () => {} };
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
    fetch: async () => { throw new Error('offline'); },
  });
  vm.runInContext(readFileSync(resolve(__dirname, '../docs/sw.js'), 'utf8'), context);
  return { handlers, removed, opened, ownResponse };
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
  for (const asset of ['main.js', 'sponsor.js']) {
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
