// Execute the production search block with controlled network, timers and DOM adapters.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const vm = require('node:vm');
const flush = () => new Promise(setImmediate);
const response = (data) => ({ ok: true, status: 200, json: async () => data });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
function element() {
  return {
    value: '', innerHTML: '', handlers: {},
    addEventListener(name, fn) { this.handlers[name] = fn; },
    querySelectorAll() { return []; }, focus() {}, scrollIntoView() {},
  };
}
function searchHarness(fetchIndex, fetchCommunity = async () => response({ nodes: [] })) {
  const timers = new Map();
  let timerId = 0;
  const context = vm.createContext({
    AbortController, URLSearchParams, Map, console,
    searchInput: element(), searchResults: element(), communityFilter: element(),
    document: { addEventListener() {}, getElementById() { return null; } },
    window: { location: { hash: '', search: '' }, requestIdleCallback() {} },
    setTimeout(fn, delay) { timers.set(++timerId, { fn, delay }); return timerId; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, options) {
      return url === 'search-index.json' ? fetchIndex(options) : fetchCommunity(options);
    },
    escapeHtml: (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;'),
    isRoadmapPageId: () => false, wikiTypeLabel: () => '概念',
  });
  const source = readFileSync(resolve(__dirname, '../../docs/main.js'), 'utf8');
  const start = source.indexOf('    var _selectedIndex = -1;');
  const end = source.indexOf('\n  }\n\n  function updateRecentVisits', start);
  assert.ok(start >= 0 && end > start, 'search entry block is located');
  vm.runInContext(source.slice(start, end), context);
  context.expire = (delay) => {
    for (const [id, timer] of [...timers]) {
      if (timer.delay === delay) { timers.delete(id); timer.fn(); }
    }
  };
  return context;
}
const indexData = { docs: [
  { id: 'ppo', title: 'PPO', path: 'wiki/ppo.md', tokens: { ppo: 1 } },
  { id: 'slam', title: 'SLAM', path: 'wiki/slam.md', tokens: { slam: 1 } },
] };
module.exports = { searchHarness, indexData, deferred, flush, response };
