// 详情页多个区块共用一份 link-graph.json：同 URL 只请求并解析一次，失败可重试（编号 8）
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const start = source.indexOf('  var _linkGraphData = null;');
const end = source.indexOf('  function ensureDetailCommunityIndex(');
assert.ok(start >= 0 && end > start, 'shared link-graph loader is located');

function harness(responder) {
  const calls = [];
  const context = vm.createContext({
    fetch: async (url) => {
      calls.push(url);
      return responder(calls.length);
    },
  });
  vm.runInContext(source.slice(start, end), context);
  return { calls, ensure: () => context.ensureLinkGraphData() };
}

test('concurrent and later readers share one request and one parse', async () => {
  let parses = 0;
  const h = harness(() => ({
    ok: true,
    status: 200,
    json: async () => { parses += 1; return { nodes: [{ id: 'wiki/a.md' }] }; },
  }));
  const [first, second] = await Promise.all([h.ensure(), h.ensure()]);
  const third = await h.ensure();
  assert.deepEqual(h.calls, ['exports/link-graph.json']);
  assert.equal(parses, 1);
  assert.equal(first, second);
  assert.equal(second, third);
});

test('a failed load is not cached and the next reader retries', async () => {
  const h = harness((n) => (n === 1
    ? { ok: false, status: 503 }
    : { ok: true, status: 200, json: async () => ({ nodes: [{ id: 'wiki/b.md' }] }) }));
  await assert.rejects(h.ensure(), /503/);
  const data = await h.ensure();
  assert.equal(h.calls.length, 2);
  assert.equal(data.nodes[0].id, 'wiki/b.md');
});

test('detail modules keep no private graph fetch of their own', () => {
  const sites = source.split("fetch('exports/link-graph.json'").length - 1;
  assert.equal(sites, 2, '只剩共享加载器与搜索块自带超时/取消的社区索引');
});
