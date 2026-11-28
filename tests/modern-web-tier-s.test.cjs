// Tier S modern-web-guidance 回归测试（Issue #4311）：
// #1 IME 组字回车防误提交/误跳转（kb-chat.js / main.js）
// #2 sponsor.js async+low、非图谱页底部脚本 defer+high、index 尾随低优先级、SW/预取 priority:'low'
// #3 MPA 悬停预取 Speculation Rules API（支持检测、规则边界、幂等与降级）
// #4 屏下重区块 content-visibility: auto 与 contain-intrinsic-size（排除首屏 mini-graph）
// #5 Mermaid/Plotly CDN SRI（真实 SHA-384 摘要校验）、no-referrer 与全站 Referrer-Policy meta
// #6 AI 问答输入框 .kb-chat-input field-sizing: content 与 height: auto
// #7 @view-transition { navigation: auto; } 的 prefers-reduced-motion 与主题切换隔离
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { existsSync, readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');
const { searchHarness, indexData, flush, response } = require('./helpers/search-harness.cjs');

const readDoc = (file) => readFileSync(resolve(__dirname, '../docs', file), 'utf8');
const mainSource = readDoc('main.js');
const chatSource = readDoc('kb-chat.js');
const styleSource = readDoc('style.css');
const starHistoryHtml = readDoc('star-history.html');

const ALL_HTML_PAGES = [
  '404.html',
  'change-log.html',
  'company.html',
  'detail.html',
  'discovery.html',
  'graph.html',
  'hubs.html',
  'index.html',
  'lab.html',
  'references.html',
  'roadmap.html',
  'star-history.html',
];
const SPONSOR_PAGES = ALL_HTML_PAGES.filter((file) => file !== 'references.html');
const DEFERRED_CONTENT_PAGES = [
  '404.html',
  'change-log.html',
  'company.html',
  'detail.html',
  'discovery.html',
  'hubs.html',
  'index.html',
  'lab.html',
  'roadmap.html',
];

function parseExternalScripts(html) {
  const tags = [];
  const re = /<script\b([^>]*)>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const attrs = m[1];
    const srcMatch = attrs.match(/\bsrc\s*=\s*"([^"]+)"/i);
    if (!srcMatch) continue;
    const fpMatch = attrs.match(/\bfetchpriority\s*=\s*"([^"]+)"/i);
    tags.push({
      src: srcMatch[1],
      hasDefer: /\bdefer\b/i.test(attrs),
      hasAsync: /\basync\b/i.test(attrs),
      fetchPriority: fpMatch ? fpMatch[1] : null,
      raw: m[0],
    });
  }
  return tags;
}

function makeKeyEvent(WindowCtor, { key = 'Enter', shiftKey = false, isComposing = false, keyCode = 13 } = {}) {
  const ev = new WindowCtor.KeyboardEvent('keydown', {
    key,
    shiftKey,
    isComposing,
    bubbles: true,
    cancelable: true,
  });
  Object.defineProperty(ev, 'keyCode', { value: keyCode });
  return ev;
}

test('#1 kb-chat.js #kbChatInput blocks IME Enter (isComposing / keyCode 229) and submits on normal Enter', () => {
  const dom = new JSDOM('<!DOCTYPE html><html lang="zh-CN"><head></head><body><main></main><footer class="footer">Footer</footer></body></html>', {
    url: 'https://imchong.github.io/Robotics_Notebooks/index.html',
    runScripts: 'outside-only',
  });
  const w = dom.window;
  let fetchCount = 0;
  w.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  w.matchMedia = () => ({ matches: false });
  w.RN_KB_CHAT_CONFIG = { endpoint: '', turnstileSiteKey: '' };
  w.fetch = async () => {
    fetchCount += 1;
    return { ok: false, status: 404, json: async () => ({}) };
  };

  w.eval(chatSource);
  w.RNKbChat.open();
  const input = w.document.getElementById('kbChatInput');
  assert.ok(input, '#kbChatInput mounted');

  input.value = '你好';
  const composingEv = makeKeyEvent(w, { key: 'Enter', isComposing: true, keyCode: 13 });
  input.dispatchEvent(composingEv);
  assert.equal(composingEv.defaultPrevented, true, 'preventDefault runs before early return');
  assert.equal(w.document.querySelectorAll('.kb-chat-msg--user').length, 0, 'isComposing Enter must not submit');
  assert.equal(input.value, '你好');

  const vkProcessEv = makeKeyEvent(w, { key: 'Enter', isComposing: false, keyCode: 229 });
  input.dispatchEvent(vkProcessEv);
  assert.equal(vkProcessEv.defaultPrevented, true);
  assert.equal(w.document.querySelectorAll('.kb-chat-msg--user').length, 0, 'keyCode 229 Enter must not submit');
  assert.equal(input.value, '你好');

  const shiftEv = makeKeyEvent(w, { key: 'Enter', shiftKey: true, isComposing: false, keyCode: 13 });
  input.dispatchEvent(shiftEv);
  assert.equal(shiftEv.defaultPrevented, false, 'Shift+Enter keeps newline behavior');
  assert.equal(w.document.querySelectorAll('.kb-chat-msg--user').length, 0);

  const normalEv = makeKeyEvent(w, { key: 'Enter', isComposing: false, keyCode: 13 });
  input.dispatchEvent(normalEv);
  assert.equal(normalEv.defaultPrevented, true);
  assert.equal(w.document.querySelectorAll('.kb-chat-msg--user').length, 1, 'normal Enter submits question');
  assert.equal(input.value, '');
  assert.equal(fetchCount, 0);

  dom.window.close();
});

test('#1 main.js #wikiSearchInput blocks IME Enter (isComposing / keyCode 229) and navigates on normal Enter', () => {
  const ctx = searchHarness(async () => response(indexData));
  ctx.searchResults.querySelectorAll = () => [{
    getAttribute: (name) => (name === 'data-result-url' ? 'detail.html?id=ppo' : null),
  }];

  let prevented = 0;
  ctx.searchInput.handlers.keydown({
    key: 'Enter',
    isComposing: true,
    keyCode: 13,
    preventDefault() { prevented += 1; },
  });
  assert.equal(prevented, 1);
  assert.equal(ctx.window.location.href, undefined, 'isComposing Enter must not navigate');

  ctx.searchInput.handlers.keydown({
    key: 'Enter',
    isComposing: false,
    keyCode: 229,
    preventDefault() { prevented += 1; },
  });
  assert.equal(prevented, 2);
  assert.equal(ctx.window.location.href, undefined, 'keyCode 229 Enter must not navigate');

  ctx.searchInput.handlers.keydown({
    key: 'Enter',
    isComposing: false,
    keyCode: 13,
    preventDefault() { prevented += 1; },
  });
  assert.equal(prevented, 3);
  assert.equal(ctx.window.location.href, 'detail.html?id=ppo', 'normal Enter opens first search card');
});

test('#2 HTML scripts avoid head-of-line priority inversions while deferring non-graph pages', () => {
  for (const file of SPONSOR_PAGES) {
    const scripts = parseExternalScripts(readDoc(file));
    const sponsor = scripts.find((s) => s.src.startsWith('sponsor.js'));
    assert.ok(sponsor, `${file} includes sponsor.js`);
    assert.equal(sponsor.hasAsync, true, `${file} sponsor.js uses async`);
    assert.equal(sponsor.hasDefer, false, `${file} sponsor.js must not occupy defer queue head`);
    assert.equal(sponsor.fetchPriority, 'low', `${file} sponsor.js has fetchpriority="low"`);
  }

  for (const file of DEFERRED_CONTENT_PAGES) {
    const scripts = parseExternalScripts(readDoc(file));
    const bottomScripts = scripts.filter(
      (s) => s.src !== 'theme-init.js' && !s.src.startsWith('sponsor.js'),
    );
    assert.ok(bottomScripts.length > 0, `${file} has bottom external scripts`);
    for (const s of bottomScripts) {
      assert.equal(s.hasDefer, true, `${file} ${s.src} uses defer`);
    }
    const mainIdx = bottomScripts.findIndex((s) => s.src === 'main.js');
    assert.ok(mainIdx >= 0, `${file} includes main.js`);
    assert.equal(bottomScripts[mainIdx].fetchPriority, 'high', `${file} main.js has fetchpriority="high"`);

    const orgIdx = bottomScripts.findIndex((s) => s.src === 'org-route-order.js');
    if (orgIdx >= 0) {
      assert.ok(orgIdx < mainIdx, `${file} org-route-order.js runs before main.js`);
    }
    for (let i = 0; i < mainIdx; i += 1) {
      assert.notEqual(
        bottomScripts[i].fetchPriority,
        'low',
        `${file} ${bottomScripts[i].src} precedes main.js and must not carry fetchpriority="low"`,
      );
    }
  }

  const indexScripts = parseExternalScripts(readDoc('index.html'));
  const indexMainPos = indexScripts.findIndex((s) => s.src === 'main.js');
  for (const trailingSrc of ['vendor/d3.min.js', 'graph-perspective.js', 'mini-graph.js']) {
    const pos = indexScripts.findIndex((s) => s.src === trailingSrc);
    assert.ok(pos > indexMainPos, `index.html ${trailingSrc} loads after main.js`);
    assert.equal(indexScripts[pos].hasDefer, true);
    assert.equal(indexScripts[pos].fetchPriority, 'low');
  }

  const graphScripts = parseExternalScripts(readDoc('graph.html'));
  for (const syncSrc of ['graph-perspective.js', 'graph-3d.js']) {
    const tag = graphScripts.find((s) => s.src === syncSrc);
    assert.ok(tag, `graph.html includes ${syncSrc}`);
    assert.equal(tag.hasDefer, false);
    assert.equal(tag.hasAsync, false);
    assert.equal(tag.fetchPriority, null, `graph.html ${syncSrc} avoids sync priority inversion`);
  }
});

test('#2 sw.js SWR background refresh and main.js silent prefetches pass priority: "low"', async () => {
  const swFetchCalls = [];
  const handlers = {};
  const cachedResponse = { source: 'cached' };
  const fakeCache = {
    match: async () => cachedResponse,
    put: async () => {},
  };
  const swCtx = vm.createContext({
    URL,
    console,
    self: {
      registration: { scope: 'https://imchong.github.io/Robotics_Notebooks/' },
      location: { origin: 'https://imchong.github.io' },
      addEventListener: (name, fn) => { handlers[name] = fn; },
      clients: { claim() {} },
      skipWaiting() {},
    },
    caches: {
      open: async () => fakeCache,
      keys: async () => [],
      delete: async () => true,
    },
    fetch: async (req, init) => {
      swFetchCalls.push({ url: req.url, init });
      return { status: 200, clone() { return this; } };
    },
  });
  vm.runInContext(readDoc('sw.js'), swCtx);

  let respondedPromise;
  handlers.fetch({
    request: { url: 'https://imchong.github.io/Robotics_Notebooks/main.js', method: 'GET' },
    respondWith(p) { respondedPromise = p; },
  });
  const swResult = await respondedPromise;
  assert.equal(swResult, cachedResponse);
  await flush();
  assert.equal(swFetchCalls.length, 1);
  assert.equal(swFetchCalls[0].init?.priority, 'low', 'SWR revalidation uses priority: "low"');

  // main.js silent search prefetch passes priority: 'low', active search does not
  const searchFetches = [];
  const ctx = searchHarness(async (opts) => {
    searchFetches.push(opts || {});
    return response(indexData);
  });
  ctx.searchInput.handlers.focus();
  await flush();
  assert.equal(searchFetches.length, 1);
  assert.equal(searchFetches[0].priority, 'low', 'prefetchWikiSearchIndex uses priority: "low"');

  const activeFetches = [];
  const activeCtx = searchHarness(async (opts) => {
    activeFetches.push(opts || {});
    return response(indexData);
  });
  activeCtx.searchInput.value = 'ppo';
  activeCtx.searchInput.handlers.focus();
  await flush();
  assert.equal(activeFetches.length, 1);
  assert.equal(activeFetches[0].priority, undefined, 'active search keeps default fetch priority');

  // main.js ensureLinkGraphSummaries fetches summaries with priority: 'low'
  const linkGraphStart = mainSource.indexOf('  var _linkGraphData = null;');
  const linkGraphEnd = mainSource.indexOf('  function ensureDetailCommunityIndex()', linkGraphStart);
  assert.ok(linkGraphStart >= 0 && linkGraphEnd > linkGraphStart, 'link graph loader block located');
  const summaryFetches = [];
  const lgCtx = vm.createContext({
    Promise,
    Array,
    window: {},
    fetch: async (url, init) => {
      summaryFetches.push({ url, init });
      if (url === 'exports/link-graph.json') {
        return { ok: true, status: 200, json: async () => ({ nodes: [{ id: 'a' }], edges: [] }) };
      }
      return { ok: true, status: 200, json: async () => ['summary-a'] };
    },
  });
  vm.runInContext(mainSource.slice(linkGraphStart, linkGraphEnd), lgCtx);
  const data = await lgCtx.ensureLinkGraphSummaries();
  assert.equal(data.nodes[0].summary, 'summary-a');
  const sumCall = summaryFetches.find((c) => c.url === 'exports/link-graph-summaries.json');
  assert.equal(sumCall?.init?.priority, 'low', 'link-graph-summaries fetch uses priority: "low"');
});

test('#3 installSpeculationRules progressively enhances supported MPAs, excludes heavy canvases, and is idempotent', () => {
  const specStart = mainSource.indexOf('/* ── 跨页悬停预取');
  assert.ok(specStart >= 0, 'speculation rules block present at end of main.js');
  const specBlock = mainSource.slice(specStart);

  // Supported browser -> injects exactly one prefetch moderate rule
  const domSupported = new JSDOM('<!DOCTYPE html><html><head></head><body></body></html>', {
    url: 'https://example.test/index.html',
    runScripts: 'outside-only',
  });
  domSupported.window.HTMLScriptElement.supports = (type) => type === 'speculationrules';
  domSupported.window.eval(specBlock);

  const injected = domSupported.window.document.querySelectorAll('script[type="speculationrules"][data-rn-speculation="1"]');
  assert.equal(injected.length, 1);
  const parsed = JSON.parse(injected[0].textContent);
  assert.equal(parsed.prefetch.length, 1);
  assert.equal(parsed.prefetch[0].tag, 'rn-mpa-prefetch');
  assert.equal(parsed.prefetch[0].eagerness, 'moderate');
  const andClauses = parsed.prefetch[0].where.and;
  const includeClause = andClauses.find((c) => typeof c.selector_matches === 'string');
  const notClause = andClauses.find((c) => c.not)?.not;
  for (const page of ['detail.html', 'roadmap.html', 'company.html', 'lab.html', 'hubs.html', 'change-log.html', 'discovery.html']) {
    assert.ok(includeClause.selector_matches.includes(page), `speculation includes ${page}`);
  }
  for (const excluded of ['graph.html', 'star-history.html', "[target='_blank']", "[rel~='nofollow']", '.do-not-prefetch']) {
    assert.ok(notClause.selector_matches.includes(excluded), `speculation excludes ${excluded}`);
  }
  assert.equal(domSupported.window.RNInstallSpeculationRules(), false, 'second invocation is idempotent');
  assert.equal(domSupported.window.document.querySelectorAll('script[type="speculationrules"]').length, 1);
  domSupported.window.close();

  // Unsupported browsers (supports() returns false or undefined) -> silent no-op
  for (const supportsImpl of [() => false, undefined]) {
    const domUnsupported = new JSDOM('<!DOCTYPE html><html><head></head><body></body></html>', {
      url: 'https://example.test/index.html',
      runScripts: 'outside-only',
    });
    if (supportsImpl) {
      domUnsupported.window.HTMLScriptElement.supports = supportsImpl;
    } else {
      delete domUnsupported.window.HTMLScriptElement.supports;
    }
    domUnsupported.window.eval(specBlock);
    assert.equal(domUnsupported.window.RNInstallSpeculationRules(), false);
    assert.equal(domUnsupported.window.document.querySelectorAll('script[type="speculationrules"]').length, 0);
    domUnsupported.window.close();
  }
});

test('#4 content-visibility: auto pairs with contain-intrinsic-size on offscreen sections only', () => {
  assert.match(
    styleSource,
    /#home-hub-section\s*\{[^}]*content-visibility:\s*auto;[^}]*contain-intrinsic-size:\s*auto none auto 520px;/,
  );
  assert.match(
    styleSource,
    /#detail-related:not\(\[hidden\]\),\s*#detail-recent-ingest-section:not\(\[hidden\]\),\s*#detail-recommended:not\(\[hidden\]\),\s*#detail-sources:not\(\[hidden\]\),\s*#detail-recent:not\(\[hidden\]\)\s*\{[^}]*content-visibility:\s*auto;[^}]*contain-intrinsic-size:\s*auto none auto 280px;/,
  );
  const miniGraphRules = styleSource.match(/#mini-graph-section\b[^{]*\{[^}]*\}/g) || [];
  for (const block of miniGraphRules) {
    assert.doesNotMatch(block, /content-visibility:\s*auto/, 'above-the-fold #mini-graph-section stays eager');
  }
});

test('#5 Mermaid/Plotly SRI hashes match vendored bundles and external subresources enforce Referrer-Policy', async () => {
  const plotlySha = 'sha384-' + crypto.createHash('sha384').update(readDoc('vendor/plotly.min.js')).digest('base64');
  const starIntegrityMatch = starHistoryHtml.match(/var PLOTLY_CDN_INTEGRITY = '([^']+)'/);
  assert.ok(starIntegrityMatch, 'PLOTLY_CDN_INTEGRITY declared in star-history.html');
  assert.equal(starIntegrityMatch[1], plotlySha, 'Plotly CDN integrity matches docs/vendor/plotly.min.js');

  const mermaidLocalPath = resolve(__dirname, '../node_modules/mermaid/dist/mermaid.min.js');
  const mermaidIntegrityMatch = mainSource.match(/var MERMAID_CDN_INTEGRITY = '([^']+)'/);
  assert.ok(mermaidIntegrityMatch, 'MERMAID_CDN_INTEGRITY declared in main.js');
  if (existsSync(mermaidLocalPath)) {
    const mermaidSha = 'sha384-' + crypto.createHash('sha384').update(readFileSync(mermaidLocalPath)).digest('base64');
    assert.equal(mermaidIntegrityMatch[1], mermaidSha, 'Mermaid CDN integrity matches installed mermaid.min.js');
  }

  // Test loadExternalAsset / ensureMermaidLoaded / ensureKatexLoaded in VM
  const loaderStart = mainSource.indexOf('  // ── 公式与图表组件按内容加载（编号 9）──');
  const loaderEnd = mainSource.indexOf('  var mermaidLightboxEl = null;', loaderStart);
  assert.ok(loaderStart >= 0 && loaderEnd > loaderStart);
  const appendedEls = [];
  const assetCtx = vm.createContext({
    Promise,
    window: {},
    document: {
      createElement: (tag) => ({ tagName: tag.toUpperCase(), onload: null, onerror: null, parentNode: null }),
      head: {
        appendChild(el) {
          appendedEls.push(el);
          setImmediate(() => el.onload && el.onload());
          el.parentNode = this;
        },
        removeChild() {},
      },
    },
  });
  vm.runInContext(mainSource.slice(loaderStart, loaderEnd), assetCtx);

  await assetCtx.ensureMermaidLoaded();
  const mermaidEl = appendedEls.find((e) => String(e.src).includes('mermaid@12.3.0'));
  assert.ok(mermaidEl);
  assert.equal(mermaidEl.integrity, mermaidIntegrityMatch[1]);
  assert.equal(mermaidEl.crossOrigin, 'anonymous');
  assert.equal(mermaidEl.referrerPolicy, 'no-referrer');

  await assetCtx.ensureKatexLoaded();
  const katexEls = appendedEls.filter((e) => String(e.src || e.href).includes('katex@0.16.11'));
  assert.equal(katexEls.length, 3);
  for (const el of katexEls) {
    assert.equal(el.crossOrigin, 'anonymous');
    assert.equal(el.referrerPolicy, 'no-referrer');
  }

  await assetCtx.loadExternalAsset({ url: 'graph-perspective.js' });
  const localEl = appendedEls.find((e) => e.src === 'graph-perspective.js');
  assert.ok(localEl);
  assert.equal(localEl.referrerPolicy, undefined, 'same-origin script does not force no-referrer');

  // Test star-history.html ensurePlotly fallback onto CDN with SRI + crossOrigin + referrerPolicy
  const plotlyStart = starHistoryHtml.indexOf('      var plotlyPromise = null;');
  const plotlyEnd = starHistoryHtml.indexOf('      function esc(s) {', plotlyStart);
  assert.ok(plotlyStart >= 0 && plotlyEnd > plotlyStart);
  const starScripts = [];
  const starCtx = vm.createContext({
    Promise,
    Error,
    window: {},
    document: {
      createElement: () => ({ onload: null, onerror: null }),
      head: {
        appendChild(el) {
          starScripts.push(el);
          setImmediate(() => {
            if (el.src === 'vendor/plotly.min.js') {
              el.onerror();
              return;
            }
            starCtx.window.Plotly = { newPlot() {} };
            el.onload();
          });
        },
      },
    },
  });
  vm.runInContext(starHistoryHtml.slice(plotlyStart, plotlyEnd), starCtx);
  const plotlyApi = await starCtx.ensurePlotly();
  assert.ok(plotlyApi);
  assert.equal(starScripts.length, 2);
  const cdnScript = starScripts[1];
  assert.equal(cdnScript.integrity, plotlySha);
  assert.equal(cdnScript.crossOrigin, 'anonymous');
  assert.equal(cdnScript.referrerPolicy, 'no-referrer');

  // All 12 HTML files declare <meta name="referrer" content="strict-origin-when-cross-origin" /> inside <head>
  for (const file of ALL_HTML_PAGES) {
    const html = readDoc(file);
    const head = html.match(/<head>([\s\S]*?)<\/head>/i)?.[1] || '';
    assert.match(
      head,
      /<meta\s+name="referrer"\s+content="strict-origin-when-cross-origin"\s*\/?>/i,
      `${file} sets Referrer-Policy meta in <head>`,
    );
  }
});

test('#6 & #7 style.css enables field-sizing: content on .kb-chat-input and reduced-motion-safe @view-transition', () => {
  assert.match(
    styleSource,
    /\.kb-chat-input\s*\{[^}]*flex:\s*1 1 auto;[^}]*field-sizing:\s*content;[^}]*height:\s*auto;[^}]*min-height:\s*38px;[^}]*max-height:\s*84px;/,
  );
  assert.match(
    styleSource,
    /@media\s*\(prefers-reduced-motion:\s*no-preference\)\s*\{[\s\S]*?@view-transition\s*\{\s*navigation:\s*auto;\s*\}[\s\S]*?html:not\(\.is-theme-switching\)::view-transition-old\(root\),\s*html:not\(\.is-theme-switching\)::view-transition-new\(root\)\s*\{\s*animation-duration:\s*0\.18s;\s*\}/,
  );
});
