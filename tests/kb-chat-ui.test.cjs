const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const { JSDOM } = require('jsdom');

const mainSource = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const chatSource = readFileSync(resolve(__dirname, '../docs/kb-chat.js'), 'utf8');

const fabStart = mainSource.indexOf('/* ── 全站「回到顶部」按钮（带阅读进度环）与右下角 AI 知识库问答入口 ──');
const fabEnd = mainSource.indexOf('/* ── 首页章节跳转按钮 ──', fabStart);
assert.ok(fabStart >= 0 && fabEnd > fabStart, 'right fab stack slice markers present in docs/main.js');
const fabBlock = mainSource.slice(fabStart, fabEnd);

const HASH_WBC = 'c'.repeat(64);
const HASH_ROADMAP = 'd'.repeat(64);

const mockLiteIndex = {
  meta: { version: 'v1', tier: 'lite', N: 2 },
  docs: [
    {
      id: 'wiki-concepts-wbc',
      path: 'wiki/concepts/whole-body-control.md',
      title: '全身控制（Whole-Body Control, WBC）',
      summary: '全身控制用于在多接触约束下协调人形机器人动力学与运动任务。',
      page_type: 'concept',
      tags: ['wbc', 'humanoid'],
      search_aliases: ['WBC', 'Whole-Body Control'],
    },
    {
      id: 'roadmap-motion-control',
      path: 'roadmap/motion-control.md',
      title: '运动控制主路线',
      summary: '从经典控制、MPC 到 WBC 与强化学习的机器人运动控制路线图。',
      page_type: 'roadmap_page',
      tags: ['roadmap'],
      search_aliases: ['运动控制路线'],
    },
  ],
};

const mockCatalog = {
  version: 'v1',
  content_mode: 'per-page-v1',
  pages: {
    detail_pages: {
      'wiki-concepts-wbc': {
        title: '全身控制（Whole-Body Control, WBC）',
        content_url: `exports/page-content/${HASH_WBC}.json`,
      },
      'roadmap-motion-control': {
        title: '运动控制主路线',
        type: 'roadmap_page',
        content_url: `exports/page-content/${HASH_ROADMAP}.json`,
      },
    },
    roadmap_pages: { 'roadmap-motion-control': {} },
    page_aliases: {},
  },
};

const mockBodies = {
  [`exports/page-content/${HASH_WBC}.json`]: {
    version: 'v1',
    id: 'wiki-concepts-wbc',
    content_markdown:
      '# 全身控制（WBC）\n\n## 核心原理\n\n全身控制（WBC）将质心轨迹与接触力约束统一到分层二次规划（QP）中求解。\n\n## 与 MPC 协同\n\n上层 MPC 规划步态与落足点，底层 WBC 跟踪期望加速度与接触力。',
  },
  [`exports/page-content/${HASH_ROADMAP}.json`]: {
    version: 'v1',
    id: 'roadmap-motion-control',
    content_markdown:
      '# 运动控制主路线\n\n## Stage 2：全身控制\n\n重点学习 Whole-Body Control 与逆动力学跟踪。',
  },
};

function setupDom({ withFooter = true, config = null, fetchImpl = null, lang = 'zh-CN' } = {}) {
  const html = withFooter
    ? `<!DOCTYPE html><html lang="${lang}"><head></head><body><main></main><footer class="footer">Footer</footer></body></html>`
    : `<!DOCTYPE html><html lang="${lang}"><head></head><body><main id="graph-wrap"></main></body></html>`;
  const dom = new JSDOM(html, {
    url: 'https://imchong.github.io/Robotics_Notebooks/index.html',
    runScripts: 'outside-only',
  });
  const { window } = dom;
  const fetchCalls = [];
  window.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  window.matchMedia = () => ({ matches: false });
  if (config) window.RN_KB_CHAT_CONFIG = config;
  window.fetch = async (url, opts) => {
    fetchCalls.push({ url: String(url), opts });
    if (fetchImpl) return fetchImpl(String(url), opts);
    if (String(url) === 'search-index-lite.json') {
      return { ok: true, status: 200, json: async () => mockLiteIndex };
    }
    if (String(url) === 'exports/site-catalog-v1.json') {
      return { ok: true, status: 200, json: async () => mockCatalog };
    }
    if (mockBodies[String(url)]) {
      return { ok: true, status: 200, json: async () => mockBodies[String(url)] };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };
  return { dom, window, fetchCalls };
}

test('Footer pages mount #fabStackRight with #backToTop and #kbChatToggle; graph.html (no footer) mounts neither; 0 fetches before click', () => {
  const graphDom = setupDom({ withFooter: false });
  graphDom.window.eval(fabBlock);
  assert.equal(graphDom.window.document.getElementById('fabStackRight'), null);
  assert.equal(graphDom.window.document.getElementById('backToTop'), null);
  assert.equal(graphDom.window.document.getElementById('kbChatToggle'), null);
  graphDom.dom.window.close();

  const homeDom = setupDom({ withFooter: true });
  homeDom.window.eval(fabBlock);
  const stack = homeDom.window.document.getElementById('fabStackRight');
  const btt = homeDom.window.document.getElementById('backToTop');
  const chatBtn = homeDom.window.document.getElementById('kbChatToggle');
  assert.ok(stack, '#fabStackRight mounted');
  assert.ok(stack.contains(btt), '#backToTop inside #fabStackRight');
  assert.ok(stack.contains(chatBtn), '#kbChatToggle inside #fabStackRight');
  assert.equal(homeDom.window.document.querySelectorAll('script[src="kb-chat.js"]').length, 0);
  assert.equal(homeDom.fetchCalls.length, 0);

  // Clicking #kbChatToggle lazily injects kb-chat.js once
  chatBtn.click();
  const injected = homeDom.window.document.querySelectorAll('script[src="kb-chat.js"]');
  assert.equal(injected.length, 1);
  assert.equal(homeDom.fetchCalls.length, 0);
  homeDom.dom.window.close();
});

test('Local read-only mode answers Chinese & English questions with clickable citations and reports not-found cleanly', async () => {
  const { dom, window, fetchCalls } = setupDom({ withFooter: true });
  window.eval(fabBlock);
  window.eval(chatSource);

  window.RNKbChat.open();
  assert.equal(window.RNKbChat.isOpen(), true);
  assert.equal(window.document.getElementById('kbChatPanel').hidden, false);

  // Chinese grounded query
  await window.RNKbChat.ask('什么是全身控制 WBC？');
  const messages = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(messages.length, 1);
  assert.match(messages[0].textContent, /全身控制/);
  const links = messages[0].querySelectorAll('a.kb-chat-citation-card');
  assert.ok(links.length >= 1);
  assert.equal(links[0].getAttribute('href'), 'detail.html?id=wiki-concepts-wbc');

  // Switch to English and ask English query
  window.RNKbChat.setLanguage('en');
  await window.RNKbChat.ask('How does Whole-Body Control work?');
  const messagesAfterEn = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(messagesAfterEn.length, 2);
  assert.match(messagesAfterEn[1].textContent, /Grounded excerpts/);

  // Unmatched query reports not-found cleanly
  await window.RNKbChat.ask('completelyunrelatedtopicxyz');
  const messagesUnmatched = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(messagesUnmatched.length, 3);
  assert.match(messagesUnmatched[2].textContent, /No direct supporting evidence/);
  assert.ok(messagesUnmatched[2].querySelector('a[href="graph.html"]'));
  assert.ok(fetchCalls.some((c) => c.url === 'search-index-lite.json'));

  dom.window.close();
});

test('Cloud quota exhaustion (503) and rate limit (429) automatically fall back to local read-only paragraph results', async () => {
  for (const statusCode of [503, 429]) {
    const { dom, window } = setupDom({
      withFooter: true,
      config: {
        endpoint: 'https://kb-chat.example.workers.dev/chat',
        turnstileSiteKey: '1x00000000000000000000AA',
      },
      fetchImpl: async (url) => {
        if (url === 'https://kb-chat.example.workers.dev/chat') {
          return {
            ok: false,
            status: statusCode,
            json: async () => ({
              ok: false,
              code: statusCode === 503 ? 'QUOTA_OR_AI_UNAVAILABLE' : 'RATE_LIMITED',
              fallback: 'local_search',
            }),
          };
        }
        if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
        if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => mockCatalog };
        if (mockBodies[url]) return { ok: true, status: 200, json: async () => mockBodies[url] };
        return { ok: false, status: 404, json: async () => ({}) };
      },
    });
    window.turnstile = {
      render: (_el, opts) => {
        window.__tsCallback = opts.callback;
        return 'wid-1';
      },
      reset: () => {},
      execute: () => window.__tsCallback('tok-demo'),
    };
    window.eval(fabBlock);
    window.eval(chatSource);
    window.RNKbChat.open();

    await window.RNKbChat.ask('全身控制 WBC 的原理是什么');
    const msg = window.document.querySelector('.kb-chat-msg--assistant');
    assert.ok(msg);
    const note = msg.querySelector('.kb-chat-status-note');
    assert.ok(note);
    assert.match(note.textContent, statusCode === 503 ? /免费额度/ : /过于频繁/);
    assert.ok(msg.querySelector('a[href="detail.html?id=wiki-concepts-wbc"]'));
    assert.equal(window.document.querySelector('.kb-chat-mode-badge').getAttribute('data-mode'), 'fallback');
    dom.window.close();
  }
});

test('XSS payloads and dangerous citation URLs are rendered inertly without scripts or external links', async () => {
  const { dom, window } = setupDom({
    withFooter: true,
    config: {
      endpoint: 'https://kb-chat.example.workers.dev/chat',
      turnstileSiteKey: '1x00000000000000000000AA',
    },
    fetchImpl: async (url) => {
      if (url === 'https://kb-chat.example.workers.dev/chat') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            ok: true,
            mode: 'ai',
            grounded: true,
            cached: false,
            answer:
              '<img src=x onerror=alert(1)><script>window.__xss=1</script>WBC uses QP [1] and roadmap [2].',
            citations: [
              {
                index: 1,
                id: 'wiki-concepts-wbc',
                title: '<svg onload=alert(1)>WBC',
                heading: '<b>Core</b>',
                snippet: '<script>evil()</script>Safe text',
                url: 'javascript:alert(1)',
              },
              {
                index: 2,
                id: 'roadmap-motion-control',
                title: 'Motion Roadmap',
                heading: 'Stage 2',
                snippet: 'Learn WBC',
                url: 'https://evil.example/phish',
              },
            ],
          }),
        };
      }
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => mockCatalog };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  window.turnstile = {
    render: (_el, opts) => {
      window.__tsCallback = opts.callback;
      return 'wid-1';
    },
    reset: () => {},
    execute: () => window.__tsCallback('tok-xss'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  const msg = window.document.querySelector('.kb-chat-msg--assistant');
  assert.ok(msg);
  assert.equal(window.__xss, undefined);
  assert.equal(msg.querySelectorAll('script, img, svg').length, 0, 'no HTML tags injected');
  const anchors = [...msg.querySelectorAll('a')];
  assert.ok(anchors.length >= 2);
  for (const a of anchors) {
    const href = a.getAttribute('href');
    assert.match(href, /^(?:detail|roadmap)\.html\?id=/, `unexpected href: ${href}`);
    assert.doesNotMatch(href, /^(?:javascript:|https?:|\/\/)/i);
  }
  dom.window.close();
});
