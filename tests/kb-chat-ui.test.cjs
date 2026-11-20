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
  window.RN_KB_CHAT_CONFIG = config || { endpoint: '', turnstileSiteKey: '' };
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

test('Footer pages mount #fabStackRight with merged scroll launcher; graph.html (no footer) mounts neither; 0 fetches before click', () => {
  const graphDom = setupDom({ withFooter: false });
  graphDom.window.eval(fabBlock);
  assert.equal(graphDom.window.document.getElementById('fabStackRight'), null);
  assert.equal(graphDom.window.document.getElementById('backToTop'), null);
  assert.equal(graphDom.window.document.getElementById('kbChatToggle'), null);
  assert.equal(graphDom.window.document.getElementById('fabMainToggle'), null);
  graphDom.dom.window.close();

  const homeDom = setupDom({ withFooter: true });
  homeDom.window.eval(fabBlock);
  const stack = homeDom.window.document.getElementById('fabStackRight');
  const actionsRow = homeDom.window.document.getElementById('fabActionsRow');
  const btt = homeDom.window.document.getElementById('backToTop');
  const slot = homeDom.window.document.getElementById('kbChatSlot');
  const chatBtn = homeDom.window.document.getElementById('kbChatToggle');
  const mainBtn = homeDom.window.document.getElementById('fabMainToggle');
  assert.ok(stack, '#fabStackRight mounted');
  assert.ok(stack.contains(slot), '#kbChatSlot inside #fabStackRight');
  assert.ok(stack.contains(actionsRow), '#fabActionsRow inside #fabStackRight');
  assert.ok(actionsRow.contains(btt), '#backToTop inside #fabActionsRow');
  assert.ok(actionsRow.contains(chatBtn), '#kbChatToggle inside #fabActionsRow');
  assert.ok(actionsRow.contains(mainBtn), '#fabMainToggle inside #fabActionsRow');
  assert.equal(homeDom.window.document.querySelector('.fab-stack-right__bar'), null);
  assert.deepEqual(
    Array.from(stack.children).map((el) => el.id),
    ['kbChatSlot', 'fabActionsRow'],
    'panel slot sits above the bottom actions row',
  );
  assert.deepEqual(
    Array.from(actionsRow.children).map((el) => el.id),
    ['backToTop', 'kbChatToggle', 'fabMainToggle'],
    'bottom actions row places #backToTop to the left of #kbChatToggle',
  );
  // At page top (scrollY = 0), only #kbChatToggle is visible so user opens chat in 1 click
  assert.equal(btt.hidden, true);
  assert.equal(mainBtn.hidden, true);
  assert.equal(chatBtn.hidden, false);
  assert.ok(chatBtn.querySelector('svg.kb-chat-toggle__icon .kb-chat-toggle__bubble'), 'chat button renders speech-bubble outline');
  assert.equal(chatBtn.querySelectorAll('svg.kb-chat-toggle__icon circle.kb-chat-toggle__dot').length, 3, 'chat button renders three dots inside bubble');
  assert.equal(homeDom.window.document.querySelectorAll('script[src="kb-chat.js"]').length, 0);
  assert.equal(homeDom.fetchCalls.length, 0);

  // Scrolled past threshold: #fabMainToggle merges #backToTop and #kbChatToggle into 1 button
  Object.defineProperty(homeDom.window.document.documentElement, 'scrollHeight', { value: 2400, configurable: true });
  Object.defineProperty(homeDom.window, 'innerHeight', { value: 800, configurable: true });
  homeDom.window.document.documentElement.scrollTop = 900;
  homeDom.window.dispatchEvent(new homeDom.window.Event('scroll'));
  homeDom.dom.window.close();
});

test('Scrolled merged button expands to vertical 2-button stack and collapses on outside-click, Escape, or chat close', async () => {
  const { dom, window } = setupDom({ withFooter: true });
  let rafQueue = [];
  window.requestAnimationFrame = (cb) => {
    rafQueue.push(cb);
    return rafQueue.length;
  };
  const flushRaf = () => {
    while (rafQueue.length) {
      const next = rafQueue.shift();
      next();
    }
  };

  Object.defineProperty(window.document.documentElement, 'scrollHeight', { value: 2400, configurable: true });
  Object.defineProperty(window, 'innerHeight', { value: 800, configurable: true });
  window.eval(fabBlock);

  const stack = window.document.getElementById('fabStackRight');
  const btt = window.document.getElementById('backToTop');
  const chatBtn = window.document.getElementById('kbChatToggle');
  const mainBtn = window.document.getElementById('fabMainToggle');

  // Scroll down -> collapsed to single #fabMainToggle with progress ring
  window.document.documentElement.scrollTop = 900;
  window.dispatchEvent(new window.Event('scroll'));
  flushRaf();
  assert.equal(mainBtn.hidden, false, '#fabMainToggle shown when scrolled');
  assert.equal(btt.hidden, true, '#backToTop hidden before expand');
  assert.equal(chatBtn.hidden, true, '#kbChatToggle hidden before expand');
  assert.ok(parseFloat(mainBtn.style.getPropertyValue('--btt-progress')) > 0, 'progress ring updated on #fabMainToggle');

  // Clicking #fabMainToggle expands into vertical #backToTop (top) + #kbChatToggle (bottom)
  mainBtn.click();
  assert.equal(mainBtn.hidden, true);
  assert.equal(btt.hidden, false);
  assert.equal(chatBtn.hidden, false);
  assert.ok(btt.classList.contains('is-visible'));
  assert.ok(stack.classList.contains('is-expanded'));

  // Clicking outside collapses back to #fabMainToggle
  window.document.querySelector('main').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  assert.equal(mainBtn.hidden, false);
  assert.equal(btt.hidden, true);
  assert.equal(chatBtn.hidden, true);

  // Expand again -> Escape collapses back to #fabMainToggle
  mainBtn.click();
  assert.equal(btt.hidden, false);
  window.document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  assert.equal(mainBtn.hidden, false);
  assert.equal(btt.hidden, true);
  assert.equal(chatBtn.hidden, true);

  // Opening chat via #kbChatToggle keeps #backToTop + #kbChatToggle visible; clicking #kbChatToggle again collapses chat and returns to #fabMainToggle
  window.eval(chatSource);
  mainBtn.click();
  chatBtn.click();
  const panel = window.document.getElementById('kbChatPanel');
  assert.equal(panel.hidden, false);
  assert.equal(chatBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(chatBtn.getAttribute('title'), '收起对话');
  assert.equal(btt.hidden, false);
  assert.equal(chatBtn.hidden, false);
  assert.equal(mainBtn.hidden, true);
  chatBtn.click();
  assert.equal(panel.hidden, true);
  assert.equal(chatBtn.getAttribute('aria-expanded'), 'false');
  assert.equal(chatBtn.getAttribute('title'), '知识问答');
  assert.equal(mainBtn.hidden, false);
  assert.equal(btt.hidden, true);
  assert.equal(chatBtn.hidden, true);

  // Scrolling back to top collapses back to single #kbChatToggle and clicking it toggles open/close cleanly
  mainBtn.click();
  btt.focus();
  window.document.documentElement.scrollTop = 0;
  window.dispatchEvent(new window.Event('scroll'));
  flushRaf();
  assert.equal(mainBtn.hidden, true);
  assert.equal(btt.hidden, true);
  assert.equal(chatBtn.hidden, false);
  assert.equal(window.document.activeElement, chatBtn);
  chatBtn.click();
  assert.equal(panel.hidden, false);
  chatBtn.click();
  assert.equal(panel.hidden, true);
  assert.equal(chatBtn.hidden, false);

  dom.window.close();
});

test('Top-of-page single #kbChatToggle lazily injects kb-chat.js once even on repeated clicks', () => {
  const homeDom = setupDom({ withFooter: true });
  homeDom.window.eval(fabBlock);
  const chatBtn = homeDom.window.document.getElementById('kbChatToggle');
  assert.equal(chatBtn.hidden, false);
  chatBtn.click();
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
  assert.equal(window.document.querySelector('.kb-chat-header__title').textContent, '知识问答');
  assert.equal(window.document.getElementById('kbChatToggle').getAttribute('title'), '收起对话');

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
  assert.equal(window.document.querySelector('.kb-chat-header__title').textContent, 'Knowledge Q&A');
  assert.equal(window.document.getElementById('kbChatToggle').getAttribute('title'), 'Close Q&A');
  await window.RNKbChat.ask('How does Whole-Body Control work?');
  const messagesAfterEn = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(messagesAfterEn.length, 2);
  assert.match(messagesAfterEn[1].textContent, /Relevant sections from the wiki/);

  // Unmatched query reports not-found cleanly
  await window.RNKbChat.ask('completelyunrelatedtopicxyz');
  const messagesUnmatched = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(messagesUnmatched.length, 3);
  assert.match(messagesUnmatched[2].textContent, /No direct match found in the wiki/);
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
    assert.match(note.textContent, statusCode === 503 ? /繁忙/ : /频繁/);
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
  assert.equal(window.document.getElementById('kbChatTurnstile').hidden, true);
  await window.RNKbChat.ask('在吗');
  const greetingMsgs = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(greetingMsgs.length, 2);
  assert.match(greetingMsgs[1].textContent, /只读问答助手/);
  dom.window.close();
});

test('LaTeX math delimiters normalize safely and invoke KaTeX rendering without HTML injection', async () => {
  let renderedContainers = 0;
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
              '动力学方程 $\\tau = J^T F$，代价约 $50k–$70k，块级公式：$$\\min_x \\|Ax-b\\|^2$$ [1]。<script>window.__mathXss=1</script>',
            citations: [
              {
                index: 1,
                id: 'wiki-concepts-wbc',
                title: 'Whole-Body Control',
                heading: 'QP',
                snippet: '满足 $Ax = b$ 约束',
                url: 'detail.html?id=wiki-concepts-wbc',
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
  window.renderMathInElement = (el, opts) => {
    renderedContainers += 1;
    assert.ok(Array.isArray(opts.delimiters));
    assert.ok(el.textContent.includes('\\(\\tau = J^T F\\)'));
    assert.ok(el.textContent.includes('$$\\min_x \\|Ax-b\\|^2$$'));
    assert.ok(el.textContent.includes('$50k–$70k'));
    assert.ok(el.textContent.includes('\\(Ax = b\\)'));
  };
  window.turnstile = {
    render: (_el, opts) => {
      window.__tsCallback = opts.callback;
      return 'wid-1';
    },
    reset: () => {},
    execute: () => window.__tsCallback('tok-math'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC $J^T F$？');
  assert.ok(renderedContainers >= 2, 'should invoke KaTeX for both user and assistant messages');
  assert.equal(window.__mathXss, undefined);
  assert.equal(window.document.querySelectorAll('.kb-chat-msg script').length, 0);
  dom.window.close();
});

test('Markdown headings, lists, code blocks, tables, bold/italic and safe internal links render cleanly without XSS', async () => {
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
            answer: [
              '## 核心原理',
              '- **优势**：使用 `QP` 求解动力学 $\\tau = J^T F$ [1]',
              '- *注意*：查看 [站内节点](detail.html?id=wiki-concepts-wbc) 与 [外链](https://evil.example) 和 [危险](javascript:alert(1))',
              '',
              '| 模块 | 作用 |',
              '| --- | --- |',
              '| WBC | 全身力矩分配 |',
              '',
              '```python',
              'solve_qp(A, b)',
              '```',
            ].join('\n'),
            citations: [
              {
                index: 1,
                id: 'wiki-concepts-wbc',
                title: 'Whole-Body Control',
                heading: 'QP',
                snippet: '满足 QP 约束',
                url: 'detail.html?id=wiki-concepts-wbc',
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
    execute: () => window.__tsCallback('tok-md'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  const msg = window.document.querySelector('.kb-chat-msg--assistant');
  assert.ok(msg);
  assert.equal(msg.querySelector('.kb-chat-md-h2').textContent, '核心原理');
  assert.equal(msg.querySelectorAll('.kb-chat-md-list li').length, 2);
  assert.equal(msg.querySelector('strong').textContent, '优势');
  assert.equal(msg.querySelector('em').textContent, '注意');
  assert.equal(msg.querySelector('.kb-chat-md-code').textContent, 'QP');
  assert.equal(msg.querySelector('.kb-chat-md-pre code').textContent, 'solve_qp(A, b)');
  assert.equal(msg.querySelectorAll('.kb-chat-md-table tbody tr').length, 1);
  const mdLinks = [...msg.querySelectorAll('a.kb-chat-md-link')];
  assert.equal(mdLinks.length, 1);
  assert.equal(mdLinks[0].getAttribute('href'), 'detail.html?id=wiki-concepts-wbc');
  assert.match(msg.textContent, /外链/);
  assert.match(msg.textContent, /危险/);
  dom.window.close();
});

test('Citation cards display canonical node title, community, roadmap, updated date, and health score', async () => {
  const richCatalog = {
    version: 'v1',
    content_mode: 'per-page-v1',
    pages: {
      detail_pages: {
        'wiki-concepts-wbc': {
          title: '全身控制（Whole-Body Control, WBC）',
          path: 'wiki/concepts/whole-body-control.md',
          updated: '2026-11-18',
          health_score: 3,
          content_url: `exports/page-content/${HASH_WBC}.json`,
        },
      },
      roadmap_pages: {
        'roadmap-motion-control': {
          title: '运动控制主路线（Motion Control）',
          stages: [
            {
              id: 'stage-2',
              related_items: ['wiki-concepts-wbc'],
            },
          ],
        },
      },
      page_aliases: {},
    },
  };
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
            answer: 'WBC 核心是分层 QP [1]。',
            citations: [
              {
                index: 1,
                id: 'wiki-concepts-wbc',
                title: 'H1 Fallback Title',
                heading: '核心原理',
                snippet: '分层 QP 求解接触力。',
                url: 'detail.html?id=wiki-concepts-wbc',
              },
            ],
          }),
        };
      }
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => richCatalog };
      if (url === 'exports/link-graph.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            nodes: [{ id: 'wiki/concepts/whole-body-control.md', community: 'comm-control' }],
            communities: [{ id: 'comm-control', label: '运动控制（Motion Control） 社区' }],
          }),
        };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  window.turnstile = {
    render: (_el, opts) => {
      window.__tsCallback = opts.callback;
      return 'wid-1';
    },
    reset: () => {},
    execute: () => window.__tsCallback('tok-meta'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  // Flush microtasks for async link-graph community lookup
  await new Promise((r) => setTimeout(r, 15));

  const card = window.document.querySelector('.kb-chat-citation-card');
  assert.ok(card);
  assert.equal(card.querySelector('.kb-chat-citation-title').textContent, '[1] 全身控制（Whole-Body Control, WBC）');
  assert.equal(card.querySelector('.kb-chat-citation-community').textContent, '社区 · 运动控制');
  assert.equal(card.querySelector('.kb-chat-citation-roadmap').textContent, '路线 · 运动控制主路线');
  assert.equal(card.querySelector('.kb-chat-citation-updated').textContent, '更新 2026-11-18');
  assert.match(card.querySelector('.kb-chat-citation-health').textContent, /健康度 3\/3/);
  assert.equal(card.querySelectorAll('.kb-chat-citation-health .node-health--3 i.is-on').length, 3);
  dom.window.close();
});

test('Mermaid fenced code blocks render as diagrams and support click-to-zoom lightbox', async () => {
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
              '流程如下：\n```mermaid\ngraph LR\n  A --> B\n```\n时序如下：\n```\nsequenceDiagram\n  A->>B: ping\n```',
            citations: [],
          }),
        };
      }
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => mockCatalog };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  let mermaidInitSecurity = '';
  window.mermaid = {
    initialize: (opts) => {
      mermaidInitSecurity = opts.securityLevel;
    },
    run: async ({ nodes }) => {
      for (const n of nodes) {
        const svg = window.document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        n.appendChild(svg);
      }
    },
  };
  window.turnstile = {
    render: (_el, opts) => {
      window.__tsCallback = opts.callback;
      return 'wid-1';
    },
    reset: () => {},
    execute: () => window.__tsCallback('tok-mermaid'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  await new Promise((r) => setTimeout(r, 15));

  assert.equal(mermaidInitSecurity, 'strict');
  const mermaidHosts = window.document.querySelectorAll('.kb-chat-md-body .mermaid');
  assert.equal(mermaidHosts.length, 2);
  const mermaidHost = mermaidHosts[0];
  assert.equal(window.document.querySelector('.kb-chat-md-pre'), null);
  assert.ok(mermaidHost.classList.contains('mermaid-zoomable'));
  mermaidHost.click();
  const lightbox = window.document.querySelector('.mermaid-lightbox');
  assert.ok(lightbox && lightbox.hidden === false);
  assert.ok(lightbox.querySelector('svg'));
  dom.window.close();
});

test('Chat history persists in localStorage across page reloads until cache is cleared', async () => {
  const first = setupDom({ withFooter: true });
  first.window.eval(fabBlock);
  first.window.eval(chatSource);
  first.window.RNKbChat.open();
  await first.window.RNKbChat.ask('什么是全身控制 WBC？');
  const savedRaw = first.window.localStorage.getItem('rn_kb_chat_history_v1');
  assert.ok(savedRaw && savedRaw.includes('什么是全身控制 WBC？'));
  first.dom.window.close();

  const second = setupDom({ withFooter: true });
  second.window.localStorage.setItem('rn_kb_chat_history_v1', savedRaw);
  second.window.eval(fabBlock);
  second.window.eval(chatSource);
  second.window.RNKbChat.open();

  const userMsgs = second.window.document.querySelectorAll('.kb-chat-msg--user');
  const assistantMsgs = second.window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(userMsgs.length, 1);
  assert.equal(assistantMsgs.length, 1);
  assert.equal(userMsgs[0].textContent, '什么是全身控制 WBC？');
  assert.ok(assistantMsgs[0].querySelector('a.kb-chat-citation-card'));
  second.dom.window.close();
});
