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

test('Footer and graph.html pages mount #fabStackRight appropriately with 0 fetches before click', () => {
  const graphDom = setupDom({ withFooter: false });
  graphDom.window.eval(fabBlock);
  assert.ok(graphDom.window.document.getElementById('fabStackRight'), 'graph.html mounts #fabStackRight');
  assert.ok(graphDom.window.document.getElementById('kbChatToggle'), 'graph.html mounts chat toggle');
  assert.equal(graphDom.window.document.getElementById('backToTop').hidden, true);
  assert.equal(graphDom.window.document.getElementById('fabMainToggle').hidden, true);
  assert.equal(graphDom.fetchCalls.length, 0);
  graphDom.dom.window.close();

  const bareDom = setupDom({ withFooter: false });
  bareDom.window.document.getElementById('graph-wrap').remove();
  bareDom.window.eval(fabBlock);
  assert.equal(bareDom.window.document.getElementById('fabStackRight'), null);
  bareDom.dom.window.close();

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
  const links = messages[0].querySelectorAll('a.kb-chat-citation-passage');
  assert.ok(links.length >= 1);
  assert.ok(links[0].getAttribute('href').startsWith('detail.html?id=wiki-concepts-wbc'));
  assert.match(links[0].getAttribute('href'), /&cite=/);

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
    assert.ok(msg.querySelector('a[href^="detail.html?id=wiki-concepts-wbc"]'));
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

  const group = window.document.querySelector('.kb-chat-citation-group');
  assert.ok(group);
  assert.equal(group.querySelector('.kb-chat-citation-title').textContent, '全身控制（Whole-Body Control, WBC）');
  assert.equal(group.querySelector('.kb-chat-citation-community').textContent, '运动控制');
  assert.equal(group.querySelector('.kb-chat-citation-community').getAttribute('title'), '社区 · 运动控制');
  assert.equal(group.querySelector('.kb-chat-citation-roadmap').textContent, '运动控制主路线');
  assert.equal(group.querySelector('.kb-chat-citation-roadmap').getAttribute('title'), '路线 · 运动控制主路线');
  assert.equal(group.querySelector('.kb-chat-citation-updated').textContent, '2026-11-18');
  assert.equal(group.querySelector('.kb-chat-citation-updated').getAttribute('title'), '更新 2026-11-18');
  assert.match(group.querySelector('.kb-chat-citation-health').textContent, /健康度 3\/3/);
  assert.equal(group.querySelectorAll('.kb-chat-citation-health .node-health--3 i.is-on').length, 3);
  assert.equal(group.querySelector('.kb-chat-citation-idx').textContent, '[1]');
  assert.equal(group.querySelector('.kb-chat-citation-heading').textContent, '核心原理');
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
  assert.ok(assistantMsgs[0].querySelector('a.kb-chat-citation-passage'));
  second.dom.window.close();
});

test('Clear local chat history button clears UI and localStorage without sending any network request', async () => {
  const { dom, window, fetchCalls } = setupDom({ withFooter: true });
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  const clearBtn = window.document.getElementById('kbChatClear');
  assert.ok(clearBtn, '#kbChatClear button exists');
  assert.equal(clearBtn.disabled, true, 'clear button disabled when empty');

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  assert.equal(clearBtn.disabled, false, 'clear button enabled after conversation');
  assert.ok(window.localStorage.getItem('rn_kb_chat_history_v1'));
  assert.ok(window.document.querySelectorAll('.kb-chat-msg').length >= 2);

  const fetchCountBeforeClear = fetchCalls.length;
  clearBtn.click();

  assert.equal(window.document.querySelectorAll('.kb-chat-msg').length, 0);
  assert.equal(window.localStorage.getItem('rn_kb_chat_history_v1'), null);
  assert.equal(clearBtn.disabled, true);
  assert.equal(fetchCalls.length, fetchCountBeforeClear, 'clearing local history never triggers server requests');
  assert.ok(window.document.querySelector('.kb-chat-welcome'));
  assert.ok(window.document.querySelectorAll('.kb-chat-starter-btn').length >= 1);
  dom.window.close();
});

test('Clicking citation card scrolls to quoted passage inside details and pulses highlight twice', async () => {
  const citeStart = mainSource.indexOf('function normalizeCiteText(str)');
  const citeEnd = mainSource.indexOf('// ⚡ Bolt Optimization', citeStart);
  assert.ok(citeStart >= 0 && citeEnd > citeStart, 'citation jump block present in docs/main.js');
  const citeBlock = mainSource.slice(citeStart, citeEnd);

  const cssSource = readFileSync(resolve(__dirname, '../docs/style.css'), 'utf8');
  assert.match(cssSource, /@keyframes kbCitePulseTwice/);
  assert.match(cssSource, /72%[\s\S]*?26%/);

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
            answer: 'WBC 采用分层 QP 求解 [1]。',
            citations: [
              {
                index: 1,
                id: 'wiki-concepts-wbc',
                title: '全身控制（Whole-Body Control, WBC）',
                heading: '核心原理',
                snippet: '全身控制（WBC）将质心轨迹与接触力约束统一到分层二次规划（QP）中求解。',
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

  // Simulate currently viewing detail.html?id=wiki-concepts-wbc with a collapsed section containing the quoted paragraph
  dom.reconfigure({ url: 'https://imchong.github.io/Robotics_Notebooks/detail.html?id=wiki-concepts-wbc' });
  const detailContent = window.document.createElement('div');
  detailContent.id = 'detailContent';
  detailContent.innerHTML =
    '<h2>其他章节</h2><p>无关内容。</p>' +
    '<h2>核心原理</h2><details id="secFold"><summary>展开</summary>' +
    '<p id="targetPara"><strong>全身控制（WBC）</strong>将质心轨迹与接触力约束统一到分层二次规划（QP）中求解。</p>' +
    '</details>';
  window.document.querySelector('main').appendChild(detailContent);

  let scrolledTarget = null;
  const targetPara = window.document.getElementById('targetPara');
  targetPara.scrollIntoView = (opts) => {
    scrolledTarget = { el: targetPara, opts };
  };

  window.turnstile = {
    render: (_el, opts) => {
      window.__tsCallback = opts.callback;
      return 'wid-1';
    },
    reset: () => {},
    execute: () => window.__tsCallback('tok-cite'),
  };

  window.eval(citeBlock);
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  const passage = window.document.querySelector('a.kb-chat-citation-passage');
  assert.ok(passage);
  assert.match(passage.getAttribute('href'), /&cite=.*&cite_h=/);

  passage.click();
  assert.equal(window.document.getElementById('secFold').open, true, 'collapsed section auto-expands');
  assert.equal(scrolledTarget && scrolledTarget.el, targetPara, 'scrolled to exact quoted paragraph');
  assert.ok(targetPara.classList.contains('kb-cite-highlight'), 'highlight animation class applied');
  dom.window.close();
});

test('Same-document citations group into one compact card with collapsible excerpts and async community persistence', async () => {
  const richCatalog = {
    version: 'v1',
    content_mode: 'per-page-v1',
    pages: {
      detail_pages: {
        'wiki-concepts-wbc': {
          title: '全身控制（Whole-Body Control, WBC）',
          path: 'wiki/concepts/whole-body-control.md',
          updated: '2026-11-20',
          health_score: 3,
          content_url: `exports/page-content/${HASH_WBC}.json`,
        },
        'roadmap-motion-control': {
          title: '运动控制主路线',
          path: 'roadmap/motion-control.md',
          updated: '2026-11-18',
          health_score: 3,
          content_url: `exports/page-content/${HASH_ROADMAP}.json`,
        },
      },
      roadmap_pages: {
        'roadmap-motion-control': {
          title: '运动控制主路线（Motion Control）',
          stages: [{ id: 'stage-1', related_items: ['wiki-concepts-wbc'] }],
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
            answer: 'WBC 综合 [1]、[2]、[3] 与路线 [4]。',
            citations: [
              {
                index: 1,
                id: 'wiki-concepts-wbc',
                title: 'WBC',
                heading: '核心原理',
                snippet: '第一段摘录：分层 QP 求解。',
                url: 'detail.html?id=wiki-concepts-wbc',
              },
              {
                index: 2,
                id: 'wiki-concepts-wbc',
                title: 'WBC',
                heading: '与 MPC 协同',
                snippet: '第二段摘录：上层规划落足点。',
                url: 'detail.html?id=wiki-concepts-wbc',
              },
              {
                index: 3,
                id: 'wiki-concepts-wbc',
                title: 'WBC',
                heading: '参考来源',
                snippet: '第三段摘录：经典论文出处。',
                url: 'detail.html?id=wiki-concepts-wbc',
              },
              {
                index: 4,
                id: 'roadmap-motion-control',
                title: '运动控制主路线',
                heading: 'Stage 2',
                snippet: '第四段摘录：路线衔接说明。',
                url: 'roadmap.html?id=roadmap-motion-control',
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
    execute: () => window.__tsCallback('tok-group'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  await new Promise((r) => setTimeout(r, 20));

  const citeWrap = window.document.querySelector('.kb-chat-citations');
  assert.ok(citeWrap);
  assert.equal(citeWrap.querySelector('.kb-chat-citations__label').textContent, '参考来源 · 2 篇文档 4 处引用');
  assert.equal(citeWrap.classList.contains('is-snippets-open'), false, 'snippets collapsed by default');

  const groups = citeWrap.querySelectorAll('.kb-chat-citation-group');
  assert.equal(groups.length, 2, '3 citations from same page collapse into 1 group card');
  assert.equal(groups[0].querySelectorAll('.kb-chat-citation-meta').length, 1);
  assert.equal(groups[0].querySelectorAll('a.kb-chat-citation-passage').length, 3);
  const hrefs = [...groups[0].querySelectorAll('a.kb-chat-citation-passage')].map((a) => a.getAttribute('href'));
  assert.equal(new Set(hrefs).size, 3, 'each passage keeps its own section cite_h anchor');

  const toggleBtn = citeWrap.querySelector('.kb-chat-citations__toggle');
  assert.ok(toggleBtn);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'false');
  assert.equal(toggleBtn.textContent, '显示摘录 ▾');

  toggleBtn.click();
  assert.equal(citeWrap.classList.contains('is-snippets-open'), true);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(toggleBtn.textContent, '收起摘录 ▴');

  toggleBtn.click();
  assert.equal(citeWrap.classList.contains('is-snippets-open'), false);
  assert.equal(toggleBtn.getAttribute('aria-expanded'), 'false');

  const saved = JSON.parse(window.localStorage.getItem('rn_kb_chat_history_v1'));
  const savedCites = saved[1].citations;
  assert.equal(savedCites[0].community_label, '运动控制');
  assert.equal(savedCites[1].community_label, '运动控制');
  assert.equal(savedCites[2].community_label, '运动控制');
  dom.window.close();
});

test('Mermaid edge syntax sanitizer fixes trailing > on flowchart labels before rendering', async () => {
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
            answer: '```mermaid\ngraph LR\n  A[接触感知] -->|关键模态|> B[触觉]\n```',
            citations: [],
          }),
        };
      }
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => mockCatalog };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  let capturedSource = '';
  window.mermaid = {
    initialize: () => {},
    run: async ({ nodes }) => {
      for (const n of nodes) {
        capturedSource = n.textContent || '';
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
    execute: () => window.__tsCallback('tok-mmd-fix'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('目前机器人触觉一共分几类？');
  await new Promise((r) => setTimeout(r, 15));

  assert.equal(capturedSource, 'graph LR\n  A[接触感知] -->|关键模态| B[触觉]');
  assert.equal(window.document.querySelector('.kb-chat-md-pre'), null);
  dom.window.close();
});

test('graph.html harmonizes #kbChatToggle with #graph-legend-fab while leaving global style.css accent intact', () => {
  const graphHtml = readFileSync(resolve(__dirname, '../docs/graph.html'), 'utf8');
  const styleCss = readFileSync(resolve(__dirname, '../docs/style.css'), 'utf8');

  assert.match(
    graphHtml,
    /\.fab-stack-right\s+\.kb-chat-toggle\s*\{[^}]*background:\s*rgba\(13,\s*17,\s*23,\s*0\.92\)/,
    'graph.html dark chat toggle matches #graph-legend-fab surface',
  );
  assert.match(
    graphHtml,
    /\[data-theme="light"\]\s+\.fab-stack-right\s+\.kb-chat-toggle\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.96\)/,
    'graph.html light chat toggle matches #graph-legend-fab surface',
  );
  assert.match(
    graphHtml,
    /\.fab-stack-right\s+\.kb-chat-toggle\[aria-expanded="true"\]/,
    'graph.html overrides expanded chat toggle state for dark mode',
  );
  assert.match(
    graphHtml,
    /\[data-theme="light"\]\s+\.fab-stack-right\s+\.kb-chat-toggle\[aria-expanded="true"\]/,
    'graph.html overrides expanded chat toggle state for light mode',
  );
  assert.match(
    graphHtml,
    /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{\s*\.fab-stack-right\s+\.kb-chat-toggle\s*\{\s*transition:\s*none;/,
    'graph.html respects prefers-reduced-motion on chat toggle',
  );
  assert.match(
    styleCss,
    /\.kb-chat-toggle\s*\{[^}]*background:\s*var\(--accent\)/,
    'global style.css keeps default accent styling for non-graph pages',
  );
});

test('Author info queries reply immediately with clickable personal website link and 0 network fetches while sparing paper author queries', async () => {
  const { dom, window, fetchCalls } = setupDom({
    withFooter: true,
    config: {
      endpoint: 'https://kb-chat.example.workers.dev/chat',
      turnstileSiteKey: '1x00000000000000000000AA',
    },
  });
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('请问作者信息是什么？个人网站链接发一下');
  assert.equal(fetchCalls.length, 0, 'Chinese author query requires 0 network requests');
  const msgsZh = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(msgsZh.length, 1);
  const linkZh = msgsZh[0].querySelector('a.kb-chat-md-link');
  assert.ok(linkZh);
  assert.equal(linkZh.getAttribute('href'), 'https://imchong.github.io/index.html');
  assert.equal(linkZh.getAttribute('target'), '_blank');
  assert.equal(linkZh.getAttribute('rel'), 'noopener noreferrer');

  window.RNKbChat.setLanguage('en');
  await window.RNKbChat.ask('Who is the author of this knowledge base?');
  assert.equal(fetchCalls.length, 0, 'English author query requires 0 network requests');
  const msgsEn = window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.equal(msgsEn.length, 2);
  const linkEn = msgsEn[1].querySelector('a.kb-chat-md-link');
  assert.ok(linkEn);
  assert.equal(linkEn.getAttribute('href'), 'https://imchong.github.io/index.html');
  assert.match(msgsEn[1].textContent, /Chong Liu/);

  // Paper / third-party repo author questions must NOT be intercepted as site author info
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('这个项目的作者是谁？'), true);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('你是谁做的？'), true);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('PPO 论文作者是谁？'), false);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('BeyondMimic 的作者是谁？'), false);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('BeyondMimic 项目的作者是谁？'), false);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('Isaac Lab 仓库的维护者是谁？'), false);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('你可以告诉我 BeyondMimic 的作者是谁吗？'), false);
  assert.equal(window.RNKbChat._internals.isAuthorInfoQuery('Who are the authors of Diffusion Policy?'), false);
  assert.equal(window.RNKbChat._internals.isKbTrendQuery('本仓库最近节点趋势'), true);
  assert.equal(window.RNKbChat._internals.isKbTrendQuery('Isaac Lab 仓库最近有什么更新？'), false);
  assert.equal(window.RNKbChat._internals.isKbTrendQuery('scene graph recent updates'), false);
  dom.window.close();
});

test('Markdown link sanitizer allows strict author URL and safe static pages while stripping arbitrary external URLs', async () => {
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
              '[作者主页](https://imchong.github.io/index.html) · [节点趋势](star-history.html#nodes-daily) · [恶意](https://evil.example) · [伪造子路径](https://imchong.github.io/evil) · [XSS](javascript:alert(1))',
            citations: [],
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
    execute: () => window.__tsCallback('tok-links'),
  };
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  await window.RNKbChat.ask('什么是全身控制 WBC？');
  const msg = window.document.querySelector('.kb-chat-msg--assistant');
  const links = [...msg.querySelectorAll('a.kb-chat-md-link')];
  assert.equal(links.length, 2);
  assert.equal(links[0].getAttribute('href'), 'https://imchong.github.io/index.html');
  assert.equal(links[0].getAttribute('target'), '_blank');
  assert.equal(links[1].getAttribute('href'), 'star-history.html#nodes-daily');
  assert.equal(links[1].getAttribute('target'), null);
  assert.match(msg.textContent, /恶意/);
  assert.match(msg.textContent, /伪造子路径/);
  assert.match(msg.textContent, /XSS/);
  dom.window.close();
});

test('Local node trends answer reads graph-stats.json or falls back to compact wiki-activity.json cleanly', async () => {
  const sampleStatsWithTrends = {
    generated_at: '2026-11-23',
    node_count: 8200,
    edge_count: 67000,
    community_count: 35,
    node_trends: {
      as_of: '2026-11-23',
      latest_activity_date: '2026-11-23',
      total_nodes: 8200,
      total_edges: 67000,
      total_communities: 35,
      windows: {
        '7d': {
          days: 7,
          start_date: '2026-11-17',
          end_date: '2026-11-23',
          added: 45,
          prev_added: 30,
          maintained: 16,
          prev_maintained: 12,
          active_days: 6,
          open_source_added: 20,
          top_communities: [{ label: '触觉感知', added: 12 }, { label: 'VLA', added: 9 }],
          type_breakdown: { entity: 35, method: 10 },
        },
        '30d': {
          days: 30,
          start_date: '2026-10-25',
          end_date: '2026-11-23',
          added: 180,
          prev_added: 140,
          maintained: 65,
          prev_maintained: 50,
          active_days: 25,
          open_source_added: 70,
          top_communities: [{ label: 'VLA', added: 48 }],
          type_breakdown: { entity: 140, method: 40 },
        },
      },
      daily_recent: [
        { date: '2026-11-22', added: 8, maintained: 2 },
        { date: '2026-11-23', added: 10, maintained: 4 },
      ],
      monthly_recent: [{ month: '2026-11', added: 180, maintained: 65 }],
      recent_added_nodes: [
        {
          detail_id: 'wiki-concepts-wbc',
          label: '全身控制（WBC）',
          date: '2026-11-23',
          type: 'concept',
          community_label: '全身控制',
          has_repo: true,
        },
      ],
    },
  };

  // 1. Primary path: graph-stats.json includes node_trends
  const primary = setupDom({
    withFooter: true,
    fetchImpl: async (url) => {
      if (url === 'exports/graph-stats.json') {
        return { ok: true, status: 200, json: async () => sampleStatsWithTrends };
      }
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => mockCatalog };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  primary.window.eval(fabBlock);
  primary.window.eval(chatSource);
  primary.window.RNKbChat.open();

  await primary.window.RNKbChat.ask('最近7天本知识库节点趋势怎么样？');
  const msgZh = primary.window.document.querySelector('.kb-chat-msg--assistant');
  assert.ok(msgZh);
  assert.match(msgZh.textContent, /8200/);
  assert.match(msgZh.textContent, /45/);
  assert.match(msgZh.textContent, /触觉感知/);
  assert.ok(msgZh.querySelector('a[href="star-history.html#nodes-daily"]'));
  assert.ok(msgZh.querySelector('a[href="change-log.html"]'));
  assert.ok(msgZh.querySelector('a[href^="detail.html?id=wiki-concepts-wbc"]'));

  primary.window.RNKbChat.setLanguage('en');
  await primary.window.RNKbChat.ask('How many nodes were added in the last 30 days?');
  const msgsEn = primary.window.document.querySelectorAll('.kb-chat-msg--assistant');
  assert.match(msgsEn[1].textContent, /Last 30 days/);
  assert.match(msgsEn[1].textContent, /\+180/);
  primary.dom.window.close();

  // 2. Fallback path: graph-stats.json lacks node_trends -> computes from compact-v1 wiki-activity.json
  const compactActivity = {
    generated_at: '2026-11-23',
    source: 'git',
    format: 'compact-v1',
    node_index: {
      'wiki-entities-demo-1': {
        label: 'Demo Node 1',
        type: 'entity',
        path: 'wiki/entities/demo-1.md',
        has_repo: true,
        community_label: '触觉感知（Tactile） 社区',
      },
      'wiki-entities-demo-2': {
        label: 'Demo Node 2',
        type: 'method',
        path: 'wiki/methods/demo-2.md',
        community_label: 'VLA 社区',
      },
    },
    days: [
      { date: '2026-11-20', count: 1, nodes: [['wiki-entities-demo-1', 'added']] },
      { date: '2026-11-23', count: 2, nodes: [['wiki-entities-demo-2', 'added'], ['wiki-entities-demo-1', 'maintained']] },
    ],
  };
  const fallbackAct = setupDom({
    withFooter: true,
    fetchImpl: async (url) => {
      if (url === 'exports/graph-stats.json') {
        return {
          ok: true,
          status: 200,
          json: async () => ({ generated_at: '2026-11-23', node_count: 500, edge_count: 2000, community_count: 10 }),
        };
      }
      if (url === 'exports/wiki-activity.json') {
        return { ok: true, status: 200, json: async () => compactActivity };
      }
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => mockLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => mockCatalog };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  fallbackAct.window.eval(fabBlock);
  fallbackAct.window.eval(chatSource);
  fallbackAct.window.RNKbChat.open();

  await fallbackAct.window.RNKbChat.ask('最近本知识库有哪些节点趋势？');
  const actMsg = fallbackAct.window.document.querySelector('.kb-chat-msg--assistant');
  assert.ok(actMsg);
  assert.match(actMsg.textContent, /500/);
  assert.match(actMsg.textContent, /Demo Node 2/);
  assert.match(actMsg.textContent, /新增 2 个节点/);
  fallbackAct.dom.window.close();
});

test('Cloud node trends support trendGrounded AI answers, silent legacy-worker fallback, and 503 quota fallback', async () => {
  const sampleStats = {
    generated_at: '2026-11-23',
    node_count: 8200,
    edge_count: 67000,
    community_count: 35,
    node_trends: {
      as_of: '2026-11-23',
      latest_activity_date: '2026-11-23',
      total_nodes: 8200,
      total_edges: 67000,
      total_communities: 35,
      windows: {
        '7d': {
          days: 7,
          start_date: '2026-11-17',
          end_date: '2026-11-23',
          added: 45,
          prev_added: 30,
          maintained: 16,
          prev_maintained: 12,
          active_days: 6,
          open_source_added: 20,
          top_communities: [{ label: '触觉感知', added: 12 }],
          type_breakdown: { entity: 45 },
        },
      },
      daily_recent: [{ date: '2026-11-23', added: 10, maintained: 4 }],
      monthly_recent: [{ month: '2026-11', added: 45, maintained: 16 }],
      recent_added_nodes: [
        { detail_id: 'wiki-concepts-wbc', label: '全身控制 WBC', date: '2026-11-23', type: 'concept' },
      ],
    },
  };

  for (const modeCase of ['ai-ok', 'legacy-400', 'legacy-200', 'quota-503']) {
    const { dom, window } = setupDom({
      withFooter: true,
      config: {
        endpoint: 'https://kb-chat.example.workers.dev/chat',
        turnstileSiteKey: '1x00000000000000000000AA',
      },
      fetchImpl: async (url) => {
        if (url === 'https://kb-chat.example.workers.dev/chat') {
          if (modeCase === 'ai-ok') {
            return {
              ok: true,
              status: 200,
              json: async () => ({
                ok: true,
                mode: 'ai',
                grounded: true,
                trendGrounded: true,
                cached: false,
                answer: 'AI 趋势总结：近 7 天新增 45 个节点，见 [1] 与 [2]。',
                citations: [
                  {
                    index: 1,
                    id: 'page-star-history',
                    title: '知识库节点趋势统计',
                    heading: '近期增长',
                    snippet: '近 7 天新增 45 个节点。',
                    url: 'star-history.html#nodes-daily',
                  },
                  {
                    index: 2,
                    id: 'page-change-log',
                    title: '更新记录',
                    heading: '最新新增节点',
                    snippet: '2026-11-23 全身控制 WBC',
                    url: 'change-log.html',
                  },
                ],
              }),
            };
          }
          if (modeCase === 'legacy-400') {
            return {
              ok: false,
              status: 400,
              json: async () => ({
                ok: false,
                code: 'INVALID_PAYLOAD',
                fallback: 'local_search',
              }),
            };
          }
          if (modeCase === 'legacy-200') {
            return {
              ok: true,
              status: 200,
              json: async () => ({
                ok: true,
                mode: 'ungrounded',
                grounded: false,
                cached: false,
                answer: '站内暂未找到与该问题直接相关的内容。',
                citations: [],
              }),
            };
          }
          return {
            ok: false,
            status: 503,
            json: async () => ({
              ok: false,
              code: 'QUOTA_OR_AI_UNAVAILABLE',
              fallback: 'local_search',
            }),
          };
        }
        if (url === 'exports/graph-stats.json') return { ok: true, status: 200, json: async () => sampleStats };
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
      execute: () => window.__tsCallback(`tok-${modeCase}`),
    };
    window.eval(fabBlock);
    window.eval(chatSource);
    window.RNKbChat.open();

    await window.RNKbChat.ask('最近本知识库有哪些节点趋势？');
    const msg = window.document.querySelector('.kb-chat-msg--assistant');
    assert.ok(msg);
    const badgeMode = window.document.querySelector('.kb-chat-mode-badge').getAttribute('data-mode');
    const note = msg.querySelector('.kb-chat-status-note');
    if (modeCase === 'ai-ok') {
      assert.equal(badgeMode, 'ai');
      assert.equal(note, null);
      assert.match(msg.textContent, /AI 趋势总结：近 7 天新增 45 个节点/);
      assert.ok(msg.querySelector('a[href="star-history.html#nodes-daily"]'));
    } else if (modeCase === 'legacy-400' || modeCase === 'legacy-200') {
      assert.equal(badgeMode, 'local');
      assert.equal(note, null, 'legacy worker silently renders accurate local trend summary');
      assert.match(msg.textContent, /8200/);
      assert.match(msg.textContent, /45/);
    } else {
      assert.equal(badgeMode, 'fallback');
      assert.ok(note && /繁忙/.test(note.textContent));
      assert.match(msg.textContent, /45/);
    }
    dom.window.close();
  }
});

test('Fuzzy search handles English typos, separators, Chinese typos and skipped chars with safe history restore and zero false positives', async () => {
  const H_WBC = '1'.repeat(64);
  const H_DP = '2'.repeat(64);
  const H_SIM = '3'.repeat(64);
  const H_MPC = '4'.repeat(64);
  const H_HUM = '5'.repeat(64);
  const H_ISAAC = '6'.repeat(64);
  const H_PI0 = '7'.repeat(64);
  const H_DEX = '8'.repeat(64);

  const fuzzyLiteIndex = {
    meta: { version: 'v1', tier: 'lite', N: 8 },
    docs: [
      {
        id: 'wiki-concepts-wbc',
        path: 'wiki/concepts/whole-body-control.md',
        title: '全身控制（Whole-Body Control, WBC）',
        summary: '全身控制用于在多接触约束下协调人形机器人动力学与运动任务。',
        page_type: 'concept',
        tags: ['wbc', 'humanoid'],
        search_aliases: ['WBC', 'Whole-Body Control', '全身控制'],
      },
      {
        id: 'wiki-methods-diffusion-policy',
        path: 'wiki/methods/diffusion-policy.md',
        title: 'Diffusion Policy（扩散策略）',
        summary: 'Diffusion Policy 将机器人视觉运动策略建模为条件去噪扩散过程。',
        page_type: 'method',
        tags: ['diffusion', 'policy', 'manipulation'],
        search_aliases: ['Diffusion Policy', '扩散策略'],
      },
      {
        id: 'wiki-concepts-sim2real',
        path: 'wiki/concepts/sim2real.md',
        title: 'Sim2Real（仿真到真机）',
        summary: '通过域随机化与系统辨识缩小仿真到真机（Sim2Real）的性能差距。',
        page_type: 'concept',
        tags: ['sim2real', 'domain-randomization'],
        search_aliases: ['Sim2Real', '仿真到真机'],
      },
      {
        id: 'wiki-concepts-mpc',
        path: 'wiki/concepts/model-predictive-control.md',
        title: '模型预测控制（Model Predictive Control, MPC）',
        summary: '模型预测控制在有限时域内滚动优化机器人轨迹与接触力。',
        page_type: 'concept',
        tags: ['mpc', 'control'],
        search_aliases: ['MPC', '模型预测控制'],
      },
      {
        id: 'wiki-concepts-humanoid',
        path: 'wiki/concepts/humanoid-robot.md',
        title: '人形机器人（Humanoid Robot）',
        summary: '人形机器人结合双足动力学与人形控制实现全身协调移动与操作。',
        page_type: 'concept',
        tags: ['humanoid', 'biped'],
        search_aliases: ['人形机器人', '人形控制', 'Humanoid'],
      },
      {
        id: 'wiki-entities-isaaclab',
        path: 'wiki/entities/isaaclab.md',
        title: 'Isaac Lab',
        summary: 'IsaacLab 是 NVIDIA 基于 Isaac Sim 的模块化机器人学习仿真框架。',
        page_type: 'entity',
        tags: ['isaaclab', 'simulation'],
        search_aliases: ['IsaacLab', 'Isaac Lab'],
      },
      {
        id: 'wiki-entities-pi0',
        path: 'wiki/entities/pi0.md',
        title: 'π0（Pi0 Generalist Policy）',
        summary: 'Pi0 是面向跨本体机器人控制的视觉语言动作基础模型。',
        page_type: 'entity',
        tags: ['pi0', 'vla'],
        search_aliases: ['pi0', 'π0'],
      },
      {
        id: 'wiki-concepts-dexterous',
        path: 'wiki/concepts/dexterous-manipulation.md',
        title: '灵巧手操作（Dexterous Manipulation）',
        summary: '灵巧手通过多指触觉与力反馈完成精细抓握与工具使用。',
        page_type: 'concept',
        tags: ['dexterous', '灵巧手', 'manipulation'],
        search_aliases: ['灵巧手', 'Dexterous Hand'],
      },
    ],
  };

  const fuzzyCatalog = {
    version: 'v1',
    content_mode: 'per-page-v1',
    pages: {
      detail_pages: {
        'wiki-concepts-wbc': { title: '全身控制（Whole-Body Control, WBC）', content_url: `exports/page-content/${H_WBC}.json` },
        'wiki-methods-diffusion-policy': { title: 'Diffusion Policy（扩散策略）', content_url: `exports/page-content/${H_DP}.json` },
        'wiki-concepts-sim2real': { title: 'Sim2Real（仿真到真机）', content_url: `exports/page-content/${H_SIM}.json` },
        'wiki-concepts-mpc': { title: '模型预测控制（Model Predictive Control, MPC）', content_url: `exports/page-content/${H_MPC}.json` },
        'wiki-concepts-humanoid': { title: '人形机器人（Humanoid Robot）', content_url: `exports/page-content/${H_HUM}.json` },
        'wiki-entities-isaaclab': { title: 'Isaac Lab', content_url: `exports/page-content/${H_ISAAC}.json` },
        'wiki-entities-pi0': { title: 'π0（Pi0 Generalist Policy）', content_url: `exports/page-content/${H_PI0}.json` },
        'wiki-concepts-dexterous': { title: '灵巧手操作（Dexterous Manipulation）', content_url: `exports/page-content/${H_DEX}.json` },
      },
      roadmap_pages: {},
      page_aliases: {},
    },
  };

  const fuzzyBodies = {
    [`exports/page-content/${H_WBC}.json`]: {
      version: 'v1',
      id: 'wiki-concepts-wbc',
      content_markdown: '# 全身控制（WBC）\n\n## 核心原理\n\n全身控制（Whole-Body Control, WBC）通过分层 QP 协调人形机器人的质心与接触力约束。',
    },
    [`exports/page-content/${H_DP}.json`]: {
      version: 'v1',
      id: 'wiki-methods-diffusion-policy',
      content_markdown: '# Diffusion Policy\n\n## 核心机制\n\nDiffusion Policy 利用扩散去噪生成机器人多模态动作序列。',
    },
    [`exports/page-content/${H_SIM}.json`]: {
      version: 'v1',
      id: 'wiki-concepts-sim2real',
      content_markdown: '# Sim2Real（仿真到真机）\n\n## 域随机化\n\nSim2Real 通过域随机化缩小仿真到真机迁移差距。',
    },
    [`exports/page-content/${H_MPC}.json`]: {
      version: 'v1',
      id: 'wiki-concepts-mpc',
      content_markdown: '# 模型预测控制（MPC）\n\n## 滚动优化\n\n模型预测控制在有限时域内求解最优落足点与地面反力。',
    },
    [`exports/page-content/${H_HUM}.json`]: {
      version: 'v1',
      id: 'wiki-concepts-humanoid',
      content_markdown: '# 人形机器人\n\n## 人形控制\n\n人形机器人通过人形控制与双足步态规划维持动态平衡。',
    },
    [`exports/page-content/${H_ISAAC}.json`]: {
      version: 'v1',
      id: 'wiki-entities-isaaclab',
      content_markdown: '# Isaac Lab\n\n## 仿真引擎\n\nIsaacLab 支持 GPU 并行强化学习与模仿学习训练。',
    },
    [`exports/page-content/${H_PI0}.json`]: {
      version: 'v1',
      id: 'wiki-entities-pi0',
      content_markdown: '# π0（Pi0）\n\n## 通用策略\n\nPi0 基于流匹配（Flow Matching）生成高频连续机器人动作。',
    },
    [`exports/page-content/${H_DEX}.json`]: {
      version: 'v1',
      id: 'wiki-concepts-dexterous',
      content_markdown: '# 灵巧手操作\n\n## 触觉与抓握\n\n灵巧手依靠指尖触觉传感实现接触丰富的物体旋转与操作。',
    },
  };

  const { dom, window } = setupDom({
    withFooter: true,
    fetchImpl: async (url) => {
      if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => fuzzyLiteIndex };
      if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => fuzzyCatalog };
      if (fuzzyBodies[url]) return { ok: true, status: 200, json: async () => fuzzyBodies[url] };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  });
  window.eval(fabBlock);
  window.eval(chatSource);
  window.RNKbChat.open();

  const cases = [
    { q: 'difusion policy', expectedId: 'wiki-methods-diffusion-policy', chipPattern: /difusion → diffusion/ },
    { q: 'humnaoid wbc', expectedId: 'wiki-concepts-wbc', chipPattern: /humnaoid → humanoid/ },
    { q: 'sim2rael', expectedId: 'wiki-concepts-sim2real', chipPattern: /sim2rael → sim2real/ },
    { q: 'sim 2 real', expectedId: 'wiki-concepts-sim2real', chipPattern: /sim 2 real → sim2real/ },
    { q: 'isaac lab', expectedId: 'wiki-entities-isaaclab', chipPattern: /isaac lab → isaaclab/ },
    { q: 'pi 0', expectedId: 'wiki-entities-pi0', chipPattern: /pi 0 → pi0/ },
    { q: '什么是全神控制？', expectedId: 'wiki-concepts-wbc', chipPattern: /全神控制 → 全身控制/ },
    { q: '人行机器人', expectedId: 'wiki-concepts-humanoid', chipPattern: /人行机器人 → 人形机器人/ },
    { q: '什么是灵巧收', expectedId: 'wiki-concepts-dexterous', chipPattern: /灵巧收 → 灵巧手/ },
    { q: '请问人形控怎么实现？', expectedId: 'wiki-concepts-humanoid', chipPattern: /人形控 → 人形控制/ },
    { q: '仿真真机差距怎么缩小？', expectedId: 'wiki-concepts-sim2real', chipPattern: /仿真真机 → 仿真到真机/ },
    { q: '模预控有什么特点？', expectedId: 'wiki-concepts-mpc', chipPattern: /模预控 → 模型预测控制/ },
  ];

  for (let i = 0; i < cases.length; i += 1) {
    const tc = cases[i];
    await window.RNKbChat.ask(tc.q);
    const msgs = window.document.querySelectorAll('.kb-chat-msg--assistant');
    const lastMsg = msgs[msgs.length - 1];
    const fuzzyNote = lastMsg.querySelector('.kb-chat-fuzzy-note');
    assert.ok(fuzzyNote, `expected .kb-chat-fuzzy-note for "${tc.q}"`);
    assert.match(fuzzyNote.textContent, tc.chipPattern, `expected chip match for "${tc.q}"`);
    const link = lastMsg.querySelector(`a.kb-chat-citation-title[href*="id=${tc.expectedId}"]`);
    assert.ok(link, `expected citation link to ${tc.expectedId} for "${tc.q}"`);
  }

  // Exact queries should NOT render .kb-chat-fuzzy-note
  for (const exactQ of ['什么是全身控制 WBC？', 'How does Whole-Body Control work?']) {
    await window.RNKbChat.ask(exactQ);
    const msgs = window.document.querySelectorAll('.kb-chat-msg--assistant');
    const lastMsg = msgs[msgs.length - 1];
    assert.equal(lastMsg.querySelector('.kb-chat-fuzzy-note'), null, `no fuzzy banner for exact "${exactQ}"`);
    assert.ok(lastMsg.querySelector('a.kb-chat-citation-title[href*="id=wiki-concepts-wbc"]'));
  }

  // Unrelated queries must NOT trigger fuzzy matches or citations
  for (const offQ of [
    'completelyunrelatedtopicxyz',
    '今天晚饭吃什么比较健康',
    'what to eat for dinner today',
    'how to bake a chocolate cake',
  ]) {
    await window.RNKbChat.ask(offQ);
    const msgs = window.document.querySelectorAll('.kb-chat-msg--assistant');
    const lastMsg = msgs[msgs.length - 1];
    assert.equal(lastMsg.querySelector('.kb-chat-fuzzy-note'), null, `no fuzzy match for off-topic "${offQ}"`);
    assert.equal(lastMsg.querySelectorAll('.kb-chat-citation-group').length, 0, `no citations for "${offQ}"`);
    assert.ok(lastMsg.querySelector('a[href="graph.html"]'));
  }

  // Reload persistence preserves fuzzyMatches chip
  const savedHistory = window.localStorage.getItem('rn_kb_chat_history_v1');
  dom.window.close();

  const restored = setupDom({ withFooter: true });
  restored.window.localStorage.setItem('rn_kb_chat_history_v1', savedHistory);
  restored.window.eval(fabBlock);
  restored.window.eval(chatSource);
  restored.window.RNKbChat.open();
  const restoredNotes = restored.window.document.querySelectorAll('.kb-chat-fuzzy-note');
  assert.equal(restoredNotes.length, cases.length);
  assert.match(restoredNotes[0].textContent, /difusion → diffusion/);
  restored.dom.window.close();
});

test('Cloud mode renders sanitized fuzzyMatches and falls back cleanly when legacy Worker returns ungrounded', async () => {
  const HASH_DP = '9'.repeat(64);
  const dpLite = {
    meta: { version: 'v1', tier: 'lite', N: 1 },
    docs: [
      {
        id: 'wiki-methods-diffusion-policy',
        path: 'wiki/methods/diffusion-policy.md',
        title: 'Diffusion Policy（扩散策略）',
        summary: 'Diffusion Policy 将机器人视觉运动策略建模为条件去噪扩散过程。',
        page_type: 'method',
        tags: ['diffusion', 'policy'],
        search_aliases: ['Diffusion Policy'],
      },
    ],
  };
  const dpCatalog = {
    version: 'v1',
    content_mode: 'per-page-v1',
    pages: {
      detail_pages: {
        'wiki-methods-diffusion-policy': {
          title: 'Diffusion Policy（扩散策略）',
          content_url: `exports/page-content/${HASH_DP}.json`,
        },
      },
      roadmap_pages: {},
      page_aliases: {},
    },
  };
  const dpBody = {
    version: 'v1',
    id: 'wiki-methods-diffusion-policy',
    content_markdown: '# Diffusion Policy\n\n## 核心机制\n\nDiffusion Policy 通过条件扩散模型迭代去噪生成动作序列。',
  };

  for (const scenario of ['cloud-fuzzy', 'legacy-ungrounded']) {
    const { dom, window } = setupDom({
      withFooter: true,
      config: {
        endpoint: 'https://kb-chat.example.workers.dev/chat',
        turnstileSiteKey: '1x00000000000000000000AA',
      },
      fetchImpl: async (url) => {
        if (url === 'https://kb-chat.example.workers.dev/chat') {
          if (scenario === 'cloud-fuzzy') {
            return {
              ok: true,
              status: 200,
              json: async () => ({
                ok: true,
                mode: 'ai',
                grounded: true,
                cached: false,
                answer: 'Diffusion Policy 利用条件扩散模型生成动作 [1]。',
                fuzzyMatches: [{ from: '<script>window.__fuzzyXss=1</script>difusion', to: 'diffusion' }],
                citations: [
                  {
                    index: 1,
                    id: 'wiki-methods-diffusion-policy',
                    title: 'Diffusion Policy（扩散策略）',
                    heading: '核心机制',
                    snippet: 'Diffusion Policy 通过条件扩散模型迭代去噪生成动作序列。',
                    url: 'detail.html?id=wiki-methods-diffusion-policy',
                  },
                ],
              }),
            };
          }
          return {
            ok: true,
            status: 200,
            json: async () => ({
              ok: true,
              mode: 'ungrounded',
              grounded: false,
              cached: false,
              answer: 'No direct match found in the wiki.',
              citations: [],
            }),
          };
        }
        if (url === 'search-index-lite.json') return { ok: true, status: 200, json: async () => dpLite };
        if (url === 'exports/site-catalog-v1.json') return { ok: true, status: 200, json: async () => dpCatalog };
        if (url === `exports/page-content/${HASH_DP}.json`) return { ok: true, status: 200, json: async () => dpBody };
        return { ok: false, status: 404, json: async () => ({}) };
      },
    });
    window.turnstile = {
      render: (_el, opts) => {
        window.__tsCallback = opts.callback;
        return 'wid-1';
      },
      reset: () => {},
      execute: () => window.__tsCallback(`tok-${scenario}`),
    };
    window.eval(fabBlock);
    window.eval(chatSource);
    window.RNKbChat.open();

    await window.RNKbChat.ask('What is difusion?');
    const msg = window.document.querySelector('.kb-chat-msg--assistant');
    assert.ok(msg);
    const fuzzyNote = msg.querySelector('.kb-chat-fuzzy-note');
    assert.ok(fuzzyNote, `fuzzy note visible in ${scenario}`);
    assert.match(fuzzyNote.textContent, /difusion.*→ diffusion/);
    assert.equal(window.__fuzzyXss, undefined);
    assert.equal(msg.querySelectorAll('script').length, 0);
    assert.ok(msg.querySelector('a.kb-chat-citation-title[href*="id=wiki-methods-diffusion-policy"]'));
    if (scenario === 'legacy-ungrounded') {
      assert.equal(window.document.querySelector('.kb-chat-mode-badge').getAttribute('data-mode'), 'local');
    }
    dom.window.close();
  }
});
