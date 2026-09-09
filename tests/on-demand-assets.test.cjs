// 详情/路线页按内容加载 KaTeX 与 Mermaid：没有公式、没有图表就不请求组件（编号 9）
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../docs/main.js'), 'utf8');
const loaderStart = source.indexOf('  // ── 公式与图表组件按内容加载（编号 9）──');
const loaderEnd = source.indexOf('  var mermaidLightboxEl = null;');
const mathStart = source.indexOf('  function renderDetailMath(container) {');
const mathEnd = source.indexOf('  function slugifyHeading(text) {');
assert.ok(loaderStart >= 0 && loaderEnd > loaderStart, 'on-demand loader block is located');
assert.ok(mathStart >= 0 && mathEnd > mathStart, 'math renderer is located');

const KATEX_CSS = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css';
const KATEX_JS = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js';
const KATEX_AUTO = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js';
const MERMAID_JS = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
const flush = () => new Promise(setImmediate);

function harness(options = {}) {
  const failing = new Set(options.failing || []);
  const appended = [];
  const runs = [];
  const head = {
    appendChild(el) {
      const url = el.href || el.src;
      appended.push(url);
      setImmediate(() => {
        if (failing.has(url)) { el.onerror(); return; }
        if (url === MERMAID_JS) context.window.mermaid = { run: (arg) => { runs.push(arg); return Promise.resolve(); } };
        if (url === KATEX_AUTO) context.window.renderMathInElement = (el2, opts) => runs.push({ math: el2, opts });
        el.onload();
      });
      el.parentNode = head;
    },
    removeChild() {},
  };
  const context = vm.createContext({
    Promise, Array, console: { warn() {} },
    window: {},
    document: {
      createElement: () => ({ onload: null, onerror: null, parentNode: null }),
      head,
      body: { appendChild() {}, removeChild() {} },
    },
    mermaidSourceForCurrentBrowser: (s) => s,
    initializeMermaidRenderer() {},
    getMermaidFontSizePx: () => 14,
    patchMermaidSvgLabelOverflow() {},
    enhanceMermaidZoomTargets() {},
    bindMermaidZoom() {},
  });
  vm.runInContext(source.slice(loaderStart, loaderEnd), context);
  vm.runInContext(source.slice(mathStart, mathEnd), context);
  return { context, appended, runs };
}

const plain = () => ({ textContent: '纯文字页面，没有公式', querySelectorAll: () => [] });
const withMath = () => ({ textContent: '推导：$$a^2 + b^2 = c^2$$', querySelectorAll: () => [] });
const mermaidNode = () => ({
  getAttribute: () => null, setAttribute() {}, removeAttribute() {}, textContent: 'graph TD; A-->B',
});
const withDiagram = (node) => ({
  textContent: '流程图',
  querySelectorAll: (selector) => (selector === '.mermaid' ? [node] : []),
});

test('a page with neither formulas nor diagrams downloads neither component', async () => {
  const h = harness();
  await h.context.renderDetailMath(plain());
  await h.context.renderDetailMermaid(plain());
  await flush();
  assert.deepEqual(h.appended, []);
});

test('formula content loads KaTeX in dependency order and only once', async () => {
  const h = harness();
  await h.context.renderDetailMath(withMath());
  assert.deepEqual(h.appended, [KATEX_CSS, KATEX_JS, KATEX_AUTO]);
  assert.equal(h.runs.length, 1, 'math is rendered after the component arrives');
  await h.context.renderDetailMath(withMath());
  assert.equal(h.appended.length, 3, 'an already loaded component is not requested again');
});

test('diagram content loads mermaid once and then renders the nodes', async () => {
  const h = harness();
  const node = mermaidNode();
  await h.context.renderDetailMermaid(withDiagram(node));
  assert.deepEqual(h.appended, [MERMAID_JS]);
  assert.equal(h.runs.length, 1);
  assert.equal(h.runs[0].nodes[0], node, 'the existing .mermaid nodes are handed to the renderer');
  await h.context.renderDetailMermaid(withDiagram(node));
  assert.equal(h.appended.length, 1);
  assert.equal(h.runs.length, 2, 'rendering continues with the loaded component');
});

test('a failed component load degrades quietly and is retried on the next render', async () => {
  const h = harness({ failing: [MERMAID_JS] });
  await h.context.renderDetailMermaid(withDiagram(mermaidNode()));
  assert.deepEqual(h.appended, [MERMAID_JS]);
  assert.equal(h.runs.length, 0);
  h.context.window.mermaid = undefined;
  const retried = harness();
  await retried.context.renderDetailMermaid(withDiagram(mermaidNode()));
  assert.equal(retried.runs.length, 1);
});

test('the page shells no longer ship KaTeX or Mermaid unconditionally', () => {
  for (const page of ['../docs/detail.html', '../docs/roadmap.html']) {
    const html = readFileSync(resolve(__dirname, page), 'utf8');
    assert.ok(!/katex/i.test(html), `${page} must not preload KaTeX`);
    assert.ok(!/cdn\.jsdelivr\.net.*mermaid/i.test(html), `${page} must not preload Mermaid`);
  }
});
