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
const MERMAID_JS = 'https://cdn.jsdelivr.net/npm/mermaid@10.9.5/dist/mermaid.min.js';
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

// 视口外的图先不画：滚近（IntersectionObserver 触发）才渲染，且每批最多 3 张
function lazyHarness(positions) {
  const h = harness();
  const observed = [];
  let callback = null;
  h.context.IntersectionObserver = class {
    constructor(cb) { callback = cb; }
    observe(node) { observed.push(node); }
    unobserve(node) { const i = observed.indexOf(node); if (i >= 0) observed.splice(i, 1); }
  };
  h.context.window.innerHeight = 900;
  h.context.window.setTimeout = (fn) => setImmediate(fn);
  h.context.WeakMap = WeakMap;
  const nodes = positions.map((top) => ({
    ...mermaidNode(),
    getBoundingClientRect: () => ({ top, bottom: top + 200, width: 600, height: 200 }),
  }));
  const container = withDiagram(null);
  container.querySelectorAll = (selector) => (selector === '.mermaid' ? nodes : []);
  const fire = (targets) => callback(targets.map((target) => ({ target, isIntersecting: true })));
  return { h, nodes, container, observed, fire };
}

test('diagrams far below the viewport wait until they scroll near', async () => {
  const { h, nodes, container, observed, fire } = lazyHarness([100, 400, 3000, 6000]);
  await h.context.renderDetailMermaid(container);
  await flush();
  assert.equal(h.runs.length, 1);
  // runs 里的数组来自 vm 作用域（另一 realm），逐项比较引用
  assert.deepEqual([...h.runs[0].nodes], [nodes[0], nodes[1]], 'only the two near-viewport diagrams render at once');
  assert.deepEqual(observed, [nodes[2], nodes[3]], 'the far ones are observed instead');
  fire([nodes[2]]);
  await flush();
  assert.equal(h.runs.length, 2);
  assert.deepEqual([...h.runs[1].nodes], [nodes[2]]);
  assert.deepEqual(observed, [nodes[3]], 'a rendered diagram is no longer observed');
});

test('a collapsed (display:none) diagram is deferred, not rendered blind', async () => {
  const { h, nodes, container, observed } = lazyHarness([0]);
  nodes[0].getBoundingClientRect = () => ({ top: 0, bottom: 0, width: 0, height: 0 });
  await h.context.renderDetailMermaid(container);
  await flush();
  assert.equal(h.runs.length, 0);
  assert.deepEqual(observed, [nodes[0]]);
});

test('many diagrams scrolling into view at once render in batches of three', async () => {
  const { h, container, fire, nodes } = lazyHarness([5000, 5300, 5600, 5900, 6200]);
  await h.context.renderDetailMermaid(container);
  await flush();
  assert.equal(h.runs.length, 0, 'nothing is near the viewport yet');
  fire(nodes);
  for (let i = 0; i < 6; i++) await flush();
  assert.equal(h.runs.length, 2);
  assert.equal(h.runs[0].nodes.length, 3);
  assert.equal(h.runs[1].nodes.length, 2);
});

test('without IntersectionObserver every diagram still renders immediately', async () => {
  const h = harness();
  const nodes = [0, 5000].map((top) => ({
    ...mermaidNode(),
    getBoundingClientRect: () => ({ top, bottom: top + 200, width: 600, height: 200 }),
  }));
  const container = withDiagram(null);
  container.querySelectorAll = (selector) => (selector === '.mermaid' ? nodes : []);
  await h.context.renderDetailMermaid(container);
  assert.equal(h.runs.length, 1);
  assert.deepEqual([...h.runs[0].nodes], nodes);
});

test('the page shells no longer ship KaTeX or Mermaid unconditionally', () => {
  for (const page of ['../docs/detail.html', '../docs/roadmap.html']) {
    const html = readFileSync(resolve(__dirname, page), 'utf8');
    assert.ok(!/katex/i.test(html), `${page} must not preload KaTeX`);
    assert.ok(!/cdn\.jsdelivr\.net.*mermaid/i.test(html), `${page} must not preload Mermaid`);
  }
});
