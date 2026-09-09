// 图谱动态渲染性能测量（plan.md「2026-09-08 图谱动态渲染性能专项」G0 基线 + G1/G2/G3 前后对比）。
// 分离记录：力 tick / SVG 坐标写入 / 社区标签 / 筛选，刷新布局的长任务与帧耗时，
// 3D draw calls / 三角形数 / 场景对象数，以及 3D 悬停刷新与标签同步耗时。
// 用法：node scripts/measure_graph_perf.cjs [baseUrl] [outJson]
//   先在 docs/ 下 `python3 -m http.server 8765`（数据需 `make export graph`）。
// 注意：headless + 软件 WebGL，GPU 帧率不可外推到真机；结论见 docs/checklists/graph-perf-baseline-g0.md。
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const REPEATS = 5;          // 每场景重复次数（计划要求 ≥3）
const TICK_SAMPLES = 20;    // 单次采样内的 tick 次数

function stats(arr) {
  const a = [...arr].sort((x, y) => x - y);
  const q = p => a[Math.min(a.length - 1, Math.floor(p * a.length))];
  return {
    n: a.length,
    min: +a[0].toFixed(2),
    p50: +q(0.5).toFixed(2),
    p95: +q(0.95).toFixed(2),
    max: +a[a.length - 1].toFixed(2),
    mean: +(a.reduce((s, v) => s + v, 0) / a.length).toFixed(2),
  };
}

(async () => {
  const baseUrl = process.argv[2] || 'http://127.0.0.1:8765/graph.html';
  const outJson = path.resolve(
    process.argv[3] || path.join(__dirname, '..', '.cursor-artifacts', 'graph-perf-baseline.json')
  );
  fs.mkdirSync(path.dirname(outJson), { recursive: true });

  // 固定数据哈希：所有对比必须基于同一份 link-graph.json
  const graphJson = path.join(__dirname, '..', 'exports', 'link-graph.json');
  const graphSha = fs.existsSync(graphJson)
    ? crypto.createHash('sha256').update(fs.readFileSync(graphJson)).digest('hex').slice(0, 16)
    : null;

  const browser = await puppeteer.launch({
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || '/opt/pw-browsers/chromium',
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--window-size=1440,900'],
  });

  const out = { measured_at: new Date().toISOString(), base_url: baseUrl, graph_json_sha256_16: graphSha };

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    await page.setCacheEnabled(false);
    const errors = [];
    page.on('pageerror', e => errors.push(String(e.message)));

    // ── 环境 ──
    out.env = {
      browser: await browser.version(),
      node: process.version,
      os: `${os.type()} ${os.release()}`,
      cpu_model: (os.cpus()[0] || {}).model || 'unknown',
      cpu_logical_cores: os.cpus().length,
      viewport: '1440x900',
      device_pixel_ratio: 1,
      cpu_throttling: 'none',
      cache: 'cold (setCacheEnabled=false)',
      note: 'headless Chromium + 软件光栅/SwiftShader；真机 GPU、手机与 Safari 未测',
    };

    // ── 首屏加载 ──
    const t0 = Date.now();
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForFunction(() => {
      const l = document.getElementById('graph-loading');
      const c = document.getElementById('graph-node-count');
      const hidden = !l || l.hidden || l.classList.contains('is-hidden')
        || window.getComputedStyle(l).display === 'none';
      return hidden && c && c.textContent && !c.textContent.includes('加载中');
    }, { timeout: 180000 });
    out.first_paint_ready_ms = Date.now() - t0;

    await page.waitForFunction(
      () => window.__RN_GRAPH2D_DEBUG__ && window.__RN_GRAPH2D_DEBUG__.simNodeCount,
      { timeout: 30000 }
    );
    await new Promise(r => setTimeout(r, 3000)); // 让首屏力布局自然落稳

    // ── 结构计数（精确、与硬件无关）──
    out.structure = await page.evaluate(() => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      const svg = document.getElementById('graph-canvas');
      const lines = svg ? svg.querySelectorAll('.edges line').length : -1;
      const nodeG = svg ? svg.querySelectorAll('.nodes g.node-g').length : -1;
      return {
        sim_nodes: dbg.simNodeCount(),
        sim_links: dbg.simLinkCount(),
        dom_link_lines: lines,
        dom_node_groups: nodeG,
        dom_community_labels: svg ? svg.querySelectorAll('g.community-label').length : -1,
        dom_total_elements: svg ? svg.getElementsByTagName('*').length : -1,
        // syncGraphDomFromSimulation：每边 4 个坐标属性 + 每节点 1 个 transform
        attr_writes_per_tick: lines * 4 + nodeG,
      };
    });

    // ── 主线程耗时分段（停掉模拟，避免自身 tick 干扰）──
    out.cost_split_ms = await page.evaluate(async (REPEATS, TICK_SAMPLES) => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      dbg.stop();
      const collect = async (fn, repeats) => {
        const xs = [];
        for (let r = 0; r < repeats; r++) {
          const t = performance.now();
          fn();
          xs.push(performance.now() - t);
          await new Promise(res => requestAnimationFrame(res));
        }
        return xs;
      };
      return {
        force_tick: await collect(() => dbg.tickOnce(), TICK_SAMPLES),
        sync_dom_total: await collect(() => dbg.syncDom(), TICK_SAMPLES),
        community_labels: await collect(() => dbg.syncCommunityLabels(), TICK_SAMPLES),
        apply_filters: await collect(() => dbg.applyFilters(), REPEATS),
      };
    }, REPEATS, TICK_SAMPLES);
    for (const k of Object.keys(out.cost_split_ms)) {
      out.cost_split_ms[k] = stats(out.cost_split_ms[k]);
    }
    // 派生：边/节点坐标写入 ≈ 全量 DOM 同步 − 社区标签
    out.cost_split_ms.derived_link_node_attrs_p50 =
      +(out.cost_split_ms.sync_dom_total.p50 - out.cost_split_ms.community_labels.p50).toFixed(2);

    // ── 刷新布局：长任务与帧耗时 ──
    await page.click('#physics-toggle');
    await page.waitForSelector('#physics-panel:not([hidden])');
    const refreshRuns = [];
    for (let r = 0; r < 3; r++) {
      const res = await page.evaluate(async () => {
        const longTasks = [];
        const po = new PerformanceObserver(list => {
          for (const e of list.getEntries()) longTasks.push(+e.duration.toFixed(1));
        });
        try { po.observe({ entryTypes: ['longtask'] }); } catch (e) { /* 不支持则留空 */ }

        const frames = [];
        let last = performance.now();
        let running = true;
        const loop = () => {
          const now = performance.now();
          frames.push(now - last);
          last = now;
          if (running) requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);

        const t0 = performance.now();
        document.getElementById('restart-simulation').click();
        await new Promise(res => setTimeout(res, 4000));
        running = false;
        po.disconnect();
        frames.shift(); // 首帧含点击前的间隔
        return { elapsed_ms: +(performance.now() - t0).toFixed(1), longTasks, frames };
      });
      refreshRuns.push(res);
      await new Promise(r2 => setTimeout(r2, 2500));
    }
    out.refresh_layout = refreshRuns.map(r => ({
      elapsed_ms: r.elapsed_ms,
      long_tasks_over_50ms: r.longTasks.length,
      long_task_total_ms: +r.longTasks.reduce((s, v) => s + v, 0).toFixed(1),
      long_task_max_ms: r.longTasks.length ? Math.max(...r.longTasks) : 0,
      frame_ms: stats(r.frames),
      frames_over_33ms: r.frames.filter(f => f > 33.3).length,
      frame_count: r.frames.length,
    }));

    // ── 筛选是否生成真实子图（验证计划中「普通筛选仍计算全图」）──
    out.filter_subgraph_check = await page.evaluate(async () => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      const sl = document.getElementById('sl-degree-top');
      const before = { sim_nodes: dbg.simNodeCount(), sim_links: dbg.simLinkCount() };
      if (!sl) return { error: 'sl-degree-top 不存在' };
      const max = Number(sl.max);
      sl.value = String(Math.min(300, max));
      sl.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise(r => setTimeout(r, 1500));
      const svg = document.getElementById('graph-canvas');
      return {
        slider_max: max,
        set_to: sl.value,
        label: (document.getElementById('val-degree-top') || {}).textContent,
        before,
        after: { sim_nodes: dbg.simNodeCount(), sim_links: dbg.simLinkCount() },
        dom_after: {
          link_lines: svg.querySelectorAll('.edges line').length,
          node_groups: svg.querySelectorAll('.nodes g.node-g').length,
        },
      };
    });

    // ── 筛选态下的每 tick 成本与 applyFilters 重算成本（G3 的直接验收口径）──
    out.cost_split_filtered_ms = await page.evaluate(async (REPEATS, TICK_SAMPLES) => {
      const dbg = window.__RN_GRAPH2D_DEBUG__;
      dbg.stop();
      const collect = async (fn, repeats) => {
        const xs = [];
        for (let r = 0; r < repeats; r++) {
          const t = performance.now();
          fn();
          xs.push(performance.now() - t);
          await new Promise(res => requestAnimationFrame(res));
        }
        return xs;
      };
      return {
        force_tick: await collect(() => dbg.tickOnce(), TICK_SAMPLES),
        sync_dom_total: await collect(() => dbg.syncDom(), TICK_SAMPLES),
        apply_filters: await collect(() => dbg.applyFilters(), REPEATS),
      };
    }, REPEATS, TICK_SAMPLES);
    for (const k of Object.keys(out.cost_split_filtered_ms)) {
      out.cost_split_filtered_ms[k] = stats(out.cost_split_filtered_ms[k]);
    }

    // ── 滑块连续输入：模拟主线程被阻塞时排队的 input 事件一次性回放 ──
    out.slider_burst = await page.evaluate(async () => {
      const sl = document.getElementById('sl-degree-top');
      if (!sl) return { error: 'sl-degree-top 不存在' };
      const max = Number(sl.max);
      const longTasks = [];
      const po = new PerformanceObserver(list => {
        for (const e of list.getEntries()) longTasks.push(+e.duration.toFixed(1));
      });
      try { po.observe({ entryTypes: ['longtask'] }); } catch (e) { /* 不支持则留空 */ }

      const BURST = 12;
      const t0 = performance.now();
      for (let i = 0; i < BURST; i++) {
        sl.value = String(Math.max(10, Math.round(max * (1 - i / BURST))));
        sl.dispatchEvent(new Event('input', { bubbles: true }));
      }
      const dispatchMs = performance.now() - t0;      // 事件派发本身占用的主线程时间
      await new Promise(r => setTimeout(r, 3000));    // 等合并后的重活跑完
      po.disconnect();
      const totalMs = performance.now() - t0;
      sl.value = sl.max;
      sl.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise(r => setTimeout(r, 1500));
      return {
        burst_events: BURST,
        dispatch_blocking_ms: +dispatchMs.toFixed(1),
        total_ms: +totalMs.toFixed(1),
        long_tasks_over_50ms: longTasks.length,
        long_task_total_ms: +longTasks.reduce((s, v) => s + v, 0).toFixed(1),
        long_task_max_ms: longTasks.length ? Math.max(...longTasks) : 0,
      };
    });

    // ── 3D：draw calls / 三角形数（先恢复全图，避免沿用上一步的 Top N 筛选）──
    await page.evaluate(async () => {
      const sl = document.getElementById('sl-degree-top');
      if (!sl) return;
      sl.value = sl.max;
      sl.dispatchEvent(new Event('input', { bubbles: true }));
      sl.dispatchEvent(new Event('change', { bubbles: true }));
      await new Promise(r => setTimeout(r, 1500));
    });
    out.filter_reset_ok = await page.evaluate(() => ({
      label: (document.getElementById('val-degree-top') || {}).textContent,
      visible_node_text: (document.getElementById('graph-node-count') || {}).textContent,
    }));

    try {
      await page.click('#view-mode-3d');
      await page.waitForFunction(() => !!window.__RN_GRAPH3D_VIEW__, { timeout: 60000 });
      await new Promise(r => setTimeout(r, 6000));
      out.three_d = await page.evaluate(async () => {
        const v = window.__RN_GRAPH3D_VIEW__;
        if (!v || typeof v.getRendererInfo !== 'function') {
          return { available: false, note: '未取到 three.js renderer；draw calls/三角形数未测' };
        }
        // renderer.info.render 每帧自动重置，取连续几帧的稳态值
        const samples = [];
        for (let i = 0; i < 5; i++) {
          await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
          const info = v.getRendererInfo();
          if (info) samples.push(info);
        }
        if (!samples.length) return { available: false, note: 'getRendererInfo 返回 null' };
        const last = samples[samples.length - 1];
        const cv = document.querySelector('#graph-3d canvas, canvas');
        return {
          available: true,
          per_frame_samples: samples.map(s => ({ draw_calls: s.draw_calls, triangles: s.triangles, lines: s.lines })),
          draw_calls: last.draw_calls,
          triangles: last.triangles,
          lines: last.lines,
          points: last.points,
          geometries: last.geometries,
          textures: last.textures,
          programs: last.programs,
          pixel_ratio: last.pixel_ratio,
          canvas: cv ? cv.width + 'x' + cv.height : null,
          scene: typeof v.getSceneStats === 'function' ? v.getSceneStats() : null,
        };
      });

      // 3D 悬停刷新 / 标签同步耗时（G2 的直接验收口径）
      out.three_d_interaction_ms = await page.evaluate(async (REPEATS) => {
        const v = window.__RN_GRAPH3D_VIEW__;
        if (!v || typeof v.measureHover !== 'function') return { available: false };
        const ids = (window.__RN_GRAPH2D_DEBUG__ && window.__RN_GRAPH2D_DEBUG__.topNodeIds)
          ? window.__RN_GRAPH2D_DEBUG__.topNodeIds(REPEATS)
          : [];
        const hoverOn = [];
        const hoverOff = [];
        for (let i = 0; i < REPEATS; i++) {
          const id = ids[i % Math.max(1, ids.length)] || null;
          hoverOn.push(v.measureHover(id));
          await new Promise(res => requestAnimationFrame(res));
          hoverOff.push(v.measureHover(null));
          await new Promise(res => requestAnimationFrame(res));
        }
        const labels = [];
        for (let i = 0; i < REPEATS; i++) {
          labels.push(v.measureSyncLabels());
          await new Promise(res => requestAnimationFrame(res));
        }
        return { available: true, hover_on: hoverOn, hover_off: hoverOff, sync_labels: labels };
      }, REPEATS);
      if (out.three_d_interaction_ms.available) {
        for (const k of ['hover_on', 'hover_off', 'sync_labels']) {
          out.three_d_interaction_ms[k] = stats(out.three_d_interaction_ms[k]);
        }
      }
    } catch (e) {
      out.three_d = { available: false, error: String(e.message) };
    }

    out.page_errors = errors;
  } finally {
    await browser.close();
  }

  fs.writeFileSync(outJson, JSON.stringify(out, null, 2) + '\n');
  console.log(JSON.stringify(out, null, 2));
  console.log('\n→ 写入 ' + outJson);
})().catch(e => { console.error('FAIL:', e.stack || e.message); process.exit(1); });
