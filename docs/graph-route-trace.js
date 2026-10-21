(function () {
  'use strict';

  const distance = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);

  // 直线段按屏幕长度累计：长短边之间保持匀速。pairs 中 cont 表示接续上一段（同一条折线），
  // 端点为 null（节点被筛掉）的段留缺口，不把前后节点误接起来。
  function geometry(pairs) {
    const segments = [];
    let total = 0;
    pairs.forEach(([a, b, cont], idx) => {
      if (!a || !b) return;
      const length = distance(a, b);
      if (length < 0.01) return;
      segments.push({ a, b, length, start: total, idx, cont: !!cont });
      total += length;
    });
    return { segments, total };
  }
  const xy = p => p.x.toFixed(2) + ',' + p.y.toFixed(2);
  function slice(g, from, to) {
    let d = '', tip = null, lastIdx = -2;
    for (const s of g.segments) {
      if (s.start >= to || s.start + s.length <= from) continue;
      const t0 = Math.max(0, from - s.start) / s.length;
      const t1 = Math.min(s.length, to - s.start) / s.length;
      const p0 = { x: s.a.x + (s.b.x - s.a.x) * t0, y: s.a.y + (s.b.y - s.a.y) * t0 };
      const p1 = { x: s.a.x + (s.b.x - s.a.x) * t1, y: s.a.y + (s.b.y - s.a.y) * t1 };
      if (!(s.cont && s.idx === lastIdx + 1) || t0 > 0) d += 'M' + xy(p0);
      d += 'L' + xy(p1);
      tip = p1;
      lastIdx = s.idx;
    }
    return { d, tip };
  }

  // 亮起节点比正文内链多（路线还按 ids / 社区 / 关键词命中）。从主链出发沿真实邻接边宽度优先，
  // 每个未覆盖节点接到发现它的节点；与亮起集合无边相连的，退而接到主链中同社区或起点。
  function branches(order, lit, adjacency, communityOf) {
    const reached = new Set(order.filter(id => lit.has(id)));
    const queue = [...reached], out = [];
    for (let i = 0; i < queue.length; i++) {
      const neighbors = Array.from(adjacency.get(queue[i]) || []).sort();
      for (const id of neighbors) {
        if (!lit.has(id) || reached.has(id)) continue;
        reached.add(id); queue.push(id); out.push([queue[i], id]);
      }
    }
    const byCommunity = new Map();
    order.forEach(id => { const c = communityOf(id); if (c && !byCommunity.has(c)) byCommunity.set(c, id); });
    Array.from(lit).sort().forEach(id => {
      if (reached.has(id)) return;
      out.push([byCommunity.get(communityOf(id)) || order[0], id]);
      reached.add(id);
    });
    return out;
  }

  // 独立叙事层：主题蓝直线、柔光尾迹、流动微光，不向知识图或力模拟加边。
  function create(options) {
    const svg = options.svg;
    const ns = 'http://www.w3.org/2000/svg';
    const make = (tag, attrs) => {
      const el = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      svg.appendChild(el);
      return el;
    };
    const line = (name, width, opacity, color) => make('path', {
      class: 'route-trace-' + name, fill: 'none', stroke: color || 'var(--route-trace-color, var(--accent, #5b9cf6))',
      'stroke-width': width, 'stroke-opacity': opacity,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
    });
    const halo = line('halo', 8, 0.08);
    const core = line('core', 1.5, 0.58);
    const flow = line('flow', 2, 0.56, 'var(--route-trace-glint, #d8e8fd)');
    flow.setAttribute('stroke-dasharray', '2 28');
    const tail = line('tail', 2.2, 0.95, 'var(--route-trace-glint, #d8e8fd)');
    const start = make('circle', { class: 'route-trace-start', r: 6, fill: 'none',
      stroke: 'var(--route-trace-color, var(--accent, #5b9cf6))', 'stroke-width': 1.5, 'stroke-opacity': 0.75 });
    const headGlow = make('circle', { r: 8, fill: 'var(--route-trace-color, var(--accent, #5b9cf6))', 'fill-opacity': 0.16 });
    const head = make('circle', { class: 'route-trace-head', r: 3, fill: 'var(--route-trace-glint, #d8e8fd)' });
    let ids = [], branches = [], progress = 0, startedAt = null, raf = 0, duration = 0;
    let cachedPoints = [], cachedGeometry = geometry([]);
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    function position(el, p) {
      el.style.display = p ? '' : 'none';
      if (p) { el.setAttribute('cx', p.x); el.setAttribute('cy', p.y); }
    }
    function point(id) {
      if (!options.isVisible(id)) return null;
      const p = options.project(id);
      return p && Number.isFinite(p.x) && Number.isFinite(p.y) ? p : null;
    }
    function render() {
      const on = (ids.length > 1 || branches.length) && options.isActive();
      svg.style.display = on ? '' : 'none';
      svg.classList.toggle('route-trace-reduced', reducedMotion.matches);
      if (!on) return;
      const points = ids.map(point);
      const ends = branches.map(([parent, child]) => [point(parent), point(child)]);
      const flat = points.concat(...ends);
      if (flat.length !== cachedPoints.length || flat.some((p, i) =>
        !!p !== !!cachedPoints[i] || p && (p.x !== cachedPoints[i].x || p.y !== cachedPoints[i].y))) {
        cachedPoints = flat;
        cachedGeometry = geometry(points.slice(1).map((b, i) => [points[i], b, true])
          .concat(ends.map(([a, b]) => [a, b, false])));
      }
      // 起落柔和，中段近匀速；始终沿同一条连续曲线前进。
      const eased = progress * progress * (3 - 2 * progress);
      const end = eased * cachedGeometry.total;
      const visible = slice(cachedGeometry, 0, end);
      for (const path of [halo, core, flow]) path.setAttribute('d', visible.d);
      tail.setAttribute('d', progress < 1 && !reducedMotion.matches
        ? slice(cachedGeometry, Math.max(0, end - 90), end).d : '');
      const tip = visible.tip || (end === 0 ? points[0] : null);
      position(start, points[0]);
      position(head, tip);
      position(headGlow, progress < 1 && !reducedMotion.matches ? tip : null);
      svg.setAttribute('data-progress', progress.toFixed(3));
    }
    function frame(now) {
      raf = 0;
      if (startedAt === null) startedAt = now;
      progress = reducedMotion.matches ? 1 : Math.min(1, (now - startedAt) / duration);
      render();
      if (progress < 1) raf = window.requestAnimationFrame(frame);
    }
    // ids：正文提及顺序的主链；extra：[父, 子] 支线，让同样亮起却不在正文里的节点也被连到。
    function setRoute(nextIds, extra) {
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0;
      ids = Array.from(new Set(nextIds));
      branches = extra || [];
      cachedPoints = [];
      startedAt = null;
      progress = reducedMotion.matches ? 1 : 0;
      render();
      // 以屏幕弧长计时，短路线舒缓，长路线保留完整观看节奏。
      duration = Math.max(2800, Math.min(14000, cachedGeometry.total / 220 * 1000));
      if ((ids.length > 1 || branches.length) && progress < 1) raf = window.requestAnimationFrame(frame);
    }
    function onMotionChange() {
      if (!reducedMotion.matches) return;
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0; progress = 1; render();
    }
    reducedMotion.addEventListener('change', onMotionChange);
    return { setRoute, render, getState: () => ({ ids: ids.slice(), branches: branches.length, progress, animating: !!raf }),
      destroy: () => {
        if (raf) window.cancelAnimationFrame(raf);
        reducedMotion.removeEventListener('change', onMotionChange);
        svg.replaceChildren();
      },
    };
  }
  window.RNGraphRouteTrace = { create, geometry, slice, branches };
})();
