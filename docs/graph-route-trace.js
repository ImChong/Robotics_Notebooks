(function () {
  'use strict';

  const mix = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  const distance = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
  function split(c, t) {
    const a = mix(c[0], c[1], t), b = mix(c[1], c[2], t), d = mix(c[2], c[3], t);
    const e = mix(a, b, t), f = mix(b, d, t), p = mix(e, f, t);
    return [[c[0], a, e, p], [p, f, d, c[3]]];
  }
  function portion(c, from, to) {
    const left = to < 1 ? split(c, to)[0] : c;
    return from > 0 && to > 0 ? split(left, from / to)[1] : left;
  }
  function tangent(prev, p, next) {
    const v = { x: next.x - prev.x, y: next.y - prev.y };
    const len = Math.hypot(v.x, v.y);
    const cap = Math.min(distance(prev, p) || Infinity, distance(p, next) || Infinity) * 0.3;
    const scale = len ? Math.min(0.16, cap / len) : 0;
    return { x: v.x * scale, y: v.y * scale };
  }

  // 平滑切线穿过节点；弧长采样让长短边之间保持连续速度，不按每条边跳步。
  function geometry(points) {
    const segments = [];
    let total = 0;
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i], b = points[i + 1];
      if (!a || !b || distance(a, b) < 0.01) continue;
      const va = tangent(points[i - 1] || a, a, b);
      const vb = tangent(a, b, points[i + 2] || b);
      const c = [a, { x: a.x + va.x, y: a.y + va.y },
        { x: b.x - vb.x, y: b.y - vb.y }, b];
      const lengths = [0];
      let last = a;
      for (let j = 1; j <= 24; j++) {
        const p = split(c, j / 24)[0][3];
        lengths.push(lengths[j - 1] + distance(last, p));
        last = p;
      }
      const length = lengths[24];
      segments.push({ c, lengths, length, start: total, run: i });
      total += length;
    }
    return { segments, total };
  }
  function parameter(segment, length) {
    if (length <= 0) return 0;
    if (length >= segment.length) return 1;
    const samples = segment.lengths;
    for (let i = 1; i < samples.length; i++) {
      if (samples[i] < length) continue;
      return (i - 1 + (length - samples[i - 1]) / (samples[i] - samples[i - 1] || 1)) / 24;
    }
    return 1;
  }
  const xy = p => p.x.toFixed(2) + ',' + p.y.toFixed(2);
  function slice(g, from, to) {
    let d = '', tip = null, lastRun = -2;
    for (const s of g.segments) {
      if (!s.length || s.start >= to || s.start + s.length <= from) continue;
      const a = parameter(s, Math.max(0, from - s.start));
      const b = parameter(s, Math.min(s.length, to - s.start));
      const c = portion(s.c, a, b);
      if (s.run !== lastRun + 1 || a > 0) d += 'M' + xy(c[0]);
      d += 'C' + xy(c[1]) + ' ' + xy(c[2]) + ' ' + xy(c[3]);
      tip = c[3];
      lastRun = s.run;
    }
    return { d, tip };
  }

  // 独立叙事层：主题蓝细线、柔光尾迹、流动微光，不向知识图或力模拟加边。
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
    let ids = [], progress = 0, startedAt = null, raf = 0, duration = 0;
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
      const on = ids.length > 1 && options.isActive();
      svg.style.display = on ? '' : 'none';
      svg.classList.toggle('route-trace-reduced', reducedMotion.matches);
      if (!on) return;
      const points = ids.map(point);
      if (points.length !== cachedPoints.length || points.some((p, i) =>
        !!p !== !!cachedPoints[i] || p && (p.x !== cachedPoints[i].x || p.y !== cachedPoints[i].y))) {
        cachedPoints = points;
        cachedGeometry = geometry(points);
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
    function setRoute(nextIds) {
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0;
      ids = Array.from(new Set(nextIds));
      startedAt = null;
      progress = reducedMotion.matches ? 1 : 0;
      render();
      // 以屏幕弧长计时，短路线舒缓，长路线保留完整观看节奏。
      duration = Math.max(2800, Math.min(14000, cachedGeometry.total / 220 * 1000));
      if (ids.length > 1 && progress < 1) raf = window.requestAnimationFrame(frame);
    }
    function onMotionChange() {
      if (!reducedMotion.matches) return;
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0; progress = 1; render();
    }
    reducedMotion.addEventListener('change', onMotionChange);
    return { setRoute, render, getState: () => ({ ids: ids.slice(), progress, animating: !!raf }),
      destroy: () => {
        if (raf) window.cancelAnimationFrame(raf);
        reducedMotion.removeEventListener('change', onMotionChange);
        svg.replaceChildren();
      },
    };
  }
  window.RNGraphRouteTrace = { create, geometry, slice };
})();
