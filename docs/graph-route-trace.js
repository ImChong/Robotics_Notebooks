(function () {
  'use strict';

  // 仅为路线加一层叙事连线，不写入知识图的边与力模拟。
  function create(options) {
    const svg = options.svg;
    const ns = 'http://www.w3.org/2000/svg';
    const make = (tag, attrs) => {
      const el = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      svg.appendChild(el);
      return el;
    };
    const path = make('path', {
      fill: 'none', stroke: '#f59e0b', 'stroke-width': 2.6,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
    });
    const start = make('circle', { r: 7, fill: 'none', stroke: '#f59e0b', 'stroke-width': 2.6 });
    const head = make('circle', { r: 4.5, fill: '#f59e0b', stroke: '#fff', 'stroke-width': 1.5 });
    let ids = [];
    let progress = 0;
    let startedAt = null;
    let raf = 0;
    let duration = 0;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    function position(el, point) {
      el.style.display = point ? '' : 'none';
      if (point) { el.setAttribute('cx', point.x); el.setAttribute('cy', point.y); }
    }
    function point(id) {
      if (!options.isVisible(id)) return null;
      const p = options.project(id);
      return p && Number.isFinite(p.x) && Number.isFinite(p.y) ? p : null;
    }
    function render() {
      const on = ids.length > 1 && options.isActive();
      svg.style.display = on ? '' : 'none';
      if (!on) return;
      const limit = Math.min(ids.length - 1, progress * (ids.length - 1));
      let d = '';
      let tip = null;
      for (let i = 0; i < limit; i++) {
        const a = point(ids[i]);
        const b = point(ids[i + 1]);
        // 筛选隐藏中间节点时留缺口，不把它的前后节点误接起来。
        if (!a || !b) { tip = null; continue; }
        const t = Math.min(1, limit - i);
        tip = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
        d += `M${a.x},${a.y}L${tip.x},${tip.y}`;
      }
      path.setAttribute('d', d);
      position(start, point(ids[0]));
      position(head, tip || (limit === 0 ? point(ids[0]) : null));
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
      duration = Math.max(2400, Math.min(12000, (ids.length - 1) * 240));
      render();
      if (ids.length > 1 && progress < 1) raf = window.requestAnimationFrame(frame);
    }
    function onMotionChange() {
      if (!reducedMotion.matches) return;
      if (raf) window.cancelAnimationFrame(raf);
      raf = 0;
      progress = 1;
      render();
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
  window.RNGraphRouteTrace = { create };
})();
