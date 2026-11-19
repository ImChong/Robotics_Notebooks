(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RNGraphPerspective = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * 知识图谱 CSS 3D 透视视图渲染适配层。
   * 仅作用于图谱渲染包装容器与 SVG 输入坐标逆投影：
   * 不修改图谱数据、力模拟坐标、筛选/搜索逻辑或交互语义。
   */
  var STORAGE_KEY = 'rn-graph-css-perspective';

  var BOUNDS = {
    perspective: { min: 700, max: 2000, step: 20 },
    rotateX: { min: 0, max: 58, step: 1 },
    rotateY: { min: -28, max: 28, step: 1 },
    depth: { min: 0, max: 100, step: 2 },
  };

  var DEFAULTS_BY_TIER = {
    full: { perspective: 1150, rotateX: 34, rotateY: -6, depth: 60 },
    lite: { perspective: 1200, rotateX: 26, rotateY: 0, depth: 40 },
    reduced: { perspective: 1350, rotateX: 16, rotateY: 0, depth: 25 },
    unsupported: { perspective: 1200, rotateX: 0, rotateY: 0, depth: 0 },
  };

  var FLAT_PARAMS = { perspective: 1200, rotateX: 0, rotateY: 0, depth: 0 };

  var TIER_BADGE_TEXT = {
    full: '',
    lite: '轻量模式',
    reduced: '减弱动效',
    unsupported: '降级 2D',
  };

  function getDefaultsForTier(tier) {
    var base = DEFAULTS_BY_TIER[tier] || DEFAULTS_BY_TIER.full;
    return {
      perspective: base.perspective,
      rotateX: base.rotateX,
      rotateY: base.rotateY,
      depth: base.depth,
    };
  }

  function clampNumber(raw, fallback, min, max) {
    var n = (typeof raw === 'number') ? raw : (raw !== null && raw !== '' && raw !== undefined ? Number(raw) : NaN);
    if (!Number.isFinite(n)) return fallback;
    return Math.round(Math.min(max, Math.max(min, n)));
  }

  function clampParams(raw, tier) {
    var defaults = getDefaultsForTier(tier);
    var src = (raw && typeof raw === 'object') ? raw : {};
    var maxRx = tier === 'lite' ? Math.min(50, BOUNDS.rotateX.max) : BOUNDS.rotateX.max;
    return {
      perspective: clampNumber(src.perspective, defaults.perspective, BOUNDS.perspective.min, BOUNDS.perspective.max),
      rotateX: clampNumber(src.rotateX, defaults.rotateX, BOUNDS.rotateX.min, maxRx),
      rotateY: clampNumber(src.rotateY, defaults.rotateY, BOUNDS.rotateY.min, BOUNDS.rotateY.max),
      depth: clampNumber(src.depth, defaults.depth, BOUNDS.depth.min, BOUNDS.depth.max),
    };
  }

  function resolveCapabilityTier(env) {
    var info = env || {};
    if (info.supports3D === false) return 'unsupported';
    if (info.reducedMotion) return 'reduced';
    var cores = Number(info.hardwareConcurrency);
    var mem = Number(info.deviceMemory);
    var dpr = Number(info.dpr) || 1;
    var nodeCount = Number(info.nodeCount) || 0;
    if (
      info.coarsePointer ||
      (Number.isFinite(cores) && cores > 0 && cores <= 4) ||
      (Number.isFinite(mem) && mem > 0 && mem <= 4) ||
      (dpr >= 2.5 && nodeCount > 3500)
    ) {
      return 'lite';
    }
    return 'full';
  }

  function detectEnvironment(nodeCount) {
    var supports3D;
    try {
      supports3D = typeof CSS !== 'undefined' &&
        typeof CSS.supports === 'function' &&
        CSS.supports('perspective', '1000px') &&
        CSS.supports('transform-style', 'preserve-3d');
    } catch (_e) {
      supports3D = false;
    }
    var reducedMotion = false;
    var coarsePointer = false;
    try {
      if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
        reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        coarsePointer = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
      }
    } catch (_e) {
      reducedMotion = false;
      coarsePointer = false;
    }
    var nav = typeof navigator !== 'undefined' ? navigator : {};
    var dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;
    return {
      supports3D: supports3D,
      reducedMotion: reducedMotion,
      coarsePointer: coarsePointer,
      hardwareConcurrency: nav.hardwareConcurrency || 8,
      deviceMemory: nav.deviceMemory || 8,
      dpr: dpr,
      nodeCount: nodeCount || 0,
    };
  }

  function computeSceneMetrics(params, _width, height, tier) {
    var h = Math.max(1, Number(height) || 800);
    var p = clampParams(params, tier);
    var rxRad = (p.rotateX * Math.PI) / 180;
    var ryRad = (p.rotateY * Math.PI) / 180;
    var depthRatio = p.depth / 100;
    var isLite = tier === 'lite';
    var isReduced = tier === 'reduced';
    var floorZ = Math.round(depthRatio * (isLite ? 46 : 76));
    var liftZ = Math.round(depthRatio * (isLite ? 14 : 30));
    var effectivePerspective = Math.max(p.perspective, Math.round(h * 0.92));
    var tiltFrac = Math.min(1, (Math.abs(p.rotateX) / 58) * 0.8 + (Math.abs(p.rotateY) / 28) * 0.2);
    var planeScale = Number((1 - 0.13 * tiltFrac - 0.03 * depthRatio).toFixed(4));
    var offsetY = Math.round(-(Math.abs(p.rotateX) / 58) * Math.min(h * 0.025, 18));
    var shadowOpacity = Number((depthRatio * (isLite ? 0 : isReduced ? 0.25 : 0.48)).toFixed(3));
    var gridOpacity = Number((0.22 + depthRatio * 0.48).toFixed(3));
    var labelUnscaleY = (isLite || isReduced)
      ? 1
      : Number(Math.min(1.38, 1 / Math.max(0.72, Math.cos(rxRad * 0.72))).toFixed(4));
    var labelLiftPx = (isLite || isReduced)
      ? 0
      : -Math.round(depthRatio * 9 * (Math.abs(p.rotateX) > 0 ? Math.min(1.2, Math.abs(p.rotateX) / 34) : 0.4));

    var cosx = Math.cos(rxRad);
    var sinx = Math.sin(rxRad);
    var cosy = Math.cos(ryRad);
    var siny = Math.sin(ryRad);
    // R = Rx(rx) * Ry(ry) basis vectors in CSS screen coordinates (+x right, +y down, +z toward viewer)
    var eu = [cosy, sinx * siny, -cosx * siny];
    var ev = [0, cosx, sinx];
    var n = [siny, -sinx * cosy, cosx * cosy];

    var isTilted =
      Math.abs(p.rotateX) > 0.01 ||
      Math.abs(p.rotateY) > 0.01 ||
      Math.abs(liftZ) > 0.01 ||
      Math.abs(planeScale - 1) > 0.0005 ||
      Math.abs(offsetY) > 0.01;

    return {
      perspective: effectivePerspective,
      rawPerspective: p.perspective,
      rotateX: p.rotateX,
      rotateY: p.rotateY,
      depth: p.depth,
      rxRad: rxRad,
      ryRad: ryRad,
      depthRatio: depthRatio,
      floorZ: floorZ,
      liftZ: liftZ,
      planeScale: planeScale,
      offsetY: offsetY,
      shadowOpacity: shadowOpacity,
      gridOpacity: gridOpacity,
      labelUnscaleY: labelUnscaleY,
      labelLiftPx: labelLiftPx,
      eu: eu,
      ev: ev,
      n: n,
      isTilted: isTilted,
    };
  }

  function projectSvgToClient(u, v, metrics, stageRect) {
    var w = stageRect.width || 1;
    var h = stageRect.height || 1;
    var cx = stageRect.left + w / 2;
    var cy = stageRect.top + h / 2;
    var x0 = (u - w / 2) * metrics.planeScale;
    var y0 = (v - h / 2) * metrics.planeScale;
    var z0 = metrics.liftZ;
    var eu = metrics.eu;
    var ev = metrics.ev;
    var n = metrics.n;
    var px = x0 * eu[0] + y0 * ev[0] + z0 * n[0];
    var py = x0 * eu[1] + y0 * ev[1] + z0 * n[1];
    var pz = x0 * eu[2] + y0 * ev[2] + z0 * n[2];
    var perspW = Math.max(0.05, 1 - pz / metrics.perspective);
    return [cx + px / perspW, cy + (py + metrics.offsetY) / perspW];
  }

  function unprojectClientToSvg(clientX, clientY, metrics, stageRect) {
    var w = stageRect.width || 1;
    var h = stageRect.height || 1;
    var cx = stageRect.left + w / 2;
    var cy = stageRect.top + h / 2;
    if (!metrics.isTilted) {
      return [clientX - stageRect.left, clientY - stageRect.top];
    }
    var X = clientX - cx;
    var Y = clientY - cy;
    var P = metrics.perspective;
    var eu = metrics.eu;
    var ev = metrics.ev;
    var n = metrics.n;
    var denom0 = -P * n[2];
    var rawDenom = X * n[0] + Y * n[1] - P * n[2];
    // Prevent singularity / sign inversion above vanishing horizon line
    var denom = Math.min(rawDenom, denom0 * 0.18);
    var t = (metrics.liftZ + metrics.offsetY * n[1] - P * n[2]) / denom;
    var hx = t * X;
    var hy = t * Y - metrics.offsetY;
    var hz = P * (1 - t);
    var u = w / 2 + (hx * eu[0] + hy * eu[1] + hz * eu[2]) / metrics.planeScale;
    var v = h / 2 + (hx * ev[0] + hy * ev[1] + hz * ev[2]) / metrics.planeScale;
    return [u, v];
  }

  /**
   * 为 SVG 与节点组安装实例级坐标逆映射适配器（不修改原生原型）。
   * 当 CSS 3D 透视激活时，d3.pointer() 通过 createSVGPoint + getScreenCTM().inverse()
   * 拿到的屏幕坐标会按透视平面精确反求回 SVG / 世界坐标，保证缩放中心、平移与节点拖拽跟手。
   */
  function installSvgCoordinateAdapter(opts) {
    var svgEl = opts && opts.svgEl;
    var gNodesEl = opts && opts.gNodesEl;
    if (!svgEl || typeof svgEl.createSVGPoint !== 'function' || typeof svgEl.getScreenCTM !== 'function') {
      return { uninstall: function () {} };
    }
    var isAdapterActive = opts.isAdapterActive || function () { return false; };
    var getZoomTransform = opts.getZoomTransform || function () { return { x: 0, y: 0, k: 1 }; };
    var unprojectClientPoint = opts.unprojectClientPoint || function (cx, cy) { return [cx, cy]; };

    var nativeCreateSVGPoint = svgEl.createSVGPoint;
    var nativeSvgGetScreenCTM = svgEl.getScreenCTM;
    var nativeGNodesGetScreenCTM = gNodesEl && typeof gNodesEl.getScreenCTM === 'function'
      ? gNodesEl.getScreenCTM : null;

    var svgInvMarker = { __rnPerspMode: 'svg' };
    var worldInvMarker = { __rnPerspMode: 'world' };
    var svgCtmShim = { inverse: function () { return svgInvMarker; } };
    var worldCtmShim = { inverse: function () { return worldInvMarker; } };

    svgEl.createSVGPoint = function () {
      if (!isAdapterActive()) {
        return nativeCreateSVGPoint.call(svgEl);
      }
      var pt = {
        x: 0,
        y: 0,
        z: 0,
        w: 1,
        matrixTransform: function (m) {
          if (m && m.__rnPerspMode === 'svg') {
            var uv = unprojectClientPoint(pt.x, pt.y);
            return { x: uv[0], y: uv[1], z: 0, w: 1 };
          }
          if (m && m.__rnPerspMode === 'world') {
            var uv2 = unprojectClientPoint(pt.x, pt.y);
            var zt = getZoomTransform() || { x: 0, y: 0, k: 1 };
            var k = zt.k || 1;
            return { x: (uv2[0] - zt.x) / k, y: (uv2[1] - zt.y) / k, z: 0, w: 1 };
          }
          var nativePt = nativeCreateSVGPoint.call(svgEl);
          nativePt.x = pt.x;
          nativePt.y = pt.y;
          return nativePt.matrixTransform(m);
        },
      };
      return pt;
    };

    svgEl.getScreenCTM = function () {
      if (!isAdapterActive()) {
        return nativeSvgGetScreenCTM.call(svgEl);
      }
      return svgCtmShim;
    };

    if (gNodesEl && nativeGNodesGetScreenCTM) {
      gNodesEl.getScreenCTM = function () {
        if (!isAdapterActive()) {
          return nativeGNodesGetScreenCTM.call(gNodesEl);
        }
        return worldCtmShim;
      };
    }

    return {
      uninstall: function () {
        svgEl.createSVGPoint = nativeCreateSVGPoint;
        svgEl.getScreenCTM = nativeSvgGetScreenCTM;
        if (gNodesEl && nativeGNodesGetScreenCTM) {
          gNodesEl.getScreenCTM = nativeGNodesGetScreenCTM;
        }
      },
    };
  }

  function loadStoredParams(tier) {
    try {
      if (typeof localStorage === 'undefined') return getDefaultsForTier(tier);
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return getDefaultsForTier(tier);
      return clampParams(JSON.parse(raw), tier);
    } catch (_e) {
      return getDefaultsForTier(tier);
    }
  }

  function saveStoredParams(params) {
    try {
      if (typeof localStorage === 'undefined') return;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
    } catch (_e) {
      // ignore storage errors
    }
  }

  function createController(opts) {
    opts = opts || {};
    var wrapEl = opts.wrapEl || null;
    var stageEl = opts.stageEl || null;
    var sceneEl = opts.sceneEl || null;
    var planeEl = opts.planeEl || null;
    var floorEl = opts.floorEl || null;
    var shadowEl = opts.shadowEl || null;
    var svgEl = opts.svgEl || null;
    var gNodesEl = opts.gNodesEl || null;
    var getViewportSize = opts.getViewportSize || function () {
      return {
        width: wrapEl ? wrapEl.clientWidth : 1200,
        height: wrapEl ? wrapEl.clientHeight : 800,
      };
    };
    var getZoomTransform = opts.getZoomTransform || function () { return { x: 0, y: 0, k: 1 }; };
    var onStateChange = opts.onStateChange || function () {};
    var onUnsupported = opts.onUnsupported || function () {};

    var env = detectEnvironment(opts.nodeCount || 0);
    var tier = resolveCapabilityTier(env);
    var enabled = false;
    var dragging = false;
    var params = loadStoredParams(tier);
    var metrics = null;
    var cachedStageRect = { left: 0, top: 0, width: 1200, height: 800 };
    var applyRaf = 0;
    var animIdleTimer = 0;

    function refreshStageRect() {
      var target = stageEl || wrapEl;
      if (target && typeof target.getBoundingClientRect === 'function') {
        var r = target.getBoundingClientRect();
        if (r && r.width > 0 && r.height > 0) {
          cachedStageRect = { left: r.left, top: r.top, width: r.width, height: r.height };
          return cachedStageRect;
        }
      }
      var vp = getViewportSize();
      cachedStageRect = { left: 0, top: 0, width: vp.width || 1200, height: vp.height || 800 };
      return cachedStageRect;
    }

    function clearInlineSceneStyles() {
      if (stageEl) {
        stageEl.classList.remove('is-perspective-animating');
        stageEl.style.removeProperty('--persp-distance');
      }
      if (wrapEl) {
        wrapEl.classList.remove(
          'is-perspective-mode',
          'is-perspective-lite',
          'is-perspective-reduced',
          'is-perspective-flat',
          'is-perspective-dragging'
        );
        wrapEl.style.removeProperty('--persp-distance');
        wrapEl.style.removeProperty('--persp-rx');
        wrapEl.style.removeProperty('--persp-ry');
        wrapEl.style.removeProperty('--persp-floor-z');
        wrapEl.style.removeProperty('--persp-lift-z');
        wrapEl.style.removeProperty('--persp-scale');
        wrapEl.style.removeProperty('--persp-offset-y');
        wrapEl.style.removeProperty('--persp-shadow-opacity');
        wrapEl.style.removeProperty('--persp-grid-opacity');
        wrapEl.style.removeProperty('--persp-label-unscale-y');
        wrapEl.style.removeProperty('--persp-label-lift');
      }
      if (sceneEl) sceneEl.style.transform = '';
      if (planeEl) planeEl.style.transform = '';
      if (floorEl) floorEl.style.transform = '';
      if (shadowEl) shadowEl.style.transform = '';
    }

    function bumpAnimatingWindow() {
      if (!stageEl) return;
      stageEl.classList.add('is-perspective-animating');
      clearTimeout(animIdleTimer);
      if (dragging) return;
      animIdleTimer = setTimeout(function () {
        if (!dragging && stageEl) stageEl.classList.remove('is-perspective-animating');
      }, 300);
    }

    function applyNow() {
      applyRaf = 0;
      if (!enabled || tier === 'unsupported') {
        metrics = null;
        clearTimeout(animIdleTimer);
        clearInlineSceneStyles();
        onStateChange(getPublicState());
        return;
      }
      var vp = getViewportSize();
      var rect = refreshStageRect();
      var w = vp.width || rect.width || 1200;
      var h = vp.height || rect.height || 800;
      metrics = computeSceneMetrics(params, w, h, tier);

      if (wrapEl) {
        wrapEl.classList.add('is-perspective-mode');
        wrapEl.classList.toggle('is-perspective-lite', tier === 'lite');
        wrapEl.classList.toggle('is-perspective-reduced', tier === 'reduced');
        wrapEl.classList.toggle('is-perspective-flat', !metrics.isTilted);
        wrapEl.classList.toggle('is-perspective-dragging', dragging);
        wrapEl.style.setProperty('--persp-distance', metrics.perspective + 'px');
        wrapEl.style.setProperty('--persp-rx', metrics.rotateX + 'deg');
        wrapEl.style.setProperty('--persp-ry', metrics.rotateY + 'deg');
        wrapEl.style.setProperty('--persp-floor-z', metrics.floorZ + 'px');
        wrapEl.style.setProperty('--persp-lift-z', metrics.liftZ + 'px');
        wrapEl.style.setProperty('--persp-scale', String(metrics.planeScale));
        wrapEl.style.setProperty('--persp-offset-y', metrics.offsetY + 'px');
        wrapEl.style.setProperty('--persp-shadow-opacity', String(metrics.shadowOpacity));
        wrapEl.style.setProperty('--persp-grid-opacity', String(metrics.gridOpacity));
        wrapEl.style.setProperty('--persp-label-unscale-y', String(metrics.labelUnscaleY));
        wrapEl.style.setProperty('--persp-label-lift', metrics.labelLiftPx + 'px');
      }
      if (stageEl) {
        stageEl.style.setProperty('--persp-distance', metrics.perspective + 'px');
      }
      onStateChange(getPublicState());
    }

    function scheduleApply() {
      if (applyRaf) return;
      if (typeof requestAnimationFrame === 'function') {
        applyRaf = requestAnimationFrame(applyNow);
      } else {
        applyNow();
      }
    }

    var adapter = installSvgCoordinateAdapter({
      svgEl: svgEl,
      gNodesEl: gNodesEl,
      isAdapterActive: function () {
        return enabled && !!metrics && metrics.isTilted;
      },
      getZoomTransform: getZoomTransform,
      unprojectClientPoint: function (clientX, clientY) {
        if (!metrics) return [clientX - cachedStageRect.left, clientY - cachedStageRect.top];
        return unprojectClientToSvg(clientX, clientY, metrics, cachedStageRect);
      },
    });

    function onStagePointerDown() {
      if (enabled) refreshStageRect();
    }
    if (stageEl && typeof stageEl.addEventListener === 'function') {
      stageEl.addEventListener('pointerdown', onStagePointerDown, { capture: true, passive: true });
      stageEl.addEventListener('wheel', onStagePointerDown, { capture: true, passive: true });
    }

    var reducedMq = null;
    function handleReducedMotionChange(ev) {
      env.reducedMotion = !!(ev && ev.matches);
      var nextTier = resolveCapabilityTier(env);
      if (nextTier !== tier) {
        tier = nextTier;
        params = clampParams(params, tier);
        if (enabled) scheduleApply();
        else onStateChange(getPublicState());
      }
    }
    try {
      if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
        reducedMq = window.matchMedia('(prefers-reduced-motion: reduce)');
        if (reducedMq.addEventListener) reducedMq.addEventListener('change', handleReducedMotionChange);
        else if (reducedMq.addListener) reducedMq.addListener(handleReducedMotionChange);
      }
    } catch (_e) {
      reducedMq = null;
    }

    function getPublicState() {
      return {
        supported: tier !== 'unsupported',
        tier: tier,
        tierBadge: TIER_BADGE_TEXT[tier] || '',
        enabled: enabled,
        params: {
          perspective: params.perspective,
          rotateX: params.rotateX,
          rotateY: params.rotateY,
          depth: params.depth,
        },
        metrics: metrics,
      };
    }

    function setEnabled(on, setOpts) {
      var next = !!on;
      if (next && tier === 'unsupported') {
        enabled = false;
        applyNow();
        if (!setOpts || !setOpts.silent) onUnsupported();
        return false;
      }
      if (next === enabled) {
        if (next) applyNow();
        return true;
      }
      enabled = next;
      if (enabled) {
        params = clampParams(params, tier);
        bumpAnimatingWindow();
        applyNow();
      } else {
        dragging = false;
        applyNow();
      }
      return true;
    }

    function setParam(key, value) {
      if (!Object.prototype.hasOwnProperty.call(BOUNDS, key)) return;
      var next = Object.assign({}, params);
      next[key] = value;
      params = clampParams(next, tier);
      saveStoredParams(params);
      if (enabled) {
        bumpAnimatingWindow();
        scheduleApply();
      } else {
        onStateChange(getPublicState());
      }
    }

    function setDragging(active) {
      dragging = !!active;
      if (wrapEl) wrapEl.classList.toggle('is-perspective-dragging', enabled && dragging);
      if (dragging && enabled) {
        bumpAnimatingWindow();
      } else if (!dragging && stageEl) {
        clearTimeout(animIdleTimer);
        animIdleTimer = setTimeout(function () {
          if (!dragging && stageEl) stageEl.classList.remove('is-perspective-animating');
        }, 280);
      }
    }

    function resetDefaults() {
      params = getDefaultsForTier(tier);
      saveStoredParams(params);
      if (enabled) {
        bumpAnimatingWindow();
        applyNow();
      } else {
        onStateChange(getPublicState());
      }
    }

    function setFlatView() {
      params = clampParams({
        perspective: params.perspective || FLAT_PARAMS.perspective,
        rotateX: FLAT_PARAMS.rotateX,
        rotateY: FLAT_PARAMS.rotateY,
        depth: FLAT_PARAMS.depth,
      }, tier);
      saveStoredParams(params);
      if (enabled) {
        bumpAnimatingWindow();
        applyNow();
      } else {
        onStateChange(getPublicState());
      }
    }

    function syncViewport() {
      if (!enabled) return;
      refreshStageRect();
      applyNow();
    }

    function destroy() {
      if (applyRaf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(applyRaf);
      clearTimeout(animIdleTimer);
      if (stageEl && typeof stageEl.removeEventListener === 'function') {
        stageEl.removeEventListener('pointerdown', onStagePointerDown, { capture: true });
        stageEl.removeEventListener('wheel', onStagePointerDown, { capture: true });
      }
      if (reducedMq) {
        if (reducedMq.removeEventListener) reducedMq.removeEventListener('change', handleReducedMotionChange);
        else if (reducedMq.removeListener) reducedMq.removeListener(handleReducedMotionChange);
      }
      adapter.uninstall();
      clearInlineSceneStyles();
    }

    onStateChange(getPublicState());

    return {
      isSupported: function () { return tier !== 'unsupported'; },
      getTier: function () { return tier; },
      isEnabled: function () { return enabled; },
      getParams: function () { return Object.assign({}, params); },
      getState: getPublicState,
      setEnabled: setEnabled,
      setParam: setParam,
      setDragging: setDragging,
      resetDefaults: resetDefaults,
      setFlatView: setFlatView,
      syncViewport: syncViewport,
      refreshStageRect: refreshStageRect,
      getDebugState: function () {
        return {
          supported: tier !== 'unsupported',
          tier: tier,
          enabled: enabled,
          params: Object.assign({}, params),
          metrics: metrics ? {
            perspective: metrics.perspective,
            rotateX: metrics.rotateX,
            rotateY: metrics.rotateY,
            depth: metrics.depth,
            floorZ: metrics.floorZ,
            liftZ: metrics.liftZ,
            planeScale: metrics.planeScale,
            offsetY: metrics.offsetY,
            isTilted: metrics.isTilted,
          } : null,
          stageRect: cachedStageRect,
          animating: !!(stageEl && stageEl.classList.contains('is-perspective-animating')),
        };
      },
      destroy: destroy,
    };
  }

  return {
    STORAGE_KEY: STORAGE_KEY,
    BOUNDS: BOUNDS,
    DEFAULTS_BY_TIER: DEFAULTS_BY_TIER,
    FLAT_PARAMS: FLAT_PARAMS,
    TIER_BADGE_TEXT: TIER_BADGE_TEXT,
    getDefaultsForTier: getDefaultsForTier,
    clampParams: clampParams,
    resolveCapabilityTier: resolveCapabilityTier,
    detectEnvironment: detectEnvironment,
    computeSceneMetrics: computeSceneMetrics,
    projectSvgToClient: projectSvgToClient,
    unprojectClientToSvg: unprojectClientToSvg,
    installSvgCoordinateAdapter: installSvgCoordinateAdapter,
    createController: createController,
  };
});
