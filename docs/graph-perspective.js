(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RNGraphPerspective = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * 知识图谱 3D 连接度地形渲染层（WebGL2 / WebGL1 + 屏幕空间标签叠加）。
   * - 地形高度由活跃节点的连接数（度）与局部聚类密度生成；
   * - 等高线（地平线层）与峰值辉光对应连接数层级；
   * - 边界通过径向雾化与 alpha 衰减无缝融于背景（无矩形边框）；
   * - 不修改图谱数据 schema、力模拟坐标、筛选/搜索逻辑。
   */
  var STORAGE_KEY = 'rn-graph-terrain-3d';

  var BOUNDS = {
    height: { min: 20, max: 160, step: 2 },
    contours: { min: 6, max: 28, step: 1 },
    pitch: { min: 14, max: 78, step: 1 },
    yaw: { min: -180, max: 180, step: 1 },
  };

  var DEFAULTS_BY_TIER = {
    full: { height: 100, contours: 16, pitch: 36, yaw: -18 },
    lite: { height: 88, contours: 12, pitch: 32, yaw: -12 },
    reduced: { height: 86, contours: 14, pitch: 38, yaw: -14 },
    unsupported: { height: 100, contours: 12, pitch: 36, yaw: 0 },
  };
  var PREVIEW_HEIGHT_BY_TIER = { full: 32, lite: 29, reduced: 28 };

  var TOP_VIEW_PARAMS = { pitch: 76, yaw: 0 };

  var TIER_BADGE_TEXT = {
    full: '',
    lite: '轻量模式',
    reduced: '减弱动效',
    unsupported: '降级 2D',
  };

  /**
   * 3D 地形节点拖拽松手后的回弹物理参数：
   * 对齐 2D（VELOCITY_DECAY=0.54）与 3D 立体（d3VelocityDecay=0.40）力模拟的粘滞过阻尼松弛，
   * 阻尼比 ζ > 1（主节点 ζ ≈ 1.26、邻居 ζ ≈ 1.43），平滑回拉至平衡点且零过冲弹跳。
   */
  var REBOUND_PARAMS = {
    kSpring: 0.036,
    velocityDecay: 0.38,   // 每帧保留 62% 速度（过阻尼 ζ ≈ 1.26，无过冲往复），对齐 2D/3D 力模拟粘滞收敛
    nbSpring: 0.032,
    nbVelocityDecay: 0.40, // 邻居每帧保留 60% 速度（过阻尼 ζ ≈ 1.43）
    nbResidual: 0.18,
    maxDuration: 2000,
  };

  /** 根据直连邻居数量计算回拉比例（孤立节点仅微调，高连接枢纽温和回拉而不猛弹） */
  function computeReboundPullBack(nbCount) {
    var count = nbCount > 0 ? nbCount : 0;
    if (count <= 0) return 0.10;
    return 0.30 + 0.26 * Math.min(1, Math.log(1 + count) / Math.log(30));
  }

  function getDefaultsForTier(tier) {
    var base = DEFAULTS_BY_TIER[tier] || DEFAULTS_BY_TIER.full;
    return {
      height: base.height,
      contours: base.contours,
      pitch: base.pitch,
      yaw: base.yaw,
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
    var maxHeight = tier === 'lite' ? 128 : BOUNDS.height.max;
    return {
      height: clampNumber(src.height, defaults.height, BOUNDS.height.min, maxHeight),
      contours: clampNumber(src.contours, defaults.contours, BOUNDS.contours.min, BOUNDS.contours.max),
      pitch: clampNumber(src.pitch, defaults.pitch, BOUNDS.pitch.min, BOUNDS.pitch.max),
      yaw: clampNumber(src.yaw, defaults.yaw, BOUNDS.yaw.min, BOUNDS.yaw.max),
    };
  }

  function probeWebGLSupport() {
    try {
      if (typeof document === 'undefined' || typeof document.createElement !== 'function') return false;
      var probe = document.createElement('canvas');
      if (!probe || typeof probe.getContext !== 'function') return false;
      var ctx = probe.getContext('webgl2') || probe.getContext('webgl') || probe.getContext('experimental-webgl');
      return !!ctx;
    } catch (_e) {
      return false;
    }
  }

  function resolveCapabilityTier(env) {
    var info = env || {};
    if (info.supportsWebGL === false || info.supports3D === false) return 'unsupported';
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
    var supportsWebGL = probeWebGLSupport();
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
      supportsWebGL: supportsWebGL,
      supports3D: supportsWebGL,
      reducedMotion: reducedMotion,
      coarsePointer: coarsePointer,
      hardwareConcurrency: nav.hardwareConcurrency || 8,
      deviceMemory: nav.deviceMemory || 8,
      dpr: dpr,
      nodeCount: nodeCount || 0,
    };
  }

  /* ── 颜色解析（记忆化，避免构建 8k+ 节点批次时重复正则解析）── */

  var CSS_COLOR_CACHE = Object.create(null);

  function parseCssColorFast(str, fallback) {
    var fb = fallback || [0.36, 0.61, 0.96, 1];
    if (!str || typeof str !== 'string') return fb;
    var cached = CSS_COLOR_CACHE[str];
    if (cached) return cached;
    var s = str.trim();
    var parsed = null;
    if (s.charAt(0) === '#') {
      var hex = s.slice(1);
      if (hex.length === 3) {
        hex = hex.charAt(0) + hex.charAt(0) + hex.charAt(1) + hex.charAt(1) + hex.charAt(2) + hex.charAt(2);
      }
      if (hex.length >= 6) {
        var n = parseInt(hex.slice(0, 6), 16);
        if (Number.isFinite(n)) {
          parsed = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
        }
      }
    } else {
      var m = s.match(/^rgba?\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*[,\s]\s*([\d.]+)(?:\s*[,/]\s*([\d.]+))?\s*\)/i);
      if (m) {
        parsed = [
          Math.min(1, Math.max(0, Number(m[1]) / 255)),
          Math.min(1, Math.max(0, Number(m[2]) / 255)),
          Math.min(1, Math.max(0, Number(m[3]) / 255)),
          m[4] !== undefined ? Math.min(1, Math.max(0, Number(m[4]))) : 1,
        ];
      }
    }
    if (parsed) {
      CSS_COLOR_CACHE[str] = parsed;
      return parsed;
    }
    return fb;
  }

  function parseCssColor(str, fallback) {
    var fb = fallback || [0.36, 0.61, 0.96, 1];
    return parseCssColorFast(str, fb).slice();
  }

  /* ── 连接度统计与 2D -> 3D 地形高度场（O(G^2 + N)）── */

  function edgeEndId(ep) {
    return (ep && typeof ep === 'object') ? ep.id : ep;
  }

  function computeNodeDegrees(nodes, edges) {
    var counts = Object.create(null);
    var i;
    var nList = nodes || [];
    var eList = edges || [];
    for (i = 0; i < nList.length; i++) {
      var n = nList[i];
      if (!n || !n.id) continue;
      counts[n.id] = (typeof n._degree === 'number' && n._degree > 0) ? n._degree : 0;
    }
    if (!nList.length || (nList[0] && typeof nList[0]._degree !== 'number')) {
      for (i = 0; i < eList.length; i++) {
        var e = eList[i];
        if (!e) continue;
        var s = edgeEndId(e.source);
        var t = edgeEndId(e.target);
        if (!s || !t || s === t) continue;
        if (counts[s] !== undefined) counts[s] += 1;
        if (counts[t] !== undefined) counts[t] += 1;
      }
    }
    var maxDegree = 1;
    for (i = 0; i < nList.length; i++) {
      var id = nList[i] && nList[i].id;
      if (id && counts[id] > maxDegree) maxDegree = counts[id];
    }
    return { counts: counts, maxDegree: maxDegree };
  }

  /**
   * 3D 地形球体世界半径：对齐 3D 立体视图（graph-3d.js）与全站节点标尺（graph-node-size.js），
   * 按 sqrt(degree) 映射到统一世界单位，使大小随连接数自然拉开层次，且点精灵与球体共用同一尺寸标尺。
   */
  function computeSphereRadius3D(degree, maxDegree, isPreview) {
    var r2d;
    if (typeof window !== 'undefined' && window.RNGraphNodeSize &&
        typeof window.RNGraphNodeSize.radiusForDegree === 'function') {
      r2d = window.RNGraphNodeSize.radiusForDegree(degree, maxDegree);
    } else {
      var span = Math.sqrt(Math.max(maxDegree || 1, 2)) - 1;
      var t = (Math.sqrt(Math.max(degree || 0, 1)) - 1) / span;
      if (t < 0) t = 0;
      if (t > 1) t = 1;
      r2d = 4 + 26 * t;
    }
    if (isPreview) {
      // 对齐 graph-3d.js sphereRadiusFor(d) = Math.max(11, r * 1.65)（4.5x 动态范围）
      var r3d = Math.max(11, r2d * 1.65);
      return 0.056 * (r3d / 49.5);
    }
    // 完整图谱（8k+ 节点）：大枢纽保持 0.028 醒目尺度，低度小珠收至 0.0020 对齐 2D 节点间距密度，
    // 避免拥挤盆地过度拥挤触发防重叠反复推挤震荡
    var tFull = Math.max(0, Math.min(1, (r2d - 4) / 26));
    return 0.0020 + (0.028 - 0.0020) * tFull;
  }

  function quantileBounds(values, loQ, hiQ) {
    if (!values.length) return [-1, 1];
    if (values.length <= 16) {
      var mn = Infinity;
      var mx = -Infinity;
      for (var i = 0; i < values.length; i++) {
        if (values[i] < mn) mn = values[i];
        if (values[i] > mx) mx = values[i];
      }
      if (mn === mx) return [mn - 1, mx + 1];
      return [mn, mx];
    }
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var loIdx = Math.max(0, Math.floor((sorted.length - 1) * loQ));
    var hiIdx = Math.min(sorted.length - 1, Math.ceil((sorted.length - 1) * hiQ));
    var lo = sorted[loIdx];
    var hi = sorted[hiIdx];
    if (!(hi > lo)) return [sorted[0] - 1, sorted[sorted.length - 1] + 1];
    return [lo, hi];
  }

  function separableBoxBlur(src, tmp, size, radius) {
    var r = Math.max(1, Math.min(12, radius | 0));
    var inv = 1 / (2 * r + 1);
    var x;
    var z;
    var k;
    for (z = 0; z < size; z++) {
      var rowOff = z * size;
      var acc = 0;
      for (k = -r; k <= r; k++) {
        acc += src[rowOff + Math.max(0, Math.min(size - 1, k))];
      }
      for (x = 0; x < size; x++) {
        tmp[rowOff + x] = acc * inv;
        var outX = Math.max(0, x - r);
        var inX = Math.min(size - 1, x + r + 1);
        acc += src[rowOff + inX] - src[rowOff + outX];
      }
    }
    for (x = 0; x < size; x++) {
      var accV = 0;
      for (k = -r; k <= r; k++) {
        accV += tmp[Math.max(0, Math.min(size - 1, k)) * size + x];
      }
      for (z = 0; z < size; z++) {
        src[z * size + x] = accV * inv;
        var outZ = Math.max(0, z - r);
        var inZ = Math.min(size - 1, z + r + 1);
        accV += tmp[inZ * size + x] - tmp[outZ * size + x];
      }
    }
  }

  function smoothstep(edge0, edge1, x) {
    var t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0 || 1)));
    return t * t * (3 - 2 * t);
  }

  function computeTerrainField(nodes, degreeInfo, opts) {
    opts = opts || {};
    var G = Math.max(32, Math.min(168, opts.gridSize || 144));
    var extent = opts.extent || 1.35;
    var activeSet = opts.activeIds || null;
    var list = nodes || [];
    var degCounts = (degreeInfo && degreeInfo.counts) || {};
    var maxDeg = Math.max(2, (degreeInfo && degreeInfo.maxDegree) || 2);
    var logDen = Math.log(1 + maxDeg);

    var xs = [];
    var ys = [];
    var i;
    var activeList = [];
    for (i = 0; i < list.length; i++) {
      var nd = list[i];
      if (!nd || nd.x == null || nd.y == null) continue;
      if (activeSet && !activeSet.has(nd.id)) continue;
      activeList.push(nd);
      xs.push(nd.x);
      ys.push(nd.y);
    }
    if (!activeList.length) {
      if (activeSet && activeSet.size === 0) {
        var emptyB = (opts.fixedBounds && isFinite(opts.fixedBounds.x) && isFinite(opts.fixedBounds.y) && opts.fixedBounds.halfSpan > 0)
          ? { x: opts.fixedBounds.x, y: opts.fixedBounds.y, halfSpan: opts.fixedBounds.halfSpan, coreSpan: opts.fixedBounds.coreSpan || 0.78 }
          : { x: 0, y: 0, halfSpan: 100, coreSpan: 0.78 };
        return {
          gridSize: G,
          extent: extent,
          heights: new Float32Array(G * G),
          grads: new Float32Array(G * G * 2),
          maxDegree: maxDeg,
          nodeMeta: Object.create(null),
          center: emptyB,
        };
      }
      for (i = 0; i < list.length; i++) {
        var n2 = list[i];
        if (!n2 || n2.x == null || n2.y == null) continue;
        activeList.push(n2);
        xs.push(n2.x);
        ys.push(n2.y);
      }
    }

    var cx;
    var cy;
    var halfSpan;
    var coreSpan = 0.78;
    if (opts.fixedBounds && isFinite(opts.fixedBounds.x) && isFinite(opts.fixedBounds.y) && opts.fixedBounds.halfSpan > 0) {
      cx = opts.fixedBounds.x;
      cy = opts.fixedBounds.y;
      halfSpan = opts.fixedBounds.halfSpan;
      if (opts.fixedBounds.coreSpan > 0) coreSpan = opts.fixedBounds.coreSpan;
    } else {
      var xB = quantileBounds(xs, 0.02, 0.98);
      var yB = quantileBounds(ys, 0.02, 0.98);
      cx = (xB[0] + xB[1]) / 2;
      cy = (yB[0] + yB[1]) / 2;
      halfSpan = Math.max(40, Math.max(xB[1] - xB[0], yB[1] - yB[0]) * 0.55);
    }

    // 纯线性各向同性映射（严格保形，与 2D 平面完全一致）：外围节点自然向外舒展，不再被正交方框挤压
    var nodeMeta = Object.create(null);
    var activeRadials = [];
    for (i = 0; i < list.length; i++) {
      var ndMap = list[i];
      if (!ndMap || !ndMap.id || ndMap.x == null || ndMap.y == null) continue;
      var wx0 = ((ndMap.x - cx) / halfSpan) * coreSpan;
      var wz0 = ((ndMap.y - cy) / halfSpan) * coreSpan;
      var deg0 = degCounts[ndMap.id] || 0;
      var degNorm0 = Math.pow(Math.log(1 + deg0) / logDen, 1.08);
      var isIncluded0 = !activeSet || activeSet.has(ndMap.id);
      nodeMeta[ndMap.id] = {
        wx: wx0,
        wz: wz0,
        deg: deg0,
        degNorm: degNorm0,
        active: isIncluded0,
      };
      if (isIncluded0) activeRadials.push(Math.hypot(wx0, wz0));
    }

    if (!opts.extent && activeRadials.length) {
      activeRadials.sort(function (a, b) { return a - b; });
      var p99R = activeRadials[Math.min(activeRadials.length - 1, Math.floor((activeRadials.length - 1) * 0.99))];
      extent = Math.max(1.45, Math.min(2.6, p99R * 1.24));
    }

    var broadBuf = new Float32Array(G * G);
    var peakBuf = new Float32Array(G * G);
    var tmpBuf = new Float32Array(G * G);
    var cellScale = (G - 1) / (2 * extent);

    var peakSigma = Math.max(2.0, G * 0.025);
    var peakR = Math.min(9, Math.ceil(peakSigma * 2.5));
    var peakSigma2 = 2 * peakSigma * peakSigma;
    var peakWeights = [];
    var dx;
    var dz;
    for (dz = -peakR; dz <= peakR; dz++) {
      for (dx = -peakR; dx <= peakR; dx++) {
        var wK = Math.exp(-(dx * dx + dz * dz) / peakSigma2);
        if (wK > 0.02) peakWeights.push([dx, dz, wK]);
      }
    }

    for (i = 0; i < list.length; i++) {
      var node = list[i];
      if (!node || !node.id || !nodeMeta[node.id]) continue;
      var metaN = nodeMeta[node.id];
      var wx = metaN.wx;
      var wz = metaN.wz;
      var degNorm = metaN.degNorm;
      var contrib = metaN.active ? degNorm : degNorm * 0.06;

      if (!(contrib > 0.001)) continue;
      var gxF = (wx + extent) * cellScale;
      var gzF = (wz + extent) * cellScale;
      // 越出地形网格范围的远端孤点不挤压回网格边线，避免在边界堆起隐形方框
      if (gxF < 0 || gxF >= G - 1 || gzF < 0 || gzF >= G - 1) continue;
      var gx0 = Math.floor(gxF);
      var gz0 = Math.floor(gzF);
      var tx = gxF - gx0;
      var tz = gzF - gz0;

      broadBuf[gz0 * G + gx0] += contrib * (1 - tx) * (1 - tz);
      broadBuf[gz0 * G + gx0 + 1] += contrib * tx * (1 - tz);
      broadBuf[(gz0 + 1) * G + gx0] += contrib * (1 - tx) * tz;
      broadBuf[(gz0 + 1) * G + gx0 + 1] += contrib * tx * tz;

      var peakAmp = Math.pow(contrib, 1.65);
      var gxC = Math.round(gxF);
      var gzC = Math.round(gzF);
      for (var p = 0; p < peakWeights.length; p++) {
        var kdx = gxC + peakWeights[p][0];
        var kdz = gzC + peakWeights[p][1];
        if (kdx < 0 || kdx >= G || kdz < 0 || kdz >= G) continue;
        var idxP = kdz * G + kdx;
        var val = peakAmp * peakWeights[p][2];
        var prev = peakBuf[idxP];
        // Quadratic smooth-max forms rounded U-shaped saddles between hills without sharp V-creases
        var hDiff = Math.max(0.22 - Math.abs(prev - val), 0) / 0.22;
        peakBuf[idxP] = Math.max(prev, val) + hDiff * hDiff * 0.055 + 0.06 * Math.min(prev, val);
      }
    }

    var blurR = Math.max(3, Math.round(G * 0.044));
    separableBoxBlur(broadBuf, tmpBuf, G, blurR);
    separableBoxBlur(broadBuf, tmpBuf, G, Math.max(2, Math.round(blurR * 0.72)));
    separableBoxBlur(peakBuf, tmpBuf, G, 1);

    var maxBroad = 1e-6;
    var maxPeak = 1e-6;
    for (i = 0; i < G * G; i++) {
      if (broadBuf[i] > maxBroad) maxBroad = broadBuf[i];
      if (peakBuf[i] > maxPeak) maxPeak = peakBuf[i];
    }

    var heights = new Float32Array(G * G);
    for (dz = 0; dz < G; dz++) {
      var wzCell = -extent + (2 * extent * dz) / (G - 1);
      for (dx = 0; dx < G; dx++) {
        var wxCell = -extent + (2 * extent * dx) / (G - 1);
        var idx = dz * G + dx;
        var bNorm = broadBuf[idx] / maxBroad;
        // Soft saturation keeps rolling base foothills while preserving distinct valley/ridge relief
        bNorm = bNorm / (bNorm + 0.45) * 1.45;
        var pNorm = peakBuf[idx] / maxPeak;
        var radial = Math.hypot(wxCell, wzCell);
        var envMask = 1 - smoothstep(extent * 0.70, extent * 0.94, radial);
        var combined = Math.min(1, Math.max(0, 0.26 * Math.pow(bNorm, 1.05) + 0.74 * Math.pow(pNorm, 1.08)));
        heights[idx] = Math.pow(combined, 1.16) * envMask;
      }
    }

    // Single gentle final pass keeps peaks & saddles silky round while preserving crisp valley contrast
    separableBoxBlur(heights, tmpBuf, G, 1);
    var maxH = 1e-6;
    for (dz = 0; dz < G; dz++) {
      var wzEnv = -extent + (2 * extent * dz) / (G - 1);
      for (dx = 0; dx < G; dx++) {
        var wxEnv = -extent + (2 * extent * dx) / (G - 1);
        var idxEnv = dz * G + dx;
        var maskEnd = 1 - smoothstep(extent * 0.72, extent * 0.96, Math.hypot(wxEnv, wzEnv));
        heights[idxEnv] *= maskEnd;
        if (heights[idxEnv] > maxH) maxH = heights[idxEnv];
      }
    }
    if (maxH > 0) {
      for (i = 0; i < G * G; i++) heights[i] = Math.min(1, heights[i] / maxH);
    }

    var cellStep = (2 * extent) / (G - 1);
    var grads = new Float32Array(G * G * 2);
    for (dz = 0; dz < G; dz++) {
      var zm = Math.max(0, dz - 1);
      var zp = Math.min(G - 1, dz + 1);
      for (dx = 0; dx < G; dx++) {
        var xm = Math.max(0, dx - 1);
        var xp = Math.min(G - 1, dx + 1);
        var dhdx = (heights[dz * G + xp] - heights[dz * G + xm]) / ((xp - xm) * cellStep || 1);
        var dhdz = (heights[zp * G + dx] - heights[zm * G + dx]) / ((zp - zm) * cellStep || 1);
        var o2 = (dz * G + dx) * 2;
        grads[o2] = dhdx;
        grads[o2 + 1] = dhdz;
      }
    }

    var fieldObj = {
      gridSize: G,
      extent: extent,
      heights: heights,
      grads: grads,
      maxDegree: maxDeg,
      nodeMeta: nodeMeta,
      center: { x: cx, y: cy, halfSpan: halfSpan, coreSpan: coreSpan },
    };

    var ids = Object.keys(nodeMeta);
    if (!opts.skipOverlapResolve && ids.length >= 2) {
      var isPreviewField = list.length <= 160;
      var overlapList = [];
      for (i = 0; i < ids.length; i++) {
        var metaId = ids[i];
        var mItem = nodeMeta[metaId];
        if (!mItem || (!mItem.active && activeSet)) continue;
        overlapList.push({
          node: { id: metaId },
          meta: mItem,
          wx: mItem.wx,
          wz: mItem.wz,
          degNorm: mItem.degNorm,
          baseRadius: computeSphereRadius3D(mItem.deg, maxDeg, isPreviewField),
        });
      }
      resolveSphereOverlaps3D(overlapList, null, 0, {
        gap: opts.overlapGap != null ? opts.overlapGap : (isPreviewField ? 0.003 : 0.0008),
        maxPasses: opts.overlapPasses != null ? opts.overlapPasses : 6,
        pinnedId: opts.pinnedId || null,
      });
      for (i = 0; i < overlapList.length; i++) {
        overlapList[i].meta.wx = overlapList[i].wx;
        overlapList[i].meta.wz = overlapList[i].wz;
      }
    }

    for (i = 0; i < ids.length; i++) {
      var m = nodeMeta[ids[i]];
      var surfH = sampleTerrainHeight(fieldObj, m.wx, m.wz);
      m.surfaceH = surfH;
      // Elevated node floats proportionally to its degree on a vertical drop-stem
      m.elevH = Math.max(surfH + 0.045, surfH * 0.92 + 0.045 + m.degNorm * 0.24);
    }

    return fieldObj;
  }

  function sampleTerrainHeight(field, wx, wz) {
    if (!field || !field.heights) return 0;
    var G = field.gridSize;
    var ext = field.extent;
    var u = ((wx + ext) / (2 * ext)) * (G - 1);
    var v = ((wz + ext) / (2 * ext)) * (G - 1);
    if (u < 0 || u > G - 1 || v < 0 || v > G - 1) return 0;
    var x0 = Math.max(0, Math.min(G - 2, Math.floor(u)));
    var z0 = Math.max(0, Math.min(G - 2, Math.floor(v)));
    var tx = u - x0;
    var tz = v - z0;
    var h = field.heights;
    var h00 = h[z0 * G + x0];
    var h10 = h[z0 * G + x0 + 1];
    var h01 = h[(z0 + 1) * G + x0];
    var h11 = h[(z0 + 1) * G + x0 + 1];
    return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
  }

  /**
   * 采样节点球体水平投影足迹内（中心 + 周边 8 方向）的最高地形归一化高度，
   * 避免球体位于陡坡侧面时球缘切入上坡山体。
   */
  function sampleFootprintMaxHeight(field, wx, wz, radius) {
    var hMax = sampleTerrainHeight(field, wx, wz);
    if (!field || !(radius > 0.003)) return hMax;
    var r = radius * 1.05;
    var rd = r * 0.70710678;
    var s1 = sampleTerrainHeight(field, wx + r, wz);
    var s2 = sampleTerrainHeight(field, wx - r, wz);
    var s3 = sampleTerrainHeight(field, wx, wz + r);
    var s4 = sampleTerrainHeight(field, wx, wz - r);
    var s5 = sampleTerrainHeight(field, wx + rd, wz + rd);
    var s6 = sampleTerrainHeight(field, wx - rd, wz + rd);
    var s7 = sampleTerrainHeight(field, wx + rd, wz - rd);
    var s8 = sampleTerrainHeight(field, wx - rd, wz - rd);
    if (s1 > hMax) hMax = s1;
    if (s2 > hMax) hMax = s2;
    if (s3 > hMax) hMax = s3;
    if (s4 > hMax) hMax = s4;
    if (s5 > hMax) hMax = s5;
    if (s6 > hMax) hMax = s6;
    if (s7 > hMax) hMax = s7;
    if (s8 > hMax) hMax = s8;
    return hMax;
  }

  /**
   * 计算节点球心在世界坐标中的高度 wy：
   * 无论山峰高度参数（height）调高或调矮，球体底部（含悬停放大 1.10x）始终严格浮于
   * 足迹内最高地形表面之上并留出清晰的垂线悬浮净空，杜绝圆球嵌入山峰或坡面。
   */
  function computeNodeElevationY(surfaceH, footprintMaxH, degNorm, baseRadius, heightAmp, isPreview) {
    var surf = Math.max(surfaceH || 0, footprintMaxH || 0);
    var wySurfMax = surf * heightAmp;
    var dNorm = degNorm || 0;
    var baseElevH = Math.max(surf + 0.045, surf * 0.92 + 0.045 + dNorm * 0.24);
    var propWy = baseElevH * heightAmp;
    var r = baseRadius > 0 ? baseRadius : 0;
    var effR = r * 1.12;
    var minGap = isPreview
      ? (0.026 + r * 0.42 + dNorm * 0.018)
      : (0.012 + r * 0.45 + dNorm * 0.016);
    var minWy = wySurfMax + effR + minGap;
    return Math.max(propWy, minWy);
  }

  /* ── 3D 相机矩阵与屏幕空间投影（column-major 4x4）── */

  function mulMat4(a, b) {
    var out = new Float32Array(16);
    for (var col = 0; col < 4; col++) {
      for (var row = 0; row < 4; row++) {
        out[col * 4 + row] =
          a[row] * b[col * 4] +
          a[4 + row] * b[col * 4 + 1] +
          a[8 + row] * b[col * 4 + 2] +
          a[12 + row] * b[col * 4 + 3];
      }
    }
    return out;
  }

  function perspectiveMat4(fovRad, aspect, near, far) {
    var f = 1 / Math.tan(fovRad / 2);
    var nf = 1 / (near - far);
    var out = new Float32Array(16);
    out[0] = f / Math.max(0.1, aspect);
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[14] = 2 * far * near * nf;
    return out;
  }

  function lookAtMat4(eye, target, up) {
    var zx = eye[0] - target[0];
    var zy = eye[1] - target[1];
    var zz = eye[2] - target[2];
    var zLen = Math.hypot(zx, zy, zz) || 1;
    zx /= zLen; zy /= zLen; zz /= zLen;

    var xx = up[1] * zz - up[2] * zy;
    var xy = up[2] * zx - up[0] * zz;
    var xz = up[0] * zy - up[1] * zx;
    var xLen = Math.hypot(xx, xy, xz);
    if (xLen < 1e-6) {
      xx = 1; xy = 0; xz = 0;
    } else {
      xx /= xLen; xy /= xLen; xz /= xLen;
    }

    var yx = zy * xz - zz * xy;
    var yy = zz * xx - zx * xz;
    var yz = zx * xy - zy * xx;

    var out = new Float32Array(16);
    out[0] = xx; out[4] = xy; out[8] = xz;
    out[1] = yx; out[5] = yy; out[9] = yz;
    out[2] = zx; out[6] = zy; out[10] = zz;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
  }

  function buildCameraMatrices(params, camState, aspect) {
    var pitchRad = ((params.pitch != null ? params.pitch : 36) * Math.PI) / 180;
    var yawRad = ((params.yaw != null ? params.yaw : -18) * Math.PI) / 180;
    var dist = Math.max(0.55, Math.min(4.2, (camState && camState.distance) || 1.72));
    var effDist = dist / Math.min(1, Math.max(0.42, (aspect || 1) / 1.12));
    var tx = (camState && camState.targetX) || 0;
    var tz = (camState && camState.targetZ) || 0;
    var ty = (camState && camState.targetY) || 0.08;

    var cosP = Math.cos(pitchRad);
    var sinP = Math.sin(pitchRad);
    var cosY = Math.cos(yawRad);
    var sinY = Math.sin(yawRad);

    var eye = [
      tx + effDist * cosP * sinY,
      ty + effDist * sinP,
      tz + effDist * cosP * cosY,
    ];
    var view = lookAtMat4(eye, [tx, ty, tz], [0, 1, 0]);
    var proj = perspectiveMat4((42 * Math.PI) / 180, aspect, 0.04, 40);
    var viewProj = mulMat4(proj, view);
    return {
      eye: eye,
      target: [tx, ty, tz],
      view: view,
      proj: proj,
      viewProj: viewProj,
    };
  }

  function projectWorldToScreen(x, y, z, viewProj, width, height) {
    var cx = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
    var cy = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
    var cz = viewProj[2] * x + viewProj[6] * y + viewProj[10] * z + viewProj[14];
    var cw = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
    if (cw <= 0.001) return null;
    var invW = 1 / cw;
    var ndcX = cx * invW;
    var ndcY = cy * invW;
    var ndcZ = cz * invW;
    return [
      (ndcX * 0.5 + 0.5) * width,
      (0.5 - ndcY * 0.5) * height,
      ndcZ,
      cw,
    ];
  }

  function unprojectScreenToPlaneY(localX, localY, planeY, cam, width, height) {
    if (!cam || !cam.eye || !cam.view || !width || !height) return null;
    var aspect = width / Math.max(1, height);
    var ndcX = (localX / width) * 2 - 1;
    var ndcY = 1 - (localY / height) * 2;
    var tanHalf = Math.tan((21 * Math.PI) / 180);
    var v = cam.view;
    var rx = v[0]; var ry = v[4]; var rz = v[8];
    var ux = v[1]; var uy = v[5]; var uz = v[9];
    var fx = -v[2]; var fy = -v[6]; var fz = -v[10];
    var sx = ndcX * aspect * tanHalf;
    var sy = ndcY * tanHalf;
    var dx = fx + rx * sx + ux * sy;
    var dy = fy + ry * sx + uy * sy;
    var dz = fz + rz * sx + uz * sy;
    var len = Math.hypot(dx, dy, dz);
    if (!(len > 1e-6)) return null;
    dx /= len; dy /= len; dz /= len;
    if (Math.abs(dy) < 1e-4) return null;
    var t = (planeY - cam.eye[1]) / dy;
    if (!(t > 0.05 && t < 20)) return null;
    return [cam.eye[0] + t * dx, planeY, cam.eye[2] + t * dz];
  }

  function graphToWorldCoord(gVal, centerVal, halfSpan, coreSpan) {
    return ((gVal - centerVal) / (halfSpan || 1)) * (coreSpan || 0.78);
  }

  function worldToGraphCoords(wx, wz, center) {
    if (!center) return { x: wx, y: wz };
    var cs = center.coreSpan || 0.78;
    return {
      x: center.x + (wx / cs) * center.halfSpan,
      y: center.y + (wz / cs) * center.halfSpan,
    };
  }

  /**
   * 3D 地形节点防重叠解算（O(N) 空间网格 + 对称阻尼 Jacobi 松弛）：
   * 借鉴 2D/3D 立体力模拟的阻尼收敛机制，每轮累计所有相邻球体的合力后按 kPush / sqrt(冲突度)
   * 同步更新，保证位移随轮次严格单调衰减，彻底消除拥挤节点群的来回推挤震荡；
   * 大枢纽按半径平方反比稳居山峰顶、小节点环绕舒展。
   */
  function resolveSphereOverlaps3D(items, field, heightAmp, opts) {
    if (!items || items.length < 2) return 0;
    opts = opts || {};
    var gap = opts.gap != null ? opts.gap : 0.0008;
    var maxPasses = opts.maxPasses != null ? opts.maxPasses : 6;
    var kPush = opts.strength != null ? opts.strength : 0.68;
    var pinnedId = opts.pinnedId || null;
    var n = items.length;
    var maxR = 0;
    var i;
    for (i = 0; i < n; i++) {
      if (items[i].baseRadius > maxR) maxR = items[i].baseRadius;
    }
    var cellSize = Math.max(0.012, maxR * 2 + gap * 1.5);
    var invCell = 1 / cellSize;
    var tableSize = 1024;
    while (tableSize < n * 4) tableSize <<= 1;
    var mask = tableSize - 1;
    var head = new Int32Array(tableSize);
    var next = new Int32Array(n);
    var cellX = new Int32Array(n);
    var cellZ = new Int32Array(n);
    var accX = new Float32Array(n);
    var accZ = new Float32Array(n);
    var accW = new Float32Array(n);
    var movedAny = false;
    var pass;
    var totalResolved = 0;

    for (pass = 0; pass < maxPasses; pass++) {
      head.fill(-1);
      for (i = 0; i < n; i++) {
        accX[i] = 0;
        accZ[i] = 0;
        accW[i] = 0;
        var it = items[i];
        var gx = Math.floor(it.wx * invCell) | 0;
        var gz = Math.floor(it.wz * invCell) | 0;
        cellX[i] = gx;
        cellZ[i] = gz;
        var h = (Math.imul(gx, 73856093) ^ Math.imul(gz, 19349663)) & mask;
        next[i] = head[h];
        head[h] = i;
      }

      var passOverlaps = 0;
      for (i = 0; i < n; i++) {
        var a = items[i];
        var ra = a.baseRadius;
        var gxa = cellX[i];
        var gza = cellZ[i];
        var aPinned = pinnedId && a.node && a.node.id === pinnedId;
        var invMa = aPinned ? 0 : 1 / Math.max(1e-6, ra * ra);

        for (var oz = -1; oz <= 1; oz++) {
          var qz = (gza + oz) | 0;
          var hz = Math.imul(qz, 19349663);
          for (var ox = -1; ox <= 1; ox++) {
            var qx = (gxa + ox) | 0;
            var qh = (Math.imul(qx, 73856093) ^ hz) & mask;
            for (var j = head[qh]; j !== -1; j = next[j]) {
              if (j <= i || cellX[j] !== qx || cellZ[j] !== qz) continue;
              var b = items[j];
              var rb = b.baseRadius;
              var minDist = ra + rb + gap;
              var dx = b.wx - a.wx;
              var dz = b.wz - a.wz;
              var d2XZ = dx * dx + dz * dz;
              if (d2XZ >= minDist * minDist) continue;
              var dist = Math.sqrt(d2XZ);
              var ux;
              var uz;
              if (dist < 1e-5) {
                var ang = (((i * 2654435761 + j * 1597334677) >>> 0) % 1024) * (Math.PI * 2 / 1024);
                ux = Math.cos(ang);
                uz = Math.sin(ang);
              } else {
                ux = dx / dist;
                uz = dz / dist;
              }
              var bPinned = pinnedId && b.node && b.node.id === pinnedId;
              var invMb = bPinned ? 0 : 1 / Math.max(1e-6, rb * rb);
              var invSum = invMa + invMb;
              if (!(invSum > 0)) continue;
              var pen = minDist - dist;
              var wa = invMa / invSum;
              var wb = invMb / invSum;
              accX[i] -= ux * pen * wa;
              accZ[i] -= uz * pen * wa;
              accW[i] += 1;
              accX[j] += ux * pen * wb;
              accZ[j] += uz * pen * wb;
              accW[j] += 1;
              passOverlaps += 1;
              totalResolved += 1;
            }
          }
        }
      }
      if (passOverlaps === 0) break;
      var maxStep = 0;
      for (i = 0; i < n; i++) {
        if (accW[i] <= 0) continue;
        var damp = kPush / Math.sqrt(accW[i]);
        var mx = accX[i] * damp;
        var mz = accZ[i] * damp;
        items[i].wx += mx;
        items[i].wz += mz;
        var mv = Math.hypot(mx, mz);
        if (mv > maxStep) maxStep = mv;
        movedAny = true;
      }
      if (maxStep < 1e-4) break;
    }

    if (movedAny && field && heightAmp > 0) {
      var isPreviewList = n <= 160;
      for (i = 0; i < n; i++) {
        var pn = items[i];
        var surfH = sampleTerrainHeight(field, pn.wx, pn.wz);
        var footMaxH = sampleFootprintMaxHeight(field, pn.wx, pn.wz, pn.baseRadius);
        pn.wySurf = surfH * heightAmp;
        pn.wy = computeNodeElevationY(surfH, footMaxH, pn.degNorm, pn.baseRadius, heightAmp, isPreviewList);
      }
    }
    return totalResolved;
  }

  /**
   * 3D 地形入场力模拟放松动效（与 4 组顶点着色器 applyForceMotion 完全一致）：
   * 节点先按径向错峰由内向外爆发扩散，伴随微弱旋展解缠与欠阻尼弹簧过冲，最终精确落稳回真实 (wx, wz)。
   */
  function evalForceMotionXZ(wx, wz, settle, jitterAmp) {
    if (settle == null || settle >= 0.999) return [wx, wz];
    var r = Math.hypot(wx, wz);
    var rNorm = Math.max(0, Math.min(1, r / 1.05));
    var delay = rNorm * 0.24;
    var u = Math.max(0, Math.min(1, (settle - delay) / Math.max(0.01, 1 - delay)));
    var decay = Math.exp(-5.4 * u);
    var spring = 1 - decay * Math.cos(6.28318 * u);
    var blendEnd = smoothstep(0.84, 1.0, u);
    var springNorm = spring + (1 - spring) * blendEnd;
    var radialScale = 0.46 + (1 - 0.46) * springNorm;
    var swirl = (1 - springNorm) * (0.28 + 0.16 * rNorm);
    var cs = Math.cos(swirl);
    var sn = Math.sin(swirl);
    var ox = (wx * cs - wz * sn) * radialScale;
    var oz = (wx * sn + wz * cs) * radialScale;
    if (jitterAmp && jitterAmp > 0.0001) {
      var env = Math.sin(Math.PI * u) * Math.exp(-2.6 * u) * (1 - blendEnd);
      var ph1 = wx * 17.3 + wz * 11.7;
      var ph2 = wx * 13.1 - wz * 19.4;
      ox += Math.sin(ph1 + u * 10.5) * jitterAmp * env;
      oz += Math.cos(ph2 + u * 9.3) * jitterAmp * env;
    }
    return [ox, oz];
  }

  /* ── GLSL 着色器源码 ── */

  function compileShader(gl, type, source) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      var info = gl.getShaderInfoLog(sh) || 'Shader compile error';
      gl.deleteShader(sh);
      throw new Error(info);
    }
    return sh;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    var vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    var fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      var log = gl.getProgramInfoLog(prog) || 'Program link error';
      gl.deleteProgram(prog);
      throw new Error(log);
    }
    return prog;
  }

  function buildShaders(isWebGL2, hasDerivatives) {
    var headV = isWebGL2 ? '#version 300 es\nprecision highp float;\n' : 'precision highp float;\n';
    var headF = isWebGL2
      ? '#version 300 es\nprecision highp float;\n'
      : (hasDerivatives ? '#extension GL_OES_standard_derivatives : enable\nprecision highp float;\n' : 'precision highp float;\n');
    var attr = isWebGL2 ? 'in ' : 'attribute ';
    var varyV = isWebGL2 ? 'out ' : 'varying ';
    var varyF = isWebGL2 ? 'in ' : 'varying ';
    var outDecl = isWebGL2 ? 'out vec4 fragColor;\n' : '';
    var outAssign = isWebGL2 ? 'fragColor' : 'gl_FragColor';
    var derivFn = (isWebGL2 || hasDerivatives)
      ? 'float gradMag(float val, vec2 fb) { return max(fwidth(val), 0.0005); }'
      : 'float gradMag(float val, vec2 fb) { return max(length(fb) * 0.018, 0.012); }';

    var forceMotionGLSL =
      'vec2 applyForceMotion(vec2 xz, float settle, float jitterAmp) {\n' +
      '  if (settle >= 0.999) return xz;\n' +
      '  float r = length(xz);\n' +
      '  float rNorm = clamp(r / 1.45, 0.0, 1.0);\n' +
      '  float delay = rNorm * 0.24;\n' +
      '  float u = clamp((settle - delay) / max(0.01, 1.0 - delay), 0.0, 1.0);\n' +
      '  float decay = exp(-5.4 * u);\n' +
      '  float spring = 1.0 - decay * cos(6.28318 * u);\n' +
      '  float blendEnd = smoothstep(0.84, 1.0, u);\n' +
      '  float springNorm = mix(spring, 1.0, blendEnd);\n' +
      '  float radialScale = mix(0.46, 1.0, springNorm);\n' +
      '  float swirl = (1.0 - springNorm) * (0.28 + 0.16 * rNorm);\n' +
      '  float cs = cos(swirl);\n' +
      '  float sn = sin(swirl);\n' +
      '  vec2 outXZ = vec2(xz.x * cs - xz.y * sn, xz.x * sn + xz.y * cs) * radialScale;\n' +
      '  if (jitterAmp > 0.0001) {\n' +
      '    float env = sin(3.14159 * u) * exp(-2.6 * u) * (1.0 - blendEnd);\n' +
      '    float ph1 = xz.x * 17.3 + xz.y * 11.7;\n' +
      '    float ph2 = xz.x * 13.1 - xz.y * 19.4;\n' +
      '    outXZ += vec2(sin(ph1 + u * 10.5), cos(ph2 + u * 9.3)) * (jitterAmp * env);\n' +
      '  }\n' +
      '  return outXZ;\n' +
      '}\n';

    var lambertSphereGLSL =
      'vec3 srgbToLinear(vec3 c) {\n' +
      '  vec3 lo = c * 0.0773993808;\n' +
      '  vec3 hi = pow(max(c * 0.9478672986 + vec3(0.0521327014), vec3(0.000001)), vec3(2.4));\n' +
      '  return mix(lo, hi, step(vec3(0.04045), c));\n' +
      '}\n' +
      'vec3 linearToSrgb(vec3 c) {\n' +
      '  vec3 clamped = max(c, vec3(0.0));\n' +
      '  vec3 lo = clamped * 12.92;\n' +
      '  vec3 hi = pow(max(clamped, vec3(0.000001)), vec3(0.41666667)) * 1.055 - vec3(0.055);\n' +
      '  return clamp(mix(lo, hi, step(vec3(0.0031308), clamped)), 0.0, 1.0);\n' +
      '}\n' +
      'vec3 shadeLambertSphere(vec3 baseSrgb, vec3 nView, vec3 lightDirView) {\n' +
      '  vec3 baseLin = srgbToLinear(baseSrgb);\n' +
      '  float dotNL = clamp(dot(nView, normalize(lightDirView)), 0.0, 1.0);\n' +
      '  vec3 litLin = baseLin * (0.60382734 + 0.6 * dotNL);\n' +
      '  return linearToSrgb(litLin);\n' +
      '}\n';

    var terrainVS = headV +
      attr + 'vec2 aXZ;\n' +
      attr + 'float aH;\n' +
      attr + 'vec2 aGrad;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uHeightAmp;\n' +
      'uniform float uHeightScale;\n' +
      'uniform float uForceSettle;\n' +
      'uniform float uExtent;\n' +
      varyV + 'vec3 vWorld;\n' +
      varyV + 'float vH;\n' +
      varyV + 'vec2 vGrad;\n' +
      varyV + 'float vMeshRadial;\n' +
      varyV + 'float vHeightScale;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  float effAmp = uHeightAmp * uHeightScale;\n' +
      '  float y = aH * effAmp;\n' +
      '  vec2 mXZ = applyForceMotion(aXZ, uForceSettle, 0.0);\n' +
      '  vWorld = vec3(mXZ.x, y, mXZ.y);\n' +
      '  vH = aH;\n' +
      '  vGrad = aGrad * effAmp;\n' +
      '  vMeshRadial = length(aXZ) / max(0.001, uExtent);\n' +
      '  vHeightScale = uHeightScale;\n' +
      '  gl_Position = uViewProj * vec4(vWorld, 1.0);\n' +
      '}\n';

    var terrainFS = headF + outDecl + derivFn + '\n' +
      varyF + 'vec3 vWorld;\n' +
      varyF + 'float vH;\n' +
      varyF + 'vec2 vGrad;\n' +
      varyF + 'float vMeshRadial;\n' +
      varyF + 'float vHeightScale;\n' +
      'uniform vec3 uEye;\n' +
      'uniform vec3 uBg;\n' +
      'uniform float uIsDark;\n' +
      'uniform float uContours;\n' +
      'uniform float uExtent;\n' +
      'uniform float uGlintPhase;\n' +
      'vec3 terrainRampDark(float h) {\n' +
      '  vec3 c0 = vec3(0.095, 0.112, 0.138);\n' +
      '  vec3 c1 = vec3(0.115, 0.220, 0.340);\n' +
      '  vec3 c2 = vec3(0.245, 0.495, 0.820);\n' +
      '  vec3 c3 = vec3(0.915, 0.615, 0.275);\n' +
      '  vec3 c4 = vec3(0.990, 0.455, 0.245);\n' +
      '  if (h < 0.28) return mix(c0, c1, smoothstep(0.0, 0.28, h));\n' +
      '  if (h < 0.60) return mix(c1, c2, smoothstep(0.28, 0.60, h));\n' +
      '  if (h < 0.84) return mix(c2, c3, smoothstep(0.60, 0.84, h));\n' +
      '  return mix(c3, c4, smoothstep(0.84, 1.0, h));\n' +
      '}\n' +
      'vec3 terrainRampLight(float h) {\n' +
      '  vec3 c0 = vec3(0.940, 0.938, 0.925);\n' +
      '  vec3 c1 = vec3(0.815, 0.865, 0.935);\n' +
      '  vec3 c2 = vec3(0.415, 0.625, 0.885);\n' +
      '  vec3 c3 = vec3(0.195, 0.465, 0.845);\n' +
      '  vec3 c4 = vec3(0.860, 0.485, 0.155);\n' +
      '  if (h < 0.30) return mix(c0, c1, smoothstep(0.0, 0.30, h));\n' +
      '  if (h < 0.62) return mix(c1, c2, smoothstep(0.30, 0.62, h));\n' +
      '  if (h < 0.86) return mix(c2, c3, smoothstep(0.62, 0.86, h));\n' +
      '  return mix(c3, c4, smoothstep(0.86, 1.0, h));\n' +
      '}\n' +
      'void main() {\n' +
      '  vec3 N = normalize(vec3(-vGrad.x, 1.0, -vGrad.y));\n' +
      '  vec3 L = normalize(vec3(0.42, 0.86, 0.38));\n' +
      '  vec3 V = normalize(uEye - vWorld);\n' +
      '  vec3 LBack = normalize(vec3(-uEye.x, max(0.36, uEye.y * 0.42), -uEye.z));\n' +
      '  vec3 H1 = normalize(L + V);\n' +
      '  vec3 H2 = normalize(LBack + V);\n' +
      '  float diff = clamp(dot(N, L), 0.0, 1.0);\n' +
      '  float rim = pow(1.0 - max(dot(N, V), 0.0), 2.3);\n' +
      '  float specKey = pow(max(dot(N, H1), 0.0), 26.0);\n' +
      '  float specHorizon = pow(max(dot(N, H2), 0.0), 10.0) * pow(1.0 - max(dot(N, V), 0.0), 1.45);\n' +
      '  vec3 ramp = (uIsDark > 0.5) ? terrainRampDark(vH) : terrainRampLight(vH);\n' +
      '  vec3 lit = ramp * mix(0.64, 1.12, diff) + ramp * rim * (uIsDark > 0.5 ? 0.34 : 0.14);\n' +
      '\n' +
      '  float lvl = vH * uContours;\n' +
      '  float dfC = gradMag(lvl, vGrad * uContours);\n' +
      '  float distLine = abs(fract(lvl - 0.5) - 0.5) / dfC;\n' +
      '  float cLine = (1.0 - smoothstep(0.55, 1.55, distLine)) * smoothstep(0.015, 0.08, vH);\n' +
      '  float majMod = abs(fract(lvl * 0.25 - 0.5) - 0.5) * 4.0;\n' +
      '  float isMajor = 1.0 - smoothstep(0.18, 0.36, majMod);\n' +
      '  vec3 contourDark = mix(vec3(0.42, 0.65, 0.92), vec3(1.0, 0.72, 0.42), smoothstep(0.48, 0.92, vH));\n' +
      '  vec3 contourLight = mix(vec3(0.24, 0.45, 0.74), vec3(0.84, 0.42, 0.12), smoothstep(0.55, 0.95, vH));\n' +
      '  vec3 contourCol = (uIsDark > 0.5) ? contourDark : contourLight;\n' +
      '  float lineAlpha = cLine * mix(0.32, 0.88, smoothstep(0.05, 0.85, vH)) * mix(1.0, 1.38, isMajor);\n' +
      '  lit = mix(lit, contourCol, min(0.95, lineAlpha));\n' +
      '\n' +
      '  vec2 gCoord = vWorld.xz * 7.5;\n' +
      '  float gDf = 0.025;\n' +
      '  float gx = 1.0 - smoothstep(0.5, 1.5, abs(fract(gCoord.x - 0.5) - 0.5) / gDf);\n' +
      '  float gz = 1.0 - smoothstep(0.5, 1.5, abs(fract(gCoord.y - 0.5) - 0.5) / gDf);\n' +
      '  float riseVis = smoothstep(0.18, 0.68, vHeightScale);\n' +
      '  float gridFade = (1.0 - smoothstep(0.24, 0.72, vMeshRadial)) * riseVis;\n' +
      '  float gridLine = max(gx, gz) * (1.0 - smoothstep(0.02, 0.34, vH)) * gridFade;\n' +
      '  vec3 gridCol = (uIsDark > 0.5) ? vec3(0.38, 0.54, 0.78) : vec3(0.28, 0.44, 0.68);\n' +
      '  lit = mix(lit, gridCol, gridLine * (uIsDark > 0.5 ? 0.16 : 0.14));\n' +
      '\n' +
      '  float radial = max(vMeshRadial, length(vWorld.xz) / uExtent);\n' +
      '  vec2 toFrag = normalize(vWorld.xz + vec2(0.0001));\n' +
      '  vec2 camDir = normalize(-uEye.xz + vec2(0.0001));\n' +
      '  float farSide = smoothstep(-0.15, 0.86, dot(toFrag, camDir));\n' +
      '  vec3 glintTint = (uIsDark > 0.5)\n' +
      '    ? mix(vec3(0.74, 0.90, 1.0), vec3(1.0, 0.88, 0.66), smoothstep(0.38, 0.88, vH))\n' +
      '    : mix(vec3(0.38, 0.68, 1.0), vec3(1.0, 0.72, 0.34), smoothstep(0.42, 0.90, vH));\n' +
      '  float contourReflect = cLine * (0.34 * rim + 0.92 * specKey + 1.25 * specHorizon) * mix(0.75, 1.48, isMajor);\n' +
      '  float horizonSheen = farSide * smoothstep(0.26, 0.76, radial) * (1.0 - smoothstep(0.78, 0.92, radial))\n' +
      '    * (0.35 + 0.65 * (1.0 - smoothstep(0.0, 0.40, vH))) * riseVis;\n' +
      '  vec3 horizonTint = (uIsDark > 0.5)\n' +
      '    ? mix(vec3(0.42, 0.70, 1.0), vec3(1.0, 0.72, 0.40), smoothstep(0.25, 0.85, vH))\n' +
      '    : vec3(0.32, 0.58, 0.96);\n' +
      '  float sweepWave = (uGlintPhase >= -0.25 && uGlintPhase <= 1.25)\n' +
      '    ? exp(-pow((vH - uGlintPhase) * 6.0, 2.0)) : 0.0;\n' +
      '  lit += glintTint * (contourReflect * (uIsDark > 0.5 ? 0.78 : 0.48)\n' +
      '    + specHorizon * smoothstep(0.06, 0.85, vH) * (uIsDark > 0.5 ? 0.40 : 0.20)\n' +
      '    + sweepWave * (0.20 + cLine * 1.15) * (uIsDark > 0.5 ? 0.92 : 0.54));\n' +
      '  lit += horizonTint * horizonSheen * (uIsDark > 0.5 ? 0.58 : 0.30);\n' +
      '\n' +
      '  float peakGlow = smoothstep(0.76, 1.0, vH);\n' +
      '  if (uIsDark > 0.5) lit += vec3(0.28, 0.13, 0.04) * peakGlow;\n' +
      '\n' +
      '  float radialFade = 1.0 - smoothstep(0.52, 0.94, radial);\n' +
      '  float viewDist = length(uEye - vWorld);\n' +
      '  float viewFog = exp(-max(0.0, viewDist - 4.0) * 0.22);\n' +
      '  float vis = clamp(radialFade * viewFog, 0.0, 1.0);\n' +
      '  float valleySoft = clamp(smoothstep(0.008, 0.09, vH) + gridLine * 0.55 + cLine * 0.75 + horizonSheen * 0.45, 0.0, 1.0);\n' +
      '  ' + outAssign + ' = vec4(mix(uBg, lit, 0.92), vis * valleySoft);\n' +
      '}\n';

    var lineVS = headV +
      attr + 'vec3 aPos;\n' +
      attr + 'vec4 aColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uHeightScale;\n' +
      'uniform float uForceSettle;\n' +
      varyV + 'vec4 vColor;\n' +
      varyV + 'float vRadial;\n' +
      'uniform float uExtent;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  vec2 mXZ = applyForceMotion(aPos.xz, uForceSettle, 0.026);\n' +
      '  vec3 pos = vec3(mXZ.x, aPos.y * uHeightScale, mXZ.y);\n' +
      '  vColor = aColor;\n' +
      '  vRadial = length(pos.xz) / uExtent;\n' +
      '  gl_Position = uViewProj * vec4(pos, 1.0);\n' +
      '}\n';

    var lineFS = headF + outDecl +
      'uniform float uFadeStart;\n' +
      varyF + 'vec4 vColor;\n' +
      varyF + 'float vRadial;\n' +
      'void main() {\n' +
      '  float fade = 1.0 - smoothstep(uFadeStart, max(uFadeStart + 0.14, 1.08), vRadial);\n' +
      '  ' + outAssign + ' = vec4(vColor.rgb, vColor.a * fade);\n' +
      '}\n';

    var solidVS = headV +
      attr + 'vec3 aCenter;\n' +
      attr + 'vec3 aQuad;\n' +
      attr + 'vec4 aInstColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform vec3 uCamRight;\n' +
      'uniform vec3 uCamUp;\n' +
      'uniform float uHeightScale;\n' +
      'uniform float uForceSettle;\n' +
      varyV + 'vec2 vUv;\n' +
      varyV + 'vec4 vColor;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  vec2 mXZ = applyForceMotion(aCenter.xz, uForceSettle, 0.026);\n' +
      '  vec3 centerPos = vec3(mXZ.x, aCenter.y * uHeightScale, mXZ.y);\n' +
      '  float popScale = mix(0.62, 1.0, smoothstep(0.0, 0.35, uForceSettle));\n' +
      '  vec3 worldPos = centerPos + (uCamRight * aQuad.x + uCamUp * aQuad.y) * (aQuad.z * popScale);\n' +
      '  vUv = aQuad.xy;\n' +
      '  vColor = aInstColor;\n' +
      '  gl_Position = uViewProj * vec4(worldPos, 1.0);\n' +
      '}\n';

    var solidFS = headF + outDecl + derivFn + '\n' +
      varyF + 'vec2 vUv;\n' +
      varyF + 'vec4 vColor;\n' +
      'uniform vec3 uLightDirView;\n' +
      lambertSphereGLSL +
      'void main() {\n' +
      '  float r2 = dot(vUv, vUv);\n' +
      '  if (r2 > 1.0) discard;\n' +
      '  float r = sqrt(r2);\n' +
      '  float z = sqrt(max(0.0, 1.0 - r2));\n' +
      '  vec3 N = vec3(vUv.x, vUv.y, z);\n' +
      '  vec3 lit = shadeLambertSphere(vColor.rgb, N, uLightDirView);\n' +
      '  float aa = max(gradMag(r, vec2(0.025)), 0.014);\n' +
      '  float edgeAlpha = 1.0 - smoothstep(1.0 - aa * 1.4, 1.0, r);\n' +
      '  if (edgeAlpha < 0.02) discard;\n' +
      '  ' + outAssign + ' = vec4(lit, vColor.a * edgeAlpha);\n' +
      '}\n';

    var pointVS = headV +
      attr + 'vec3 aPos;\n' +
      attr + 'float aSize;\n' +
      attr + 'vec4 aColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uPointScale;\n' +
      'uniform float uHeightScale;\n' +
      'uniform float uForceSettle;\n' +
      varyV + 'vec4 vColor;\n' +
      varyV + 'float vPointSize;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  vec2 mXZ = applyForceMotion(aPos.xz, uForceSettle, 0.026);\n' +
      '  vec3 pos = vec3(mXZ.x, aPos.y * uHeightScale, mXZ.y);\n' +
      '  vColor = aColor;\n' +
      '  gl_Position = uViewProj * vec4(pos, 1.0);\n' +
      '  float popScale = mix(0.62, 1.0, smoothstep(0.0, 0.35, uForceSettle));\n' +
      '  float ptSize = clamp(aSize * popScale * uPointScale / max(0.45, gl_Position.w), 2.5, 64.0);\n' +
      '  gl_PointSize = ptSize;\n' +
      '  vPointSize = ptSize;\n' +
      '}\n';

    var pointFS = headF + outDecl +
      varyF + 'vec4 vColor;\n' +
      varyF + 'float vPointSize;\n' +
      'uniform vec3 uLightDirView;\n' +
      lambertSphereGLSL +
      'void main() {\n' +
      '  vec2 pc = gl_PointCoord * 2.0 - 1.0;\n' +
      '  float r2 = dot(pc, pc);\n' +
      '  if (r2 > 1.0) discard;\n' +
      '  float r = sqrt(r2);\n' +
      '  float z = sqrt(max(0.0, 1.0 - r2));\n' +
      '  vec3 N = vec3(pc.x, -pc.y, z);\n' +
      '  vec3 lit = shadeLambertSphere(vColor.rgb, N, uLightDirView);\n' +
      '  float aa = clamp(2.2 / max(vPointSize, 2.5), 0.03, 0.35);\n' +
      '  float edgeAlpha = 1.0 - smoothstep(1.0 - aa, 1.0, r);\n' +
      '  if (edgeAlpha < 0.02) discard;\n' +
      '  ' + outAssign + ' = vec4(lit, vColor.a * edgeAlpha);\n' +
      '}\n';

    return {
      terrainVS: terrainVS,
      terrainFS: terrainFS,
      lineVS: lineVS,
      lineFS: lineFS,
      solidVS: solidVS,
      solidFS: solidFS,
      pointVS: pointVS,
      pointFS: pointFS,
    };
  }

  /* ── WebGL 3D 地形渲染器 ── */

  function createWebGLTerrainRenderer(glCanvas, labelCanvas, opts) {
    var gl = glCanvas.getContext('webgl2', { antialias: true, alpha: true, powerPreference: 'high-performance' });
    var isWebGL2 = !!gl;
    var hasDerivatives = true;
    if (!gl) {
      gl = glCanvas.getContext('webgl', { antialias: true, alpha: true }) ||
        glCanvas.getContext('experimental-webgl', { antialias: true, alpha: true });
      hasDerivatives = !!(gl && gl.getExtension('OES_standard_derivatives'));
    }
    if (!gl) return null;

    var labelCtx = labelCanvas ? labelCanvas.getContext('2d') : null;
    var shaders = buildShaders(isWebGL2, hasDerivatives);
    var terrainProg = createProgram(gl, shaders.terrainVS, shaders.terrainFS);
    var lineProg = createProgram(gl, shaders.lineVS, shaders.lineFS);
    var solidProg = createProgram(gl, shaders.solidVS, shaders.solidFS);
    var pointProg = createProgram(gl, shaders.pointVS, shaders.pointFS);

    var terrainXZBuf = gl.createBuffer();
    var terrainHBuf = gl.createBuffer();
    var terrainGradBuf = gl.createBuffer();
    var terrainIdxBuf = gl.createBuffer();
    var terrainIdxCount = 0;
    var uploadedGridSize = 0;
    var uploadedExtent = -1;
    var cachedEdgeAdjRef = null;
    var cachedEdgeAdjLen = -1;
    var cachedEdgeAdj = null;

    function getEdgeAdj(edges) {
      if (cachedEdgeAdj && cachedEdgeAdjRef === edges && cachedEdgeAdjLen === edges.length) {
        return cachedEdgeAdj;
      }
      var map = Object.create(null);
      for (var ei = 0; ei < edges.length; ei++) {
        var edge = edges[ei];
        if (!edge) continue;
        var es = edgeEndId(edge.source);
        var et = edgeEndId(edge.target);
        if (!es || !et) continue;
        if (!map[es]) map[es] = [];
        map[es].push({ s: es, t: et });
        if (et !== es) {
          if (!map[et]) map[et] = [];
          map[et].push({ s: es, t: et });
        }
      }
      cachedEdgeAdjRef = edges;
      cachedEdgeAdjLen = edges.length;
      cachedEdgeAdj = map;
      return map;
    }

    var linePosBuf = gl.createBuffer();
    var lineColBuf = gl.createBuffer();
    var lineVertCount = 0;

    var hiLinePosBuf = gl.createBuffer();
    var hiLineColBuf = gl.createBuffer();
    var hiLineVertCount = 0;
    var hiEdgeCount = 0;

    var solidPosBuf = gl.createBuffer();
    var solidQuadBuf = gl.createBuffer();
    var solidInstColBuf = gl.createBuffer();
    var solidOpaqueVertCount = 0;
    var solidDimVertCount = 0;

    var ptPosBuf = gl.createBuffer();
    var ptSizeBuf = gl.createBuffer();
    var ptColBuf = gl.createBuffer();
    var ptOpaqueCount = 0;
    var ptDimCount = 0;

    var width = 1200;
    var height = 800;
    var dpr = 1;
    var currentField = null;
    var cachedNodes3d = [];
    var cachedPolyNodes = [];
    var lastViewProj = null;
    var lastCam = null;
    var lastHeightScale = 1;
    var lastForceSettle = 1;
    var lastCommunityLabelCount = 0;
    var lastRouteBadgeCount = 0;

    function resolveLocs(prog, attrs, uniforms) {
      var a = {};
      var u = {};
      var i;
      for (i = 0; i < attrs.length; i++) a[attrs[i]] = gl.getAttribLocation(prog, attrs[i]);
      for (i = 0; i < uniforms.length; i++) u[uniforms[i]] = gl.getUniformLocation(prog, uniforms[i]);
      return { a: a, u: u };
    }

    var tLoc = resolveLocs(terrainProg, ['aXZ', 'aH', 'aGrad'], ['uViewProj', 'uHeightAmp', 'uHeightScale', 'uForceSettle', 'uEye', 'uBg', 'uIsDark', 'uContours', 'uExtent', 'uGlintPhase']);
    var lLoc = resolveLocs(lineProg, ['aPos', 'aColor'], ['uViewProj', 'uExtent', 'uHeightScale', 'uForceSettle', 'uFadeStart']);
    var sLoc = resolveLocs(solidProg, ['aCenter', 'aQuad', 'aInstColor'], ['uViewProj', 'uCamRight', 'uCamUp', 'uLightDirView', 'uHeightScale', 'uForceSettle']);
    var pLoc = resolveLocs(pointProg, ['aPos', 'aSize', 'aColor'], ['uViewProj', 'uPointScale', 'uLightDirView', 'uHeightScale', 'uForceSettle']);

    function resize(w, h, nextDpr) {
      width = Math.max(1, Math.round(w));
      height = Math.max(1, Math.round(h));
      dpr = Math.max(1, Math.min(opts.dprCap || 2, nextDpr || 1));
      var pxW = Math.max(1, Math.round(width * dpr));
      var pxH = Math.max(1, Math.round(height * dpr));
      if (glCanvas.width !== pxW || glCanvas.height !== pxH) {
        glCanvas.width = pxW;
        glCanvas.height = pxH;
      }
      if (labelCanvas && (labelCanvas.width !== pxW || labelCanvas.height !== pxH)) {
        labelCanvas.width = pxW;
        labelCanvas.height = pxH;
      }
      gl.viewport(0, 0, pxW, pxH);
    }

    function uploadTerrainMesh(field) {
      currentField = field;
      var G = field.gridSize;
      var ext = field.extent;
      if (uploadedGridSize !== G || uploadedExtent !== ext) {
        var xz = new Float32Array(G * G * 2);
        for (var z = 0; z < G; z++) {
          var wz = -ext + (2 * ext * z) / (G - 1);
          for (var x = 0; x < G; x++) {
            var wx = -ext + (2 * ext * x) / (G - 1);
            var off = (z * G + x) * 2;
            xz[off] = wx;
            xz[off + 1] = wz;
          }
        }
        var indices = new Uint16Array((G - 1) * (G - 1) * 6);
        var ptr = 0;
        for (var r = 0; r < G - 1; r++) {
          for (var c = 0; c < G - 1; c++) {
            var i0 = r * G + c;
            var i1 = i0 + 1;
            var i2 = i0 + G;
            var i3 = i2 + 1;
            indices[ptr++] = i0; indices[ptr++] = i2; indices[ptr++] = i1;
            indices[ptr++] = i1; indices[ptr++] = i2; indices[ptr++] = i3;
          }
        }
        terrainIdxCount = indices.length;
        gl.bindBuffer(gl.ARRAY_BUFFER, terrainXZBuf);
        gl.bufferData(gl.ARRAY_BUFFER, xz, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, terrainIdxBuf);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
        uploadedGridSize = G;
        uploadedExtent = ext;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, terrainHBuf);
      gl.bufferData(gl.ARRAY_BUFFER, field.heights, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, terrainGradBuf);
      gl.bufferData(gl.ARRAY_BUFFER, field.grads, gl.STATIC_DRAW);
    }

    function buildDynamicBatches(data, heightAmp, isDarkTheme) {
      if (!currentField) return;
      var nodes = data.nodes || [];
      var edges = data.edges || [];
      var activeIds = data.activeIds || null;
      var focusSet = data.focusIds || null;
      var hoverNode = data.hoverNode || null;
      var sidebarNode = data.sidebarNode || null;
      var routeFocusId = data.routeFocusNodeId || null;
      var getNodeColor = data.getNodeColor || function () { return '#5b9cf6'; };
      var getNodeAlpha = data.getNodeAlpha || function () { return 1; };
      var showEdges = !!data.showEdges;
      var isLite = !!data.isLite;

      var metaMap = currentField.nodeMeta;
      var highlightCenterId = (hoverNode && hoverNode.id) || (sidebarNode && sidebarNode.id) || routeFocusId || null;
      var highlightNeighbors = null;
      var edgeAdj = (highlightCenterId && edges.length > 0) ? getEdgeAdj(edges) : null;
      var hiAdjList = (highlightCenterId && edgeAdj) ? edgeAdj[highlightCenterId] : null;
      var i;
      if (highlightCenterId) {
        highlightNeighbors = Object.create(null);
        if (hiAdjList) {
          for (i = 0; i < hiAdjList.length; i++) {
            var he = hiAdjList[i];
            if (he.s === highlightCenterId && he.t !== highlightCenterId) highlightNeighbors[he.t] = true;
            else if (he.t === highlightCenterId && he.s !== highlightCenterId) highlightNeighbors[he.s] = true;
          }
        }
      }
      var applyHoverDim = !!(hoverNode && !focusSet);

      var isPreview = nodes.length <= 160;
      var maxDeg = currentField.maxDegree || 1;
      cachedNodes3d = [];
      for (i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        if (!n || !n.id) continue;
        var m = metaMap[n.id];
        if (!m) continue;
        var visible = !activeIds || activeIds.has(n.id);
        var alpha = getNodeAlpha(n);
        if (!visible && alpha <= 0.08 && (!focusSet || !focusSet.has(n.id))) continue;
        var rgb = parseCssColorFast(getNodeColor(n), [0.36, 0.61, 0.96, 1]);
        var isFocused = !!(focusSet && focusSet.has(n.id));
        var isHovered = !!(hoverNode && hoverNode.id === n.id);
        var isSelected = !!((sidebarNode && sidebarNode.id === n.id) || (routeFocusId && routeFocusId === n.id));
        var isNeighbor = !!(highlightNeighbors && highlightNeighbors[n.id]);
        var nodeAlpha = visible ? Math.max(0.2, alpha) : 0.14;
        if (applyHoverDim && !isHovered && !isSelected && !isNeighbor) {
          nodeAlpha = Math.min(nodeAlpha, 0.28);
        } else if (isHovered || isSelected || isNeighbor) {
          nodeAlpha = 1;
        }
        var baseRadius = computeSphereRadius3D(m.deg, maxDeg, isPreview);
        var footMaxH = sampleFootprintMaxHeight(currentField, m.wx, m.wz, baseRadius);
        cachedNodes3d.push({
          node: n,
          wx: m.wx,
          wz: m.wz,
          wySurf: m.surfaceH * heightAmp,
          wy: computeNodeElevationY(m.surfaceH, footMaxH, m.degNorm, baseRadius, heightAmp, isPreview),
          degNorm: m.degNorm,
          deg: m.deg,
          baseRadius: baseRadius,
          rgb: rgb,
          alpha: nodeAlpha,
          visible: visible,
          hovered: isHovered,
          selected: isSelected,
          focused: isFocused || isHovered || isSelected,
          neighbor: isNeighbor,
        });
      }

      // Rank top hubs for 3D spherical orbs + vertical stems
      var stemCap = isLite ? 55 : (isPreview ? cachedNodes3d.length : 170);
      var ranked = cachedNodes3d.slice().sort(function (a, b) {
        if (a.focused !== b.focused) return a.focused ? -1 : 1;
        if (a.neighbor !== b.neighbor) return a.neighbor ? -1 : 1;
        if (a.visible !== b.visible) return a.visible ? -1 : 1;
        return b.deg - a.deg;
      });
      var polySet = Object.create(null);
      cachedPolyNodes = [];
      for (i = 0; i < ranked.length; i++) {
        var item = ranked[i];
        if (i < stemCap || item.focused || item.neighbor) {
          polySet[item.node.id] = item;
          cachedPolyNodes.push(item);
        }
      }

      // Background points buffer (remaining lower-degree nodes on large graphs, sharing exact world baseRadius; opaque first, dimmed second)
      var opaquePts = [];
      var dimPts = [];
      for (i = 0; i < cachedNodes3d.length; i++) {
        var bn = cachedNodes3d[i];
        if (polySet[bn.node.id]) continue;
        if (bn.alpha >= 0.5) opaquePts.push(bn);
        else dimPts.push(bn);
      }
      var orderedPts = opaquePts.concat(dimPts);
      var bgPts = [];
      var bgSizes = [];
      var bgCols = [];
      for (i = 0; i < orderedPts.length; i++) {
        var pb = orderedPts[i];
        bgPts.push(pb.wx, pb.wy, pb.wz);
        bgSizes.push(pb.baseRadius);
        bgCols.push(pb.rgb[0], pb.rgb[1], pb.rgb[2], pb.alpha);
      }
      ptOpaqueCount = opaquePts.length;
      ptDimCount = dimPts.length;
      gl.bindBuffer(gl.ARRAY_BUFFER, ptPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bgPts), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ptSizeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bgSizes), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ptColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bgCols), gl.STATIC_DRAW);

      // Build batched camera-facing billboard sphere quads (opaque hubs first, dimmed hubs second)
      var quadCorners = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
      var opaqueHubs = [];
      var dimHubs = [];
      for (i = 0; i < cachedPolyNodes.length; i++) {
        if (cachedPolyNodes[i].alpha >= 0.5) opaqueHubs.push(cachedPolyNodes[i]);
        else dimHubs.push(cachedPolyNodes[i]);
      }
      var orderedHubs = opaqueHubs.concat(dimHubs);
      var sPos = [];
      var sQuad = [];
      var sCol = [];
      for (i = 0; i < orderedHubs.length; i++) {
        var pn = orderedHubs[i];
        var radius = pn.baseRadius;
        if (pn.hovered) radius *= 1.10;
        else if (pn.selected || pn.focused) radius *= 1.06;
        for (var q = 0; q < quadCorners.length; q += 2) {
          sPos.push(pn.wx, pn.wy, pn.wz);
          sQuad.push(quadCorners[q], quadCorners[q + 1], radius);
          sCol.push(pn.rgb[0], pn.rgb[1], pn.rgb[2], pn.alpha);
        }
      }
      solidOpaqueVertCount = opaqueHubs.length * 6;
      solidDimVertCount = dimHubs.length * 6;
      gl.bindBuffer(gl.ARRAY_BUFFER, solidPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sPos), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, solidQuadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sQuad), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, solidInstColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sCol), gl.STATIC_DRAW);

      // Build line segments: vertical stems + peak rays + arched 3D edges
      var lPos = [];
      var lCol = [];
      var hlPos = [];
      var hlCol = [];

      for (i = 0; i < cachedPolyNodes.length; i++) {
        var sn = cachedPolyNodes[i];
        var stemA = Math.min(0.85, sn.alpha * 0.72);
        lPos.push(sn.wx, sn.wySurf, sn.wz, sn.wx, sn.wy, sn.wz);
        lCol.push(
          sn.rgb[0], sn.rgb[1], sn.rgb[2], stemA * 0.25,
          sn.rgb[0], sn.rgb[1], sn.rgb[2], stemA
        );
        // Subtle upward light pillar on peak summits
        if (sn.degNorm > 0.62 && sn.visible) {
          var beamTop = sn.wy + 0.14 + sn.degNorm * 0.16;
          lPos.push(sn.wx, sn.wy, sn.wz, sn.wx, beamTop, sn.wz);
          lCol.push(
            sn.rgb[0], sn.rgb[1], sn.rgb[2], 0.48,
            sn.rgb[0], sn.rgb[1], sn.rgb[2], 0.0
          );
        }
      }

      var node3dMap = Object.create(null);
      for (i = 0; i < cachedNodes3d.length; i++) {
        node3dMap[cachedNodes3d[i].node.id] = cachedNodes3d[i];
      }

      var maxBgEdges = isLite ? 1400 : 4200;
      var bgEdgeDrawn = 0;
      hiEdgeCount = 0;
      var baseEdgeRgb = isDarkTheme ? [0.68, 0.78, 0.92] : [0.24, 0.36, 0.56];

      function pushRibbonStrip(pts, nx, ny, nz, halfW, rgb, alphaCenter, alphaEdge) {
        for (var k = 0; k < pts.length - 1; k++) {
          var p0 = pts[k];
          var p1 = pts[k + 1];
          // Taper ribbon slightly at endpoints so it docks neatly into spheres
          var t0 = Math.min(1, k / 2, (pts.length - 1 - k) / 2);
          var t1 = Math.min(1, (k + 1) / 2, (pts.length - 2 - k) / 2);
          var w0 = halfW * (0.55 + 0.45 * t0);
          var w1 = halfW * (0.55 + 0.45 * t1);
          var ox0 = nx * w0; var oy0 = ny * w0; var oz0 = nz * w0;
          var ox1 = nx * w1; var oy1 = ny * w1; var oz1 = nz * w1;
          var r = rgb[0]; var g = rgb[1]; var bl = rgb[2];
          // +side quad (center -> +offset): 2 triangles
          hlPos.push(
            p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p1[0] + ox1, p1[1] + oy1, p1[2] + oz1,
            p0[0], p0[1], p0[2], p1[0] + ox1, p1[1] + oy1, p1[2] + oz1, p0[0] + ox0, p0[1] + oy0, p0[2] + oz0
          );
          hlCol.push(
            r, g, bl, alphaCenter, r, g, bl, alphaCenter, r, g, bl, alphaEdge,
            r, g, bl, alphaCenter, r, g, bl, alphaEdge, r, g, bl, alphaEdge
          );
          // -side quad (center -> -offset): 2 triangles
          hlPos.push(
            p0[0], p0[1], p0[2], p0[0] - ox0, p0[1] - oy0, p0[2] - oz0, p1[0] - ox1, p1[1] - oy1, p1[2] - oz1,
            p0[0], p0[1], p0[2], p1[0] - ox1, p1[1] - oy1, p1[2] - oz1, p1[0], p1[1], p1[2]
          );
          hlCol.push(
            r, g, bl, alphaCenter, r, g, bl, alphaEdge, r, g, bl, alphaEdge,
            r, g, bl, alphaCenter, r, g, bl, alphaEdge, r, g, bl, alphaCenter
          );
        }
      }

      if (hiAdjList) {
        var hiColBase = node3dMap[highlightCenterId] ? node3dMap[highlightCenterId].rgb : null;
        var hiCoreMix = isDarkTheme ? 0.36 : 0.08;
        var wScaleHi = isPreview ? 1.0 : 0.62;
        for (i = 0; i < hiAdjList.length; i++) {
          var heItem = hiAdjList[i];
          var aHi = node3dMap[heItem.s];
          var bHi = node3dMap[heItem.t];
          if (!aHi || !bHi) continue;
          var mxHi = (aHi.wx + bHi.wx) * 0.5;
          var mzHi = (aHi.wz + bHi.wz) * 0.5;
          var distHi = Math.hypot(bHi.wx - aHi.wx, bHi.wz - aHi.wz);
          var terrainMidHi = sampleTerrainHeight(currentField, mxHi, mzHi) * heightAmp;
          var arcPeakHi = Math.max(aHi.wy, bHi.wy, terrainMidHi + 0.055) + Math.min(0.18, distHi * 0.22 + 0.032);
          var colHi = hiColBase || aHi.rgb;
          hiEdgeCount += 1;
          var coreCol = [
            colHi[0] + (1 - colHi[0]) * hiCoreMix,
            colHi[1] + (1 - colHi[1]) * hiCoreMix,
            colHi[2] + (1 - colHi[2]) * hiCoreMix,
          ];
          var perpLen = Math.max(1e-4, distHi);
          var sx = -(bHi.wz - aHi.wz) / perpLen;
          var sz = (bHi.wx - aHi.wx) / perpLen;
          var hiSegs = 14;
          var arcPts = [];
          for (var hsg = 0; hsg <= hiSegs; hsg++) {
            var hu = hsg / hiSegs;
            var homt = 1 - hu;
            arcPts.push([
              homt * homt * aHi.wx + 2 * homt * hu * mxHi + hu * hu * bHi.wx,
              homt * homt * aHi.wy + 2 * homt * hu * arcPeakHi + hu * hu * bHi.wy,
              homt * homt * aHi.wz + 2 * homt * hu * mzHi + hu * hu * bHi.wz,
            ]);
          }
          // Fine crossed horizontal + vertical luminous 3D ribbons (~1.6-2.0px crisp line across viewports)
          pushRibbonStrip(arcPts, sx, 0, sz, 0.0032 * wScaleHi, colHi, isDarkTheme ? 0.24 : 0.20, 0.0);
          pushRibbonStrip(arcPts, sx, 0, sz, 0.0013 * wScaleHi, coreCol, 0.94, 0.15);
          pushRibbonStrip(arcPts, 0, 1, 0, 0.0026 * wScaleHi, colHi, isDarkTheme ? 0.20 : 0.16, 0.0);
          pushRibbonStrip(arcPts, 0, 1, 0, 0.0011 * wScaleHi, coreCol, 0.88, 0.12);
        }
      }

      if (showEdges) {
        var bgEdgeAlpha = highlightCenterId
          ? (nodes.length <= 160 ? 0.04 : 0.03)
          : (nodes.length <= 160 ? 0.16 : 0.10);
        for (i = 0; i < edges.length && bgEdgeDrawn < maxBgEdges; i++) {
          var e = edges[i];
          if (!e) continue;
          var sId = edgeEndId(e.source);
          var tId = edgeEndId(e.target);
          if (highlightCenterId && (sId === highlightCenterId || tId === highlightCenterId)) continue;
          var a = node3dMap[sId];
          var b = node3dMap[tId];
          if (!a || !b || !a.visible || !b.visible) continue;

          var mx = (a.wx + b.wx) * 0.5;
          var mz = (a.wz + b.wz) * 0.5;
          var dist = Math.hypot(b.wx - a.wx, b.wz - a.wz);
          var terrainMid = sampleTerrainHeight(currentField, mx, mz) * heightAmp;
          var arcPeak = Math.max((a.wy + b.wy) * 0.5, terrainMid + 0.03) + Math.min(0.14, dist * 0.16);
          var px0 = a.wx;
          var py0 = a.wy;
          var pz0 = a.wz;
          for (var sg = 1; sg <= 3; sg++) {
            var u = sg / 3;
            var omt = 1 - u;
            var qx = omt * omt * a.wx + 2 * omt * u * mx + u * u * b.wx;
            var qy = omt * omt * a.wy + 2 * omt * u * arcPeak + u * u * b.wy;
            var qz = omt * omt * a.wz + 2 * omt * u * mz + u * u * b.wz;
            lPos.push(px0, py0, pz0, qx, qy, qz);
            lCol.push(baseEdgeRgb[0], baseEdgeRgb[1], baseEdgeRgb[2], bgEdgeAlpha, baseEdgeRgb[0], baseEdgeRgb[1], baseEdgeRgb[2], bgEdgeAlpha);
            px0 = qx; py0 = qy; pz0 = qz;
          }
          bgEdgeDrawn += 1;
        }
      }

      lineVertCount = lPos.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lPos), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lCol), gl.STATIC_DRAW);

      hiLineVertCount = hlPos.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, hiLinePosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hlPos), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, hiLineColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hlCol), gl.STATIC_DRAW);
    }

    var maxAttribs = 4;
    function useAttrs(activeMap) {
      for (var i = 0; i < maxAttribs; i++) {
        if (activeMap[i]) gl.enableVertexAttribArray(i);
        else gl.disableVertexAttribArray(i);
      }
    }

    function bindAttrib(loc, buf, size, activeMap) {
      if (loc < 0) return;
      activeMap[loc] = true;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    }

    function drawLabelsOverlay(viewProj, renderOpts) {
      if (!labelCtx || !labelCanvas) return;
      labelCtx.setTransform(1, 0, 0, 1, 0, 0);
      labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
      if (renderOpts.showLabels === false) return;

      labelCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var maxLabels = renderOpts.isLite ? 14 : (cachedNodes3d.length <= 160 ? 28 : 34);
      var isDarkTheme = !!renderOpts.isDark;
      var getLabelText = renderOpts.getLabelText || function (n) { return n.label || n.id; };
      var camDist = renderOpts.distance || 1.72;
      var heightScale = renderOpts.heightScale != null ? renderOpts.heightScale : 1;
      var forceSettle = renderOpts.forceSettle != null ? renderOpts.forceSettle : 1;
      var heightAmp = renderOpts.heightAmp != null ? renderOpts.heightAmp : 0.46;
      var labelFade = renderOpts.labelFade != null ? renderOpts.labelFade : 1;
      var communityLabels = Array.isArray(renderOpts.communityLabels) ? renderOpts.communityLabels : [];
      var routeBadges = Array.isArray(renderOpts.routeBadges) ? renderOpts.routeBadges : [];
      lastCommunityLabelCount = 0;
      lastRouteBadgeCount = 0;
      if (labelFade < 0.04) return;
      if (camDist < 1.1) maxLabels = Math.round(maxLabels * 1.45);

      var placed = [];
      var hasHoverFocus = !!renderOpts.hasHoverHighlight;
      var i;
      var j;

      // 0. Numbered route step badges (mirrors 2D .node-route-badge / 3D .graph-3d-route-badge)
      if (routeBadges.length && cachedNodes3d.length) {
        var node3dById = Object.create(null);
        for (i = 0; i < cachedNodes3d.length; i++) {
          var cn = cachedNodes3d[i];
          if (cn.node && cn.node.id) node3dById[cn.node.id] = cn;
        }
        // Draw non-current badges first, then 'current' badge on top
        var orderedBadges = routeBadges.slice().sort(function (a, b) {
          return (a.state === 'current' ? 1 : 0) - (b.state === 'current' ? 1 : 0);
        });
        labelCtx.textAlign = 'center';
        labelCtx.textBaseline = 'middle';
        for (i = 0; i < orderedBadges.length; i++) {
          var rb = orderedBadges[i];
          var rn3 = rb && node3dById[rb.id];
          if (!rn3) continue;
          var rMoved = evalForceMotionXZ(rn3.wx, rn3.wz, forceSettle, 0.026);
          var rPt = projectWorldToScreen(rMoved[0], rn3.wy * heightScale, rMoved[1], viewProj, width, height);
          if (!rPt || rPt[2] < -1 || rPt[2] > 1) continue;
          if (rPt[0] < 14 || rPt[0] > width - 14 || rPt[1] < 14 || rPt[1] > height - 14) continue;
          var isCurr = rb.state === 'current';
          var isDone = rb.state === 'done';
          var isTodo = rb.state === 'todo';
          var bRad = isCurr ? 10 : 8;
          var bx = rPt[0] + 10;
          var by = rPt[1] - 10;
          labelCtx.save();
          labelCtx.globalAlpha = (isTodo ? 0.48 : 0.96) * labelFade;
          if (isCurr) {
            labelCtx.shadowColor = 'rgba(255, 181, 71, 0.92)';
            labelCtx.shadowBlur = 10;
          }
          labelCtx.beginPath();
          labelCtx.arc(bx, by, bRad, 0, Math.PI * 2);
          labelCtx.fillStyle = isCurr ? '#ffb547' : (isDone ? 'rgba(122, 139, 160, 0.92)' : '#5b9cf6');
          labelCtx.fill();
          labelCtx.shadowBlur = 0;
          labelCtx.lineWidth = isCurr ? 1.8 : 1.2;
          labelCtx.strokeStyle = isDarkTheme ? 'rgba(13, 17, 23, 0.92)' : 'rgba(255, 255, 255, 0.96)';
          labelCtx.stroke();
          labelCtx.font = '700 ' + (isCurr ? '10' : '9') + 'px ui-sans-serif, system-ui, sans-serif';
          labelCtx.fillStyle = isCurr ? '#1a1a1a' : '#ffffff';
          labelCtx.fillText(String(rb.index), bx, by + 0.5);
          labelCtx.restore();
          lastRouteBadgeCount += 1;
        }
      }

      // 1. Floating community cluster capsules (mirrors 2D .community-label / 3D .graph-3d-community-label)
      if (communityLabels.length && currentField && currentField.center) {
        var cCenter = currentField.center;
        var commCandidates = [];
        for (i = 0; i < communityLabels.length; i++) {
          var cl = communityLabels[i];
          if (!cl || !cl.label || cl.x == null || cl.y == null) continue;
          var cWx = graphToWorldCoord(cl.x, cCenter.x, cCenter.halfSpan, cCenter.coreSpan);
          var cWz = graphToWorldCoord(cl.y, cCenter.y, cCenter.halfSpan, cCenter.coreSpan);
          var cSurfH = sampleTerrainHeight(currentField, cWx, cWz);
          var cWy = (cSurfH + 0.14) * heightAmp * heightScale;
          var cMoved = evalForceMotionXZ(cWx, cWz, forceSettle, 0.0);
          var cPt = projectWorldToScreen(cMoved[0], cWy, cMoved[1], viewProj, width, height);
          if (!cPt || cPt[2] < -1 || cPt[2] > 1) continue;
          if (cPt[0] < 24 || cPt[0] > width - 24 || cPt[1] < 24 || cPt[1] > height - 24) continue;
          commCandidates.push({
            cl: cl,
            sx: cPt[0],
            sy: cPt[1],
            count: Number(cl.count) || 0,
            depth: cPt[2],
          });
        }
        commCandidates.sort(function (a, b) {
          if (b.count !== a.count) return b.count - a.count;
          return a.depth - b.depth;
        });

        var commAlpha = (hasHoverFocus ? 0.28 : 0.92) * labelFade;
        labelCtx.textAlign = 'center';
        labelCtx.textBaseline = 'middle';
        for (i = 0; i < commCandidates.length; i++) {
          var cc = commCandidates[i];
          var fs = Math.max(9, Math.min(16, Number(cc.cl.fontSize) || 12));
          if (renderOpts.isLite) fs = Math.max(9, fs * 0.86);
          labelCtx.font = '700 ' + fs.toFixed(1) + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
          var cTxt = String(cc.cl.label);
          var cPadX = Math.max(7, fs * 0.72);
          var cPadY = Math.max(3.5, fs * 0.34);
          var cTw = labelCtx.measureText(cTxt).width + cPadX * 2;
          var cTh = fs + cPadY * 2;
          var cDrawX = Math.max(cTw / 2 + 12, Math.min(width - cTw / 2 - 12, cc.sx));
          if (Math.abs(cDrawX - cc.sx) > 28) continue;
          var cBx0 = cDrawX - cTw / 2;
          var cBy0 = cc.sy - cTh / 2;
          if (cBy0 < 14 || cBy0 + cTh > height - 28) continue;
          if (cBy0 < 54 && cBx0 + cTw > width - 148) continue;
          var cOverlap = false;
          for (j = 0; j < placed.length; j++) {
            var cPb = placed[j];
            if (cBx0 < cPb[2] + 6 && cBx0 + cTw > cPb[0] - 6 && cBy0 < cPb[3] + 5 && cBy0 + cTh > cPb[1] - 5) {
              cOverlap = true;
              break;
            }
          }
          if (cOverlap) continue;
          placed.push([cBx0, cBy0, cBx0 + cTw, cBy0 + cTh]);

          labelCtx.save();
          labelCtx.globalAlpha = commAlpha;
          labelCtx.shadowColor = isDarkTheme ? 'rgba(0, 0, 0, 0.38)' : 'rgba(15, 23, 42, 0.20)';
          labelCtx.shadowBlur = isDarkTheme ? 6 : 5;
          labelCtx.shadowOffsetY = 1;
          labelCtx.fillStyle = cc.cl.bgColor || '#5b9cf6';
          labelCtx.beginPath();
          if (typeof labelCtx.roundRect === 'function') {
            labelCtx.roundRect(cBx0, cBy0, cTw, cTh, cTh / 2);
          } else {
            labelCtx.rect(cBx0, cBy0, cTw, cTh);
          }
          labelCtx.fill();
          labelCtx.shadowColor = 'transparent';
          labelCtx.shadowBlur = 0;
          labelCtx.globalAlpha = (hasHoverFocus ? 0.36 : 0.98) * labelFade;
          labelCtx.fillStyle = cc.cl.textColor || '#ffffff';
          labelCtx.fillText(cTxt, cDrawX, cc.sy + 0.5);
          labelCtx.restore();
          lastCommunityLabelCount += 1;
        }
      }

      var candidates = [];
      for (i = 0; i < cachedPolyNodes.length; i++) {
        var n3 = cachedPolyNodes[i];
        if (!n3.visible && !n3.focused) continue;
        if (hasHoverFocus && !n3.focused && !n3.neighbor) continue;
        var mXZ = evalForceMotionXZ(n3.wx, n3.wz, forceSettle, 0.026);
        var labelLift = Math.max(0.025, (n3.baseRadius || 0) * 0.92 + 0.014);
        var pt = projectWorldToScreen(mXZ[0], n3.wy * heightScale + labelLift, mXZ[1], viewProj, width, height);
        if (!pt) continue;
        if (pt[0] < 24 || pt[0] > width - 24 || pt[1] < 24 || pt[1] > height - 24 || pt[2] < -1 || pt[2] > 1) continue;
        candidates.push({
          n3: n3,
          sx: pt[0],
          sy: pt[1] - 11,
          priority: (n3.focused ? 1000000 : (n3.neighbor ? 100000 : 0)) + n3.deg * 10 - pt[2],
        });
      }
      candidates.sort(function (a, b) { return b.priority - a.priority; });

      labelCtx.globalAlpha = labelFade;
      labelCtx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
      labelCtx.textAlign = 'center';
      labelCtx.textBaseline = 'middle';

      var drawn = 0;
      for (i = 0; i < candidates.length && drawn < maxLabels; i++) {
        var c = candidates[i];
        var rawTxt = String(getLabelText(c.n3.node) || c.n3.node.id);
        var txt = rawTxt.length > 18 ? rawTxt.slice(0, 17) + '…' : rawTxt;
        var tw = labelCtx.measureText(txt).width + 12;
        var th = 19;
        var drawX = Math.max(tw / 2 + 12, Math.min(width - tw / 2 - 12, c.sx));
        if (Math.abs(drawX - c.sx) > 16) continue;
        var bx0 = drawX - tw / 2;
        var by0 = c.sy - th / 2;
        if (by0 < 14 || by0 + th > height - 28) continue;
        if (by0 < 50 && bx0 + tw > width - 144) continue;
        var overlaps = false;
        if (!c.n3.focused) {
          for (j = 0; j < placed.length; j++) {
            var pb = placed[j];
            if (bx0 < pb[2] + 4 && bx0 + tw > pb[0] - 4 && by0 < pb[3] + 3 && by0 + th > pb[1] - 3) {
              overlaps = true;
              break;
            }
          }
        }
        if (overlaps) continue;
        placed.push([bx0, by0, bx0 + tw, by0 + th]);

        labelCtx.fillStyle = isDarkTheme ? 'rgba(18, 22, 28, 0.78)' : 'rgba(255, 255, 255, 0.86)';
        labelCtx.strokeStyle = c.n3.focused
          ? 'rgba(' + Math.round(c.n3.rgb[0] * 255) + ',' + Math.round(c.n3.rgb[1] * 255) + ',' + Math.round(c.n3.rgb[2] * 255) + ', 0.85)'
          : (isDarkTheme ? 'rgba(255,255,255,0.14)' : 'rgba(15,23,42,0.14)');
        labelCtx.lineWidth = 1;

        labelCtx.beginPath();
        if (typeof labelCtx.roundRect === 'function') {
          labelCtx.roundRect(bx0, by0, tw, th, 5);
        } else {
          labelCtx.rect(bx0, by0, tw, th);
        }
        labelCtx.fill();
        labelCtx.stroke();

        labelCtx.fillStyle = isDarkTheme ? 'rgba(236, 240, 244, 0.94)' : 'rgba(30, 32, 36, 0.92)';
        labelCtx.fillText(txt, drawX, c.sy + 0.5);
        drawn += 1;
      }
      labelCtx.globalAlpha = 1;
    }

    function render(params, camState, renderOpts) {
      if (!currentField) return;
      var aspect = width / Math.max(1, height);
      var cam = buildCameraMatrices(params, camState, aspect);
      lastViewProj = cam.viewProj;
      lastCam = cam;
      var heightAmp = (params.height / 100) * 0.46;
      var heightScale = renderOpts.heightScale != null ? renderOpts.heightScale : 1;
      var forceSettle = renderOpts.forceSettle != null ? renderOpts.forceSettle : 1;
      var glintPhase = renderOpts.glintPhase != null ? renderOpts.glintPhase : -1;
      lastHeightScale = heightScale;
      lastForceSettle = forceSettle;
      var isDarkTheme = !!renderOpts.isDark;
      var isClassicUi = typeof document !== 'undefined' && document.documentElement.getAttribute('data-ui') === 'classic';
      var fallbackBg = isClassicUi
        ? (isDarkTheme ? '#191919' : '#f7f6f3')
        : (isDarkTheme ? '#0a0b0d' : '#f5f4f0');
      var bgArr = parseCssColorFast(renderOpts.bgColor || fallbackBg, [0.039, 0.043, 0.051, 1]);

      gl.depthMask(true);
      gl.disable(gl.CULL_FACE);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      // 1. Terrain Surface Mesh (opaque depth write)
      var activeAttrs = {};
      gl.useProgram(terrainProg);
      gl.uniformMatrix4fv(tLoc.u.uViewProj, false, cam.viewProj);
      gl.uniform1f(tLoc.u.uHeightAmp, heightAmp);
      gl.uniform1f(tLoc.u.uHeightScale, heightScale);
      gl.uniform1f(tLoc.u.uForceSettle, forceSettle);
      gl.uniform3f(tLoc.u.uEye, cam.eye[0], cam.eye[1], cam.eye[2]);
      gl.uniform3f(tLoc.u.uBg, bgArr[0], bgArr[1], bgArr[2]);
      gl.uniform1f(tLoc.u.uIsDark, isDarkTheme ? 1 : 0);
      gl.uniform1f(tLoc.u.uContours, params.contours || 16);
      gl.uniform1f(tLoc.u.uExtent, currentField.extent);
      gl.uniform1f(tLoc.u.uGlintPhase, glintPhase);
      bindAttrib(tLoc.a.aXZ, terrainXZBuf, 2, activeAttrs);
      bindAttrib(tLoc.a.aH, terrainHBuf, 1, activeAttrs);
      bindAttrib(tLoc.a.aGrad, terrainGradBuf, 2, activeAttrs);
      useAttrs(activeAttrs);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, terrainIdxBuf);
      gl.drawElements(gl.TRIANGLES, terrainIdxCount, gl.UNSIGNED_SHORT, 0);

      // 2. Round 3D Hub Spheres & Opaque Low-Degree Orbs (opaque depth-writing passes before translucent arcs)
      var camRight = [cam.view[0], cam.view[4], cam.view[8]];
      var camUp = [cam.view[1], cam.view[5], cam.view[9]];
      var lightDirView = [cam.view[4], cam.view[5], cam.view[6]];
      if (solidOpaqueVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(solidProg);
        gl.uniformMatrix4fv(sLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform3f(sLoc.u.uCamRight, camRight[0], camRight[1], camRight[2]);
        gl.uniform3f(sLoc.u.uCamUp, camUp[0], camUp[1], camUp[2]);
        gl.uniform3f(sLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(sLoc.u.uHeightScale, heightScale);
        gl.uniform1f(sLoc.u.uForceSettle, forceSettle);
        bindAttrib(sLoc.a.aCenter, solidPosBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aQuad, solidQuadBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aInstColor, solidInstColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, 0, solidOpaqueVertCount);
      }

      // 2b. Opaque Background Low-Degree Round Node Orbs (depth-written Lambert spheres matching 3D view)
      if (ptOpaqueCount > 0) {
        activeAttrs = {};
        gl.useProgram(pointProg);
        gl.uniformMatrix4fv(pLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(pLoc.u.uPointScale, height * dpr * 2.605089);
        gl.uniform3f(pLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(pLoc.u.uHeightScale, heightScale);
        gl.uniform1f(pLoc.u.uForceSettle, forceSettle);
        bindAttrib(pLoc.a.aPos, ptPosBuf, 3, activeAttrs);
        bindAttrib(pLoc.a.aSize, ptSizeBuf, 1, activeAttrs);
        bindAttrib(pLoc.a.aColor, ptColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.POINTS, 0, ptOpaqueCount);
      }

      // Translucent overlay layers share read-only depth test against terrain & solids
      gl.depthMask(false);

      // 3. Stems, Peak Beams & Background Edges
      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(lLoc.u.uViewProj, false, cam.viewProj);
      gl.uniform1f(lLoc.u.uExtent, currentField.extent);
      gl.uniform1f(lLoc.u.uHeightScale, heightScale);
      gl.uniform1f(lLoc.u.uForceSettle, forceSettle);
      gl.uniform1f(lLoc.u.uFadeStart, 0.86);
      if (lineVertCount > 0) {
        activeAttrs = {};
        bindAttrib(lLoc.a.aPos, linePosBuf, 3, activeAttrs);
        bindAttrib(lLoc.a.aColor, lineColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.LINES, 0, lineVertCount);
      }

      // 4. Dimmed Background Low-Degree Round Node Orbs (focus/filter mode)
      if (ptDimCount > 0) {
        activeAttrs = {};
        gl.useProgram(pointProg);
        gl.uniformMatrix4fv(pLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(pLoc.u.uPointScale, height * dpr * 2.605089);
        gl.uniform3f(pLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(pLoc.u.uHeightScale, heightScale);
        gl.uniform1f(pLoc.u.uForceSettle, forceSettle);
        bindAttrib(pLoc.a.aPos, ptPosBuf, 3, activeAttrs);
        bindAttrib(pLoc.a.aSize, ptSizeBuf, 1, activeAttrs);
        bindAttrib(pLoc.a.aColor, ptColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.POINTS, ptOpaqueCount, ptDimCount);
      }

      // 4b. Dimmed Background Hub Spheres (focus/filter mode)
      if (solidDimVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(solidProg);
        gl.uniformMatrix4fv(sLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform3f(sLoc.u.uCamRight, camRight[0], camRight[1], camRight[2]);
        gl.uniform3f(sLoc.u.uCamUp, camUp[0], camUp[1], camUp[2]);
        gl.uniform3f(sLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(sLoc.u.uHeightScale, heightScale);
        gl.uniform1f(sLoc.u.uForceSettle, forceSettle);
        bindAttrib(sLoc.a.aCenter, solidPosBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aQuad, solidQuadBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aInstColor, solidInstColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, solidOpaqueVertCount, solidDimVertCount);
      }

      // 5. Focused / Hovered Highlight Neighbor Arcs (3D glowing ribbons)
      if (hiLineVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(lineProg);
        gl.uniformMatrix4fv(lLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(lLoc.u.uExtent, currentField.extent);
        gl.uniform1f(lLoc.u.uHeightScale, heightScale);
        gl.uniform1f(lLoc.u.uForceSettle, forceSettle);
        gl.uniform1f(lLoc.u.uFadeStart, 8.0);
        bindAttrib(lLoc.a.aPos, hiLinePosBuf, 3, activeAttrs);
        bindAttrib(lLoc.a.aColor, hiLineColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, 0, hiLineVertCount);
      }

      gl.depthMask(true);

      // 6. Crisp HiDPI Billboard Labels & Community Capsules
      drawLabelsOverlay(cam.viewProj, {
        isDark: isDarkTheme,
        isLite: renderOpts.isLite,
        showLabels: renderOpts.showLabels,
        getLabelText: renderOpts.getLabelText,
        distance: camState.distance,
        heightAmp: heightAmp,
        heightScale: heightScale,
        forceSettle: forceSettle,
        labelFade: renderOpts.labelFade,
        hasHoverHighlight: !!renderOpts.hasHoverHighlight,
        communityLabels: renderOpts.communityLabels,
        routeBadges: renderOpts.routeBadges,
      });
    }

    function pickNodeAtScreen(localX, localY, extraSlop) {
      if (!lastViewProj || !cachedNodes3d.length) return null;
      var vp = lastViewProj;
      var m0 = vp[0], m1 = vp[1], m2 = vp[2], m3 = vp[3];
      var m4 = vp[4], m5 = vp[5], m6 = vp[6], m7 = vp[7];
      var m8 = vp[8], m9 = vp[9], m10 = vp[10], m11 = vp[11];
      var m12 = vp[12], m13 = vp[13], m14 = vp[14], m15 = vp[15];
      var slop = extraSlop > 0 ? extraSlop : 0;
      var settled = lastForceSettle >= 0.999;
      var settleEase = settled ? 1 : (1 - Math.pow(1 - Math.max(0, lastForceSettle), 3));
      var radialAmp = 0.026 * (1 - settleEase);
      var swirlAng = (1 - settleEase) * 0.22;
      var swirlSin = settled ? 0 : Math.sin(swirlAng);
      var swirlCos = settled ? 1 : Math.cos(swirlAng);
      var hScale = lastHeightScale;
      var halfW = width * 0.5;
      var halfH = height * 0.5;
      var best = null;
      var bestScore = Infinity;
      for (var i = 0; i < cachedNodes3d.length; i++) {
        var n3 = cachedNodes3d[i];
        if (!n3.visible && !n3.focused) continue;
        var wx = n3.wx;
        var wz = n3.wz;
        if (!settled) {
          var rad = Math.hypot(wx, wz);
          if (rad > 1e-4) {
            var sRad = 1 + (radialAmp * (1 / (0.35 + rad))) / rad;
            var rx = wx * sRad;
            var rz = wz * sRad;
            var sw = swirlSin * Math.min(1, rad);
            var cwRot = Math.cos(sw);
            var swRot = Math.sin(sw);
            wx = rx * cwRot - rz * swRot;
            wz = rx * swRot + rz * cwRot;
          } else if (swirlCos !== 1) {
            wx = wx * swirlCos - wz * swirlSin;
            wz = wx * swirlSin + wz * swirlCos;
          }
        }
        var wy = n3.wy * hScale;
        var cw = m3 * wx + m7 * wy + m11 * wz + m15;
        if (cw <= 0.001) continue;
        var invW = 1 / cw;
        var sx = ((m0 * wx + m4 * wy + m8 * wz + m12) * invW + 1) * halfW;
        var sy = (1 - (m1 * wx + m5 * wy + m9 * wz + m13) * invW) * halfH;
        var ndcZ = (m2 * wx + m6 * wy + m10 * wz + m14) * invW;
        if (ndcZ < -1 || ndcZ > 1) continue;
        var projR = n3.baseRadius ? (n3.baseRadius * height * 0.75) : (n3.degNorm > 0.35 ? 16 : 11);
        var hitRadius = Math.max(9, Math.min(28, projR + 5)) + (n3.focused ? 3 : 0) + slop;
        var distPx = Math.hypot(sx - localX, sy - localY);
        if (distPx <= hitRadius) {
          var score = distPx + ndcZ * 2;
          if (score < bestScore) {
            bestScore = score;
            best = n3.node;
          }
        }
      }
      return best;
    }

    function destroy() {
      if (gl.isContextLost && gl.isContextLost()) return;
      [terrainXZBuf, terrainHBuf, terrainGradBuf, terrainIdxBuf, linePosBuf, lineColBuf,
        hiLinePosBuf, hiLineColBuf, solidPosBuf, solidQuadBuf,
        solidInstColBuf, ptPosBuf, ptSizeBuf, ptColBuf].forEach(function (b) {
        if (b) gl.deleteBuffer(b);
      });
      [terrainProg, lineProg, solidProg, pointProg].forEach(function (p) {
        if (p) gl.deleteProgram(p);
      });
    }

    return {
      isWebGL2: isWebGL2,
      resize: resize,
      uploadTerrainMesh: uploadTerrainMesh,
      buildDynamicBatches: buildDynamicBatches,
      render: render,
      pickNodeAtScreen: pickNodeAtScreen,
      getNodeWorldPos: function (nodeId) {
        for (var i = 0; i < cachedNodes3d.length; i++) {
          var n3 = cachedNodes3d[i];
          if (n3.node && n3.node.id === nodeId) {
            return { wx: n3.wx, wy: n3.wy * lastHeightScale, wz: n3.wz };
          }
        }
        return null;
      },
      unprojectToPlaneY: function (localX, localY, planeY) {
        return unprojectScreenToPlaneY(localX, localY, planeY, lastCam, width, height);
      },
      getTopScreenTargets: function (limit) {
        if (!lastViewProj) return [];
        var maxCount = limit || 12;
        var out = [];
        for (var i = 0; i < cachedPolyNodes.length && out.length < maxCount; i++) {
          var n3 = cachedPolyNodes[i];
          if (!n3.visible && !n3.focused) continue;
          var mXZ = evalForceMotionXZ(n3.wx, n3.wz, lastForceSettle, 0.026);
          var pt = projectWorldToScreen(mXZ[0], n3.wy * lastHeightScale, mXZ[1], lastViewProj, width, height);
          if (!pt || pt[2] < -1 || pt[2] > 1) continue;
          if (pt[0] < 36 || pt[0] > width - 36 || pt[1] < 36 || pt[1] > height - 36) continue;
          out.push({
            id: n3.node.id,
            label: n3.node.label || (n3.node._info && n3.node._info.title) || n3.node.id,
            deg: n3.deg,
            x: Number(pt[0].toFixed(1)),
            y: Number(pt[1].toFixed(1)),
            wy: Number(n3.wy.toFixed(4)),
            surfaceH: Number((n3.wySurf || 0).toFixed(4)),
            depth: Number(pt[2].toFixed(4)),
          });
        }
        return out;
      },
      getFieldStats: function () {
        return currentField ? {
          gridSize: currentField.gridSize,
          maxDegree: currentField.maxDegree,
          nodeCount: cachedNodes3d.length,
          polyCount: cachedPolyNodes.length,
          hiEdgeCount: hiEdgeCount,
          communityLabelCount: lastCommunityLabelCount,
          routeBadgeCount: lastRouteBadgeCount,
          forceSettle: Number(lastForceSettle.toFixed(3)),
        } : null;
      },
      destroy: destroy,
    };
  }

  /* ── 参数持久化与统一控制器 ── */

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
    var getViewportSize = opts.getViewportSize || function () {
      return {
        width: (stageEl && stageEl.clientWidth) || (wrapEl && wrapEl.clientWidth) || 1200,
        height: (stageEl && stageEl.clientHeight) || (wrapEl && wrapEl.clientHeight) || 800,
      };
    };
    var getGraphData = opts.getGraphData || function () { return { nodes: [], edges: [] }; };
    var isDark = opts.isDark || function () {
      return typeof document === 'undefined' || document.documentElement.getAttribute('data-theme') !== 'light';
    };
    var getBgColor = opts.getBgColor || function () {
      var isClassicUi = typeof document !== 'undefined' && document.documentElement.getAttribute('data-ui') === 'classic';
      if (isClassicUi) return isDark() ? '#191919' : '#f7f6f3';
      return isDark() ? '#0a0b0d' : '#f5f4f0';
    };
    var getLabelText = opts.getLabelText || function (d) { return (d && (d._info && d._info.title || d.label)) || ''; };
    var getCommunityLabels = opts.getCommunityLabels || function () { return []; };
    var getRouteBadges = opts.getRouteBadges || function () { return []; };
    var isLabelsVisible = opts.isLabelsVisible || function () { return true; };
    var isEdgesVisible = opts.isEdgesVisible || function () { return false; };
    var onNodeHover = opts.onNodeHover || function () {};
    var onNodeClick = opts.onNodeClick || function () {};
    var onBackgroundClick = opts.onBackgroundClick || function () {};
    var onNodeDragStart = opts.onNodeDragStart || function () {};
    var onNodeDrag = opts.onNodeDrag || function () {};
    var onNodeDragEnd = opts.onNodeDragEnd || function () {};
    var allowNodeDrag = opts.allowNodeDrag !== undefined ? opts.allowNodeDrag : true;
    var onStateChange = opts.onStateChange || function () {};
    var onUnsupported = opts.onUnsupported || function () {};

    var persist = opts.persist !== false;
    var allowTouchOrbit = opts.allowTouchOrbit !== false;
    var defaultOverrides = (opts.defaultParamsByTier && typeof opts.defaultParamsByTier === 'object')
      ? opts.defaultParamsByTier
      : null;

    function resolveDefaultForTier(t) {
      var base = getDefaultsForTier(t);
      if (opts.usePreviewTerrainHeight && PREVIEW_HEIGHT_BY_TIER[t] != null) {
        base.height = PREVIEW_HEIGHT_BY_TIER[t];
      }
      if (defaultOverrides && defaultOverrides[t] && typeof defaultOverrides[t] === 'object') {
        return clampParams(Object.assign({}, base, defaultOverrides[t]), t);
      }
      return base;
    }

    function saveParamsIfEnabled(nextParams) {
      if (persist) saveStoredParams(nextParams);
    }

    var env = detectEnvironment(opts.nodeCount || 0);
    var tier = resolveCapabilityTier(env);
    var enabled = false;
    var draggingSlider = false;
    var params = persist ? loadStoredParams(tier) : resolveDefaultForTier(tier);
    var camState = {
      distance: opts.defaultDistance || 1.72,
      targetX: 0,
      targetY: 0.08,
      targetZ: 0,
    };

    var renderer = null;
    var glCanvas = null;
    var labelCanvas = null;
    var degreeInfo = null;
    var terrainField = null;
    var activeDragBounds = null;
    var hoveredNode = null;
    var routeFocusNodeId = null;
    var batchesDirty = true;
    var terrainDirty = true;
    var renderRaf = 0;

    function ensureStageCanvases() {
      if (!stageEl || typeof document === 'undefined') return null;
      if (!glCanvas) {
        glCanvas = stageEl.querySelector('.graph-terrain-gl');
        if (!glCanvas) {
          glCanvas = document.createElement('canvas');
          glCanvas.className = 'graph-terrain-gl';
          stageEl.appendChild(glCanvas);
        }
      }
      if (!labelCanvas) {
        labelCanvas = stageEl.querySelector('.graph-terrain-labels');
        if (!labelCanvas) {
          labelCanvas = document.createElement('canvas');
          labelCanvas.className = 'graph-terrain-labels';
          labelCanvas.setAttribute('aria-hidden', 'true');
          stageEl.appendChild(labelCanvas);
        }
      }
      if (!renderer && tier !== 'unsupported') {
        try {
          renderer = createWebGLTerrainRenderer(glCanvas, labelCanvas, {
            dprCap: tier === 'lite' ? 1.5 : 2,
          });
        } catch (_e) {
          renderer = null;
        }
        if (!renderer) {
          tier = 'unsupported';
          return null;
        }
        glCanvas.addEventListener('webglcontextlost', function (ev) {
          ev.preventDefault();
          setEnabled(false);
          onUnsupported('WebGL 上下文丢失，已切换回 2D 平面视图。');
        });
      }
      return renderer;
    }

    var liveSimSync = false;

    function rebuildFieldAndBatches() {
      if (!renderer || !enabled) return;
      var data = getGraphData() || {};
      var nodes = data.nodes || [];
      var edges = data.edges || [];
      var activeDragId = (dragState && dragState.mode === 'node' && dragState.dragStarted && dragState.node)
        ? dragState.node.id
        : null;
      if (terrainDirty || !terrainField) {
        var isDragOrRebound = !!activeDragId || !!reboundAnim;
        if (!degreeInfo || !isDragOrRebound) {
          degreeInfo = computeNodeDegrees(nodes, edges);
        }
        var gridN = tier === 'lite' ? 92 : (nodes.length <= 160 ? 120 : 148);
        var isLiveMotion = liveSimSync || isDragOrRebound;
        // 实时力模拟 / 拖拽 / 回弹帧保持阻尼连贯运动，大图拖拽与回弹阶段延后到落稳帧再统一执行完整 Jacobi 防重叠烘焙
        terrainField = computeTerrainField(nodes, degreeInfo, {
          gridSize: gridN,
          extent: (activeDragBounds && terrainField) ? terrainField.extent : undefined,
          activeIds: data.activeIds || null,
          fixedBounds: activeDragBounds,
          skipOverlapResolve: liveSimSync || (isLiveMotion && nodes.length > 160),
          overlapPasses: isLiveMotion ? 1 : (tier === 'lite' ? 4 : 6),
          pinnedId: activeDragId,
        });
        renderer.uploadTerrainMesh(terrainField);
        terrainDirty = false;
        batchesDirty = true;
      }
      if (batchesDirty) {
        var heightAmp = (params.height / 100) * 0.46;
        renderer.buildDynamicBatches({
          nodes: nodes,
          edges: edges,
          activeIds: data.activeIds || null,
          focusIds: data.focusIds || null,
          hoverNode: hoveredNode,
          sidebarNode: data.sidebarNode || null,
          routeFocusNodeId: routeFocusNodeId,
          draggingNodeId: activeDragId,
          getNodeColor: data.getNodeColor,
          getNodeAlpha: data.getNodeAlpha,
          showEdges: isEdgesVisible(),
          isLite: tier === 'lite',
        }, heightAmp, isDark());
        batchesDirty = false;
      }
    }

    var entryAnim = null;
    var reboundAnim = null;

    function nowMs() {
      return (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now();
    }

    function cancelEntryAnim() {
      entryAnim = null;
    }

    function cancelReboundAnim(finalize) {
      if (!reboundAnim) return;
      var rb = reboundAnim;
      reboundAnim = null;
      if (finalize && rb.node) {
        activeDragBounds = null;
        onNodeDragEnd(rb.node, null);
      }
    }

    function collectDragNeighbors(targetNode) {
      if (!targetNode || !targetNode.id) return [];
      var data = getGraphData() || {};
      var nodes = data.nodes || [];
      var edges = data.edges || [];
      var nbIds = Object.create(null);
      var i;
      for (i = 0; i < edges.length; i++) {
        var e = edges[i];
        if (!e) continue;
        var s = edgeEndId(e.source);
        var t = edgeEndId(e.target);
        if (s === targetNode.id && t && t !== targetNode.id) nbIds[t] = true;
        else if (t === targetNode.id && s && s !== targetNode.id) nbIds[s] = true;
      }
      var list = [];
      for (i = 0; i < nodes.length && list.length < 120; i++) {
        var nd = nodes[i];
        if (!nd || !nd.id || !nbIds[nd.id] || nd.x == null || nd.y == null) continue;
        list.push({ node: nd, x0: nd.x, y0: nd.y, vx: 0, vy: 0 });
      }
      return list;
    }

    function stepReboundAnim() {
      if (!reboundAnim) return false;
      var nowR = nowMs();
      var elapsedFrame = Math.max(0, Math.min(600, nowR - reboundAnim.lastTime));
      reboundAnim.lastTime = nowR;
      var elapsedR = nowR - reboundAnim.startTime;
      var totalTicks = Math.max(0.5, elapsedFrame / 16.667);
      var subSteps = Math.max(1, Math.ceil(totalTicks));
      var subDt = totalTicks / subSteps; // 恒落在 [0.5, 1.0]，数值稳定且跨帧率严格同步墙钟
      var rNode = reboundAnim.node;
      // 对齐 2D/3D 立体力模拟的粘滞过阻尼松弛：固定子步平滑回落到力平衡位置，无过冲弹跳、低帧率不突跳
      var kSpring = REBOUND_PARAMS.kSpring;
      var damp = Math.pow(1 - REBOUND_PARAMS.velocityDecay, subDt);
      var nbResidual = REBOUND_PARAMS.nbResidual;
      var nbSpring = REBOUND_PARAMS.nbSpring;
      var nbDamp = Math.pow(1 - REBOUND_PARAMS.nbVelocityDecay, subDt);
      for (var s = 0; s < subSteps; s++) {
        var fx = (reboundAnim.eqX - rNode.x) * kSpring;
        var fy = (reboundAnim.eqY - rNode.y) * kSpring;
        reboundAnim.vx = (reboundAnim.vx + fx * subDt) * damp;
        reboundAnim.vy = (reboundAnim.vy + fy * subDt) * damp;
        rNode.x += reboundAnim.vx * subDt;
        rNode.y += reboundAnim.vy * subDt;
        var shiftX = rNode.x - reboundAnim.origX;
        var shiftY = rNode.y - reboundAnim.origY;
        for (var ri = 0; ri < reboundAnim.neighbors.length; ri++) {
          var rnb = reboundAnim.neighbors[ri];
          var tNx = rnb.x0 + shiftX * nbResidual;
          var tNy = rnb.y0 + shiftY * nbResidual;
          rnb.vx = (rnb.vx + (tNx - rnb.node.x) * nbSpring * subDt) * nbDamp;
          rnb.vy = (rnb.vy + (tNy - rnb.node.y) * nbSpring * subDt) * nbDamp;
          rnb.node.x += rnb.vx * subDt;
          rnb.node.y += rnb.vy * subDt;
        }
      }
      onNodeDrag(rNode, rNode.x, rNode.y, null);
      var curShiftX = rNode.x - reboundAnim.origX;
      var curShiftY = rNode.y - reboundAnim.origY;
      var maxNbEnergy = 0;
      for (var ci = 0; ci < reboundAnim.neighbors.length; ci++) {
        var cnb = reboundAnim.neighbors[ci];
        onNodeDrag(cnb.node, cnb.node.x, cnb.node.y, null);
        var cTx = cnb.x0 + curShiftX * nbResidual;
        var cTy = cnb.y0 + curShiftY * nbResidual;
        var nbE = Math.hypot(cTx - cnb.node.x, cTy - cnb.node.y) + Math.hypot(cnb.vx, cnb.vy);
        if (nbE > maxNbEnergy) maxNbEnergy = nbE;
      }

      terrainDirty = true;
      batchesDirty = true;

      var distEq = Math.hypot(reboundAnim.eqX - rNode.x, reboundAnim.eqY - rNode.y);
      var speedEq = Math.hypot(reboundAnim.vx, reboundAnim.vy);
      var stopDist = Math.max(0.35, (reboundAnim.dropDist || 0) * 0.008);
      var stopSpeed = Math.max(0.25, stopDist * 0.5);
      var stopNb = Math.max(0.45, stopDist * 1.1);
      if ((distEq < stopDist && speedEq < stopSpeed && maxNbEnergy < stopNb) || elapsedR >= reboundAnim.maxDuration) {
        rNode.x = reboundAnim.eqX;
        rNode.y = reboundAnim.eqY;
        onNodeDrag(rNode, rNode.x, rNode.y, null);
        var finalShiftX = rNode.x - reboundAnim.origX;
        var finalShiftY = rNode.y - reboundAnim.origY;
        for (var fi = 0; fi < reboundAnim.neighbors.length; fi++) {
          var fnb = reboundAnim.neighbors[fi];
          fnb.node.x = fnb.x0 + finalShiftX * nbResidual;
          fnb.node.y = fnb.y0 + finalShiftY * nbResidual;
          onNodeDrag(fnb.node, fnb.node.x, fnb.node.y, null);
        }
        var doneNode = rNode;
        reboundAnim = null;
        activeDragBounds = null;
        onNodeDragEnd(doneNode, null);
        return false;
      }
      return true;
    }

    function startCameraTransition(fromPitch, fromYaw, fromDist, fromHeightScale, durationMs, sweepGlint, forceMotion, fromTargetX, fromTargetZ) {
      if (tier === 'reduced' || typeof requestAnimationFrame !== 'function' || !durationMs || durationMs <= 0) {
        entryAnim = null;
        renderNow();
        return;
      }
      entryAnim = {
        startTime: null,
        duration: durationMs,
        fromPitch: fromPitch,
        fromYaw: fromYaw,
        fromDist: fromDist,
        fromTargetX: fromTargetX != null ? fromTargetX : camState.targetX,
        fromTargetZ: fromTargetZ != null ? fromTargetZ : camState.targetZ,
        fromHeightScale: fromHeightScale,
        sweepGlint: !!sweepGlint,
        forceMotion: !!forceMotion,
      };
      if (renderRaf && typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(renderRaf);
        renderRaf = 0;
      }
      renderNow();
    }

    function renderNow() {
      renderRaf = 0;
      if (!enabled || tier === 'unsupported') return;
      var r = ensureStageCanvases();
      if (!r) return;
      var vp = getViewportSize();
      r.resize(vp.width || 1200, vp.height || 800, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1);
      var keepGoing = false;
      if (reboundAnim && stepReboundAnim()) {
        keepGoing = true;
      }
      rebuildFieldAndBatches();
      var renderParams = params;
      var renderCam = camState;
      var heightScale = 1;
      var forceSettle = 1;
      var glintPhase = -1;
      var labelFade = 1;
      if (entryAnim) {
        if (entryAnim.startTime == null) entryAnim.startTime = nowMs();
        var elapsed = Math.max(0, nowMs() - entryAnim.startTime);
        var rawT = Math.min(1, elapsed / entryAnim.duration);
        var camEase = 1 - Math.pow(1 - rawT, 3.2);
        var riseEase;
        if (entryAnim.forceMotion) {
          var uRise = Math.min(1, rawT * 1.14);
          var riseSpring = 1 - Math.exp(-5.0 * uRise) * Math.cos(4.8 * uRise);
          var riseBlend = smoothstep(0.84, 1.0, uRise);
          riseEase = riseSpring + (1 - riseSpring) * riseBlend;
          forceSettle = rawT;
        } else {
          riseEase = 1 - Math.pow(1 - Math.min(1, rawT * 1.18), 2.5);
          forceSettle = 1;
        }
        renderParams = {
          height: params.height,
          contours: params.contours,
          pitch: entryAnim.fromPitch + (params.pitch - entryAnim.fromPitch) * camEase,
          yaw: entryAnim.fromYaw + (params.yaw - entryAnim.fromYaw) * camEase,
        };
        renderCam = {
          distance: entryAnim.fromDist + (camState.distance - entryAnim.fromDist) * camEase,
          targetX: entryAnim.fromTargetX + (camState.targetX - entryAnim.fromTargetX) * camEase,
          targetY: camState.targetY,
          targetZ: entryAnim.fromTargetZ + (camState.targetZ - entryAnim.fromTargetZ) * camEase,
        };
        heightScale = entryAnim.fromHeightScale + (1 - entryAnim.fromHeightScale) * riseEase;
        glintPhase = entryAnim.sweepGlint ? (rawT * 1.32 - 0.12) : -1;
        labelFade = (entryAnim.fromHeightScale < 0.5 || entryAnim.forceMotion)
          ? Math.max(0, Math.min(1, (rawT - 0.30) / 0.50))
          : 1;
        if (rawT < 1) {
          keepGoing = true;
        } else {
          entryAnim = null;
          heightScale = 1;
          forceSettle = 1;
          glintPhase = -1;
          labelFade = 1;
          renderParams = params;
          renderCam = camState;
        }
      }
      if (stageEl && stageEl.style && typeof stageEl.style.setProperty === 'function') {
        var pitchForGlow = renderParams.pitch != null ? renderParams.pitch : 36;
        var glowVis = (1 - smoothstep(40, 62, pitchForGlow)) * smoothstep(0.25, 0.85, heightScale);
        var glowStr = glowVis.toFixed(3);
        if (glowStr !== lastGlowVal) {
          lastGlowVal = glowStr;
          stageEl.style.setProperty('--persp-horizon-glow', glowStr);
        }
      }
      r.render(renderParams, renderCam, {
        isDark: isDark(),
        bgColor: getBgColor(),
        isLite: tier === 'lite',
        showLabels: isLabelsVisible(),
        getLabelText: getLabelText,
        heightScale: heightScale,
        forceSettle: forceSettle,
        glintPhase: glintPhase,
        labelFade: labelFade,
        hasHoverHighlight: !!hoveredNode,
        communityLabels: getCommunityLabels(),
        routeBadges: getRouteBadges(),
      });
      if (keepGoing && typeof requestAnimationFrame === 'function') {
        renderRaf = requestAnimationFrame(renderNow);
      }
    }

    function scheduleRender() {
      if (!enabled || renderRaf) return;
      if (typeof requestAnimationFrame === 'function') {
        renderRaf = requestAnimationFrame(renderNow);
      } else {
        renderNow();
      }
    }

    function clearStageClasses() {
      if (wrapEl) {
        wrapEl.classList.remove('is-perspective-mode', 'is-perspective-lite', 'is-perspective-reduced');
      }
      if (stageEl) stageEl.hidden = true;
    }

    function syncHostClasses() {
      if (!wrapEl) return;
      wrapEl.classList.toggle('is-perspective-mode', enabled);
      wrapEl.classList.toggle('is-perspective-lite', enabled && tier === 'lite');
      wrapEl.classList.toggle('is-perspective-reduced', enabled && tier === 'reduced');
      if (stageEl) stageEl.hidden = !enabled;
    }

    /* ── 鼠标 / 触控交互（轨道旋转、平移、双指捏合缩放/平移、节点拖拽与点按）── */
    var dragState = null;
    var activePointers = new Map();
    var pinchState = null;
    var gestureSuppressedTap = false;
    var cachedStageRect = null;
    var lastGlowVal = null;
    var hoverRaf = 0;
    var pendingHoverEv = null;
    var saveParamsTimer = 0;
    var pendingSaveParams = null;

    function invalidateStageRect() {
      cachedStageRect = null;
    }

    function cancelPendingHover() {
      pendingHoverEv = null;
      if (hoverRaf && typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(hoverRaf);
        hoverRaf = 0;
      }
    }

    function scheduleSaveParams(nextParams) {
      if (!persist) return;
      pendingSaveParams = nextParams;
      if (saveParamsTimer) return;
      saveParamsTimer = setTimeout(function () {
        saveParamsTimer = 0;
        if (pendingSaveParams) {
          var p = pendingSaveParams;
          pendingSaveParams = null;
          saveParamsIfEnabled(p);
        }
      }, 250);
    }

    function flushSaveParams() {
      if (saveParamsTimer) {
        clearTimeout(saveParamsTimer);
        saveParamsTimer = 0;
      }
      if (pendingSaveParams) {
        var p = pendingSaveParams;
        pendingSaveParams = null;
        saveParamsIfEnabled(p);
      }
    }

    function getStageLocalPoint(ev) {
      if (!stageEl) return [0, 0];
      if (!cachedStageRect) cachedStageRect = stageEl.getBoundingClientRect();
      return [ev.clientX - cachedStageRect.left, ev.clientY - cachedStageRect.top];
    }

    function performHoverPick(ev) {
      if (!enabled || !renderer || !ev) return;
      if (dragState && dragState.id != null) return;
      var pt = getStageLocalPoint(ev);
      var hit = renderer.pickNodeAtScreen(pt[0], pt[1]);
      var prevId = hoveredNode ? hoveredNode.id : null;
      var nextId = hit ? hit.id : null;
      if (prevId !== nextId) {
        hoveredNode = hit || null;
        if (stageEl) stageEl.style.cursor = (hit && canDragNodesNow()) ? 'grab' : '';
        batchesDirty = true;
        scheduleRender();
      }
      if (hit || prevId !== null) onNodeHover(hit || null, ev);
    }

    function canDragNodesNow() {
      return typeof allowNodeDrag === 'function' ? !!allowNodeDrag() : !!allowNodeDrag;
    }

    function getFirstTwoPointers() {
      var list = [];
      activePointers.forEach(function (v) {
        if (list.length < 2) list.push(v);
      });
      return list;
    }

    function panCameraByDelta(dx, dy) {
      var yawRad = (params.yaw * Math.PI) / 180;
      var speed = camState.distance * 0.0022;
      var rx = Math.cos(yawRad);
      var rz = -Math.sin(yawRad);
      var fx = Math.sin(yawRad);
      var fz = Math.cos(yawRad);
      var panMax = Math.max(1.8, (terrainField ? terrainField.extent : 1.45) * 1.25);
      camState.targetX = Math.max(-panMax, Math.min(panMax, camState.targetX - dx * rx * speed + dy * fx * speed));
      camState.targetZ = Math.max(-panMax, Math.min(panMax, camState.targetZ - dx * rz * speed + dy * fz * speed));
    }

    function onPointerDown(ev) {
      if (!enabled || !stageEl) return;
      cancelPendingHover();
      cachedStageRect = null;
      if (entryAnim) cancelEntryAnim();
      var isTouch = ev.pointerType === 'touch';
      // W3C Pointer Events: isPrimary 标志新一轮手势的首指，清理可能残留的旧指针避免误入双指捏合
      if (ev.isPrimary) {
        activePointers.clear();
        pinchState = null;
        gestureSuppressedTap = false;
      }
      if (ev.pointerId != null) {
        activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY, isTouch: isTouch });
      }
      if (isTouch && !allowTouchOrbit) {
        dragState = {
          id: ev.pointerId,
          startX: ev.clientX,
          startY: ev.clientY,
          lastX: ev.clientX,
          lastY: ev.clientY,
          moved: false,
          isTouch: true,
          mode: 'tap-only',
        };
        return;
      }
      // 双指触控：立即切入捏合缩放 + 双指平移，取消单指拖拽，抬指不触发误点
      if (activePointers.size >= 2) {
        if (dragState && dragState.mode === 'node' && dragState.dragStarted) {
          onNodeDragEnd(dragState.node, ev);
          activeDragBounds = null;
          terrainDirty = true;
          batchesDirty = true;
        }
        dragState = null;
        gestureSuppressedTap = true;
        var pair = getFirstTwoPointers();
        pinchState = {
          dist: Math.max(12, Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y)),
          midX: (pair[0].x + pair[1].x) * 0.5,
          midY: (pair[0].y + pair[1].y) * 0.5,
        };
        if (glCanvas && typeof glCanvas.setPointerCapture === 'function' && ev.pointerId != null) {
          try { glCanvas.setPointerCapture(ev.pointerId); } catch (_e) { /* ignore */ }
        }
        return;
      }
      gestureSuppressedTap = false;
      var isPan = ev.button === 2 || ev.button === 1 || ev.shiftKey;
      if (!isPan && (ev.button === 0 || isTouch) && canDragNodesNow() && renderer && terrainField) {
        var pt = getStageLocalPoint(ev);
        var hit = renderer.pickNodeAtScreen(pt[0], pt[1], isTouch ? 14 : 0);
        if (hit) {
          var nPos = renderer.getNodeWorldPos(hit.id);
          if (nPos) {
            var grabPt = renderer.unprojectToPlaneY(pt[0], pt[1], nPos.wy);
            dragState = {
              id: ev.pointerId,
              startX: ev.clientX,
              startY: ev.clientY,
              lastX: ev.clientX,
              lastY: ev.clientY,
              moved: false,
              isTouch: isTouch,
              mode: 'node',
              node: hit,
              planeY: nPos.wy,
              origWx: nPos.wx,
              origWz: nPos.wz,
              offsetWx: grabPt ? (nPos.wx - grabPt[0]) : 0,
              offsetWz: grabPt ? (nPos.wz - grabPt[2]) : 0,
              dragStarted: false,
            };
            if (glCanvas && typeof glCanvas.setPointerCapture === 'function' && ev.pointerId != null) {
              try { glCanvas.setPointerCapture(ev.pointerId); } catch (_e) { /* ignore */ }
            }
            return;
          }
        }
      }
      dragState = {
        id: ev.pointerId,
        startX: ev.clientX,
        startY: ev.clientY,
        lastX: ev.clientX,
        lastY: ev.clientY,
        moved: false,
        isTouch: isTouch,
        mode: isPan ? 'pan' : 'orbit',
      };
      if (glCanvas && typeof glCanvas.setPointerCapture === 'function' && ev.pointerId != null) {
        try { glCanvas.setPointerCapture(ev.pointerId); } catch (_e) { /* ignore */ }
      }
    }

    function onPointerMove(ev) {
      if (!enabled || !renderer) return;
      if (ev.pointerId != null && activePointers.has(ev.pointerId)) {
        var entry = activePointers.get(ev.pointerId);
        entry.x = ev.clientX;
        entry.y = ev.clientY;
      }
      if (pinchState && activePointers.size >= 2) {
        var pair = getFirstTwoPointers();
        var nextDist = Math.max(12, Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y));
        var nextMidX = (pair[0].x + pair[1].x) * 0.5;
        var nextMidY = (pair[0].y + pair[1].y) * 0.5;
        var zoomRatio = pinchState.dist / nextDist;
        camState.distance = Math.max(0.65, Math.min(5.5, camState.distance * zoomRatio));
        panCameraByDelta(nextMidX - pinchState.midX, nextMidY - pinchState.midY);
        pinchState.dist = nextDist;
        pinchState.midX = nextMidX;
        pinchState.midY = nextMidY;
        scheduleRender();
        return;
      }
      if (dragState && dragState.id === ev.pointerId) {
        var dx = ev.clientX - dragState.lastX;
        var dy = ev.clientY - dragState.lastY;
        dragState.lastX = ev.clientX;
        dragState.lastY = ev.clientY;
        var moveThresh = dragState.isTouch ? 8 : 4;
        if (Math.hypot(ev.clientX - dragState.startX, ev.clientY - dragState.startY) > moveThresh) {
          dragState.moved = true;
        }
        if (dragState.mode === 'node') {
          if (dragState.moved) {
            if (!dragState.dragStarted) {
              cancelReboundAnim(false);
              dragState.dragStarted = true;
              dragState.origX = dragState.node.x;
              dragState.origY = dragState.node.y;
              dragState.neighbors = collectDragNeighbors(dragState.node);
              activeDragBounds = (terrainField && terrainField.center)
                ? Object.assign({}, terrainField.center)
                : null;
              hoveredNode = dragState.node;
              if (stageEl) stageEl.style.cursor = 'grabbing';
              onNodeHover(null, ev);
              onNodeDragStart(dragState.node, ev);
            }
            var dragPt = getStageLocalPoint(ev);
            var hitWorld = renderer.unprojectToPlaneY(dragPt[0], dragPt[1], dragState.planeY);
            if (hitWorld && activeDragBounds) {
              // 仅做超大径向保护以防掠射角射线发散，不再用正交方框限制拖拽范围
              var rawWx = hitWorld[0] + dragState.offsetWx;
              var rawWz = hitWorld[2] + dragState.offsetWz;
              var maxDragR = Math.max(4.5, (terrainField ? terrainField.extent : 1.45) * 2.6);
              var rawR = Math.hypot(rawWx, rawWz);
              var dragScale = rawR > maxDragR ? (maxDragR / rawR) : 1;
              var nextWx = rawWx * dragScale;
              var nextWz = rawWz * dragScale;
              var nextCoords = worldToGraphCoords(nextWx, nextWz, activeDragBounds);
              dragState.node.x = nextCoords.x;
              dragState.node.y = nextCoords.y;
              onNodeDrag(dragState.node, nextCoords.x, nextCoords.y, ev);
              // Elastic web stretch: 按抓取点真实世界位移计算邻居牵引量，避免首帧因重叠微调跳变
              var csDrag = activeDragBounds.coreSpan || 0.78;
              var wScale = activeDragBounds.halfSpan / csDrag;
              var moveDx = (nextWx - dragState.origWx) * wScale;
              var moveDy = (nextWz - dragState.origWz) * wScale;
              var nbs = dragState.neighbors || [];
              var pullRatio = Math.min(0.24, 0.08 + 0.16 / Math.sqrt(1 + nbs.length / 8));
              for (var ni = 0; ni < nbs.length; ni++) {
                var nb = nbs[ni];
                nb.node.x = nb.x0 + moveDx * pullRatio;
                nb.node.y = nb.y0 + moveDy * pullRatio;
                onNodeDrag(nb.node, nb.node.x, nb.node.y, ev);
              }
              terrainDirty = true;
              batchesDirty = true;
              scheduleRender();
            }
          }
          return;
        }
        if (dragState.mode === 'orbit' && dragState.moved) {
          if (hoveredNode) {
            hoveredNode = null;
            if (stageEl) stageEl.style.cursor = '';
            batchesDirty = true;
            onNodeHover(null, ev);
          }
          var nextYaw = ((params.yaw - dx * 0.38 + 540) % 360) - 180;
          var nextPitch = Math.max(BOUNDS.pitch.min, Math.min(BOUNDS.pitch.max, params.pitch + dy * 0.28));
          params = clampParams({
            height: params.height,
            contours: params.contours,
            pitch: nextPitch,
            yaw: nextYaw,
          }, tier);
          scheduleSaveParams(params);
          onStateChange(getPublicState());
          scheduleRender();
          return;
        }
        if (dragState.mode === 'pan' && dragState.moved) {
          if (hoveredNode) {
            hoveredNode = null;
            if (stageEl) stageEl.style.cursor = '';
            batchesDirty = true;
            onNodeHover(null, ev);
          }
          panCameraByDelta(dx, dy);
          scheduleRender();
          return;
        }
      }
      if (ev.pointerType === 'touch') return;
      if (typeof requestAnimationFrame === 'function') {
        pendingHoverEv = {
          clientX: ev.clientX,
          clientY: ev.clientY,
          pointerType: ev.pointerType,
        };
        if (!hoverRaf) {
          hoverRaf = requestAnimationFrame(function () {
            hoverRaf = 0;
            var latest = pendingHoverEv;
            pendingHoverEv = null;
            if (latest) performHoverPick(latest);
          });
        }
      } else {
        performHoverPick(ev);
      }
    }

    function onPointerUp(ev) {
      flushSaveParams();
      if (ev.pointerId != null) activePointers.delete(ev.pointerId);
      if (pinchState) {
        if (activePointers.size < 2) {
          pinchState = null;
          dragState = null;
        }
        if (activePointers.size === 0) gestureSuppressedTap = false;
        return;
      }
      if (gestureSuppressedTap) {
        if (activePointers.size === 0) gestureSuppressedTap = false;
        dragState = null;
        return;
      }
      if (!enabled) return;
      var wasDrag = dragState && dragState.id === ev.pointerId ? dragState : null;
      dragState = null;
      if (!wasDrag) return;
      if (ev.type === 'pointercancel') {
        if (wasDrag.mode === 'node' && wasDrag.dragStarted) {
          onNodeDragEnd(wasDrag.node, ev);
          activeDragBounds = null;
          terrainDirty = true;
          batchesDirty = true;
          scheduleRender();
        }
        return;
      }
      if (wasDrag.mode === 'node' && wasDrag.dragStarted) {
        if (stageEl) stageEl.style.cursor = canDragNodesNow() ? 'grab' : '';
        var nbs = wasDrag.neighbors || [];
        var nbCount = nbs.length;
        var pullBack = computeReboundPullBack(nbCount);
        var eqX = wasDrag.node.x + (wasDrag.origX - wasDrag.node.x) * pullBack;
        var eqY = wasDrag.node.y + (wasDrag.origY - wasDrag.node.y) * pullBack;
        var dropDist = Math.hypot(wasDrag.node.x - eqX, wasDrag.node.y - eqY);
        onNodeDragEnd(wasDrag.node, ev);
        if (tier !== 'reduced' && typeof requestAnimationFrame === 'function' && dropDist > 1.5) {
          reboundAnim = {
            node: wasDrag.node,
            origX: wasDrag.origX,
            origY: wasDrag.origY,
            eqX: eqX,
            eqY: eqY,
            dropDist: dropDist,
            vx: 0,
            vy: 0,
            neighbors: nbs,
            startTime: nowMs(),
            lastTime: nowMs(),
            maxDuration: REBOUND_PARAMS.maxDuration,
          };
          terrainDirty = true;
          batchesDirty = true;
          scheduleRender();
        } else {
          activeDragBounds = null;
          terrainDirty = true;
          batchesDirty = true;
          scheduleRender();
        }
        return;
      }
      if (wasDrag.moved) return;
      if (ev.button === 2) return;
      if (!renderer) return;
      var pt = getStageLocalPoint(ev);
      var hit = renderer.pickNodeAtScreen(pt[0], pt[1], wasDrag.isTouch ? 14 : 0);
      if (hit) onNodeClick(hit, ev);
      else onBackgroundClick(ev);
    }

    function onPointerLeave(ev) {
      cancelPendingHover();
      invalidateStageRect();
      if (!enabled) return;
      if (dragState && dragState.mode === 'node' && dragState.dragStarted) return;
      if (stageEl) stageEl.style.cursor = '';
      if (hoveredNode) {
        hoveredNode = null;
        batchesDirty = true;
        scheduleRender();
      }
      onNodeHover(null, ev);
    }

    function onWheel(ev) {
      if (!enabled) return;
      if (entryAnim) cancelEntryAnim();
      ev.preventDefault();
      var factor = ev.deltaY > 0 ? 1.09 : 0.91;
      camState.distance = Math.max(0.65, Math.min(5.5, camState.distance * factor));
      scheduleRender();
    }

    function onContextMenu(ev) {
      if (enabled) ev.preventDefault();
    }

    function bindCanvasEvents() {
      if (!stageEl || typeof stageEl.addEventListener !== 'function') return;
      stageEl.addEventListener('pointerdown', onPointerDown);
      stageEl.addEventListener('pointermove', onPointerMove);
      stageEl.addEventListener('pointerup', onPointerUp);
      stageEl.addEventListener('pointercancel', onPointerUp);
      stageEl.addEventListener('pointerleave', onPointerLeave);
      stageEl.addEventListener('wheel', onWheel, { passive: false });
      stageEl.addEventListener('contextmenu', onContextMenu);
    }
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('resize', invalidateStageRect, { passive: true });
      window.addEventListener('scroll', invalidateStageRect, { passive: true });
    }
    bindCanvasEvents();

    var reducedMq = null;
    function handleReducedMotionChange(ev) {
      env.reducedMotion = !!(ev && ev.matches);
      var nextTier = resolveCapabilityTier(env);
      if (nextTier !== tier) {
        tier = nextTier;
        params = persist ? clampParams(params, tier) : resolveDefaultForTier(tier);
        if (enabled) {
          syncHostClasses();
          scheduleRender();
        }
        onStateChange(getPublicState());
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
          height: params.height,
          contours: params.contours,
          pitch: params.pitch,
          yaw: params.yaw,
        },
        camera: {
          distance: Number(camState.distance.toFixed(3)),
          targetX: Number(camState.targetX.toFixed(3)),
          targetZ: Number(camState.targetZ.toFixed(3)),
        },
      };
    }

    function setEnabled(on, setOpts) {
      var next = !!on;
      if (next && tier === 'unsupported') {
        enabled = false;
        clearStageClasses();
        onStateChange(getPublicState());
        if (!setOpts || !setOpts.silent) onUnsupported();
        return false;
      }
      if (next === enabled) {
        if (next) scheduleRender();
        return true;
      }
      enabled = next;
      if (enabled) {
        params = clampParams(params, tier);
        syncHostClasses();
        if (stageEl) {
          var ok = !!ensureStageCanvases();
          if (!ok) {
            enabled = false;
            clearStageClasses();
            onStateChange(getPublicState());
            if (!setOpts || !setOpts.silent) onUnsupported();
            return false;
          }
        }
        terrainDirty = true;
        batchesDirty = true;
        if (setOpts && setOpts.instant) {
          cancelEntryAnim();
          renderNow();
        } else {
          startCameraTransition(78, 0, camState.distance * 1.14, 0.04, tier === 'lite' ? 950 : 1450, true, true);
        }
      } else {
        cancelPendingHover();
        flushSaveParams();
        invalidateStageRect();
        cancelEntryAnim();
        cancelReboundAnim(true);
        hoveredNode = null;
        dragState = null;
        activePointers.clear();
        pinchState = null;
        gestureSuppressedTap = false;
        activeDragBounds = null;
        if (stageEl) stageEl.style.cursor = '';
        clearStageClasses();
      }
      onStateChange(getPublicState());
      return true;
    }

    function setParam(key, value) {
      if (!Object.prototype.hasOwnProperty.call(BOUNDS, key)) return;
      if (entryAnim) cancelEntryAnim();
      var next = Object.assign({}, params);
      next[key] = value;
      params = clampParams(next, tier);
      saveParamsIfEnabled(params);
      if (key === 'height') batchesDirty = true;
      if (enabled) scheduleRender();
      onStateChange(getPublicState());
    }

    function resetDefaults() {
      var prevPitch = params.pitch;
      var prevYaw = params.yaw;
      var prevDist = camState.distance;
      params = resolveDefaultForTier(tier);
      camState.distance = opts.defaultDistance || 1.72;
      camState.targetX = 0;
      camState.targetZ = 0;
      saveParamsIfEnabled(params);
      batchesDirty = true;
      if (enabled) startCameraTransition(prevPitch, prevYaw, prevDist, 1, 480, true);
      onStateChange(getPublicState());
    }

    function setTopView() {
      var prevPitch = params.pitch;
      var prevYaw = params.yaw;
      var prevDist = camState.distance;
      params = clampParams({
        height: params.height,
        contours: params.contours,
        pitch: TOP_VIEW_PARAMS.pitch,
        yaw: TOP_VIEW_PARAMS.yaw,
      }, tier);
      camState.targetX = 0;
      camState.targetZ = 0;
      saveParamsIfEnabled(params);
      if (enabled) startCameraTransition(prevPitch, prevYaw, prevDist, 1, 460, true);
      onStateChange(getPublicState());
    }

    function syncData(rebuildField, syncOpts) {
      liveSimSync = !!(syncOpts && syncOpts.live);
      if (rebuildField !== false) terrainDirty = true;
      batchesDirty = true;
      if (enabled) scheduleRender();
    }

    function refreshTheme() {
      batchesDirty = true;
      if (enabled) scheduleRender();
    }

    function fitToView() {
      var prevDist = camState.distance;
      var prevTx = camState.targetX;
      var prevTz = camState.targetZ;
      camState.distance = opts.defaultDistance || 1.72;
      camState.targetX = 0;
      camState.targetZ = 0;
      if (enabled) {
        startCameraTransition(params.pitch, params.yaw, prevDist, 1, 420, false, false, prevTx, prevTz);
      }
    }

    function focusNode(nodeId) {
      routeFocusNodeId = nodeId || null;
      batchesDirty = true;
      if (!enabled) return;
      if (!routeFocusNodeId) {
        scheduleRender();
        return;
      }
      if (reboundAnim) cancelReboundAnim(true);
      rebuildFieldAndBatches();
      var m = terrainField && terrainField.nodeMeta && terrainField.nodeMeta[routeFocusNodeId];
      if (!m) {
        scheduleRender();
        return;
      }
      var prevDist = camState.distance;
      var prevTx = camState.targetX;
      var prevTz = camState.targetZ;
      camState.targetX = Math.max(-0.9, Math.min(0.9, m.wx));
      camState.targetZ = Math.max(-0.9, Math.min(0.9, m.wz));
      camState.distance = Math.min(camState.distance, 1.05);
      startCameraTransition(params.pitch, params.yaw, prevDist, 1, 520, false, false, prevTx, prevTz);
    }

    function syncViewport() {
      invalidateStageRect();
      if (!enabled) return;
      renderNow();
    }

    function destroy() {
      cancelPendingHover();
      flushSaveParams();
      cancelEntryAnim();
      cancelReboundAnim(false);
      if (renderRaf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(renderRaf);
      if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
        window.removeEventListener('resize', invalidateStageRect);
        window.removeEventListener('scroll', invalidateStageRect);
      }
      if (stageEl && typeof stageEl.removeEventListener === 'function') {
        stageEl.removeEventListener('pointerdown', onPointerDown);
        stageEl.removeEventListener('pointermove', onPointerMove);
        stageEl.removeEventListener('pointerup', onPointerUp);
        stageEl.removeEventListener('pointercancel', onPointerUp);
        stageEl.removeEventListener('pointerleave', onPointerLeave);
        stageEl.removeEventListener('wheel', onWheel);
        stageEl.removeEventListener('contextmenu', onContextMenu);
      }
      if (reducedMq) {
        if (reducedMq.removeEventListener) reducedMq.removeEventListener('change', handleReducedMotionChange);
        else if (reducedMq.removeListener) reducedMq.removeListener(handleReducedMotionChange);
      }
      if (renderer) renderer.destroy();
      clearStageClasses();
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
      setDragging: function (active) { draggingSlider = !!active; },
      resetDefaults: resetDefaults,
      setFlatView: setTopView,
      setTopView: setTopView,
      syncData: syncData,
      refreshTheme: refreshTheme,
      replayForceBloom: function () {
        if (!enabled || tier === 'unsupported') return;
        terrainDirty = true;
        batchesDirty = true;
        startCameraTransition(params.pitch, params.yaw, camState.distance, 0.12, tier === 'lite' ? 850 : 1200, true, true);
      },
      fitToView: fitToView,
      focusNode: focusNode,
      syncViewport: syncViewport,
      refreshStageRect: invalidateStageRect,
      getScreenTargets: function (limit) {
        return renderer && typeof renderer.getTopScreenTargets === 'function'
          ? renderer.getTopScreenTargets(limit)
          : [];
      },
      getDebugState: function () {
        return {
          supported: tier !== 'unsupported',
          tier: tier,
          enabled: enabled,
          isWebGL2: renderer ? renderer.isWebGL2 : false,
          params: Object.assign({}, params),
          camera: {
            distance: Number(camState.distance.toFixed(3)),
            targetX: Number(camState.targetX.toFixed(3)),
            targetZ: Number(camState.targetZ.toFixed(3)),
          },
          fieldStats: renderer ? renderer.getFieldStats() : null,
          hoveredNodeId: hoveredNode ? hoveredNode.id : null,
          routeFocusNodeId: routeFocusNodeId,
          draggingNodeId: (dragState && dragState.mode === 'node' && dragState.dragStarted && dragState.node) ? dragState.node.id : null,
          rebounding: !!reboundAnim,
          animating: !!entryAnim || !!reboundAnim || !!draggingSlider || !!renderRaf,
          entryAnimating: !!entryAnim,
        };
      },
      destroy: destroy,
    };
  }

  return {
    STORAGE_KEY: STORAGE_KEY,
    BOUNDS: BOUNDS,
    DEFAULTS_BY_TIER: DEFAULTS_BY_TIER,
    TOP_VIEW_PARAMS: TOP_VIEW_PARAMS,
    TIER_BADGE_TEXT: TIER_BADGE_TEXT,
    REBOUND_PARAMS: REBOUND_PARAMS,
    getDefaultsForTier: getDefaultsForTier,
    clampParams: clampParams,
    resolveCapabilityTier: resolveCapabilityTier,
    detectEnvironment: detectEnvironment,
    parseCssColor: parseCssColor,
    computeNodeDegrees: computeNodeDegrees,
    computeSphereRadius3D: computeSphereRadius3D,
    computeNodeElevationY: computeNodeElevationY,
    computeTerrainField: computeTerrainField,
    sampleTerrainHeight: sampleTerrainHeight,
    sampleFootprintMaxHeight: sampleFootprintMaxHeight,
    buildCameraMatrices: buildCameraMatrices,
    projectWorldToScreen: projectWorldToScreen,
    unprojectScreenToPlaneY: unprojectScreenToPlaneY,
    worldToGraphCoords: worldToGraphCoords,
    resolveSphereOverlaps3D: resolveSphereOverlaps3D,
    evalForceMotionXZ: evalForceMotionXZ,
    computeReboundPullBack: computeReboundPullBack,
    createController: createController,
  };
});
