(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RNGraphPerspective = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * 知识图谱 3D 连接度地形渲染层（WebGL2 / WebGL1 + HiDPI 屏幕标签叠加）。
   * - 多峰 L4 高度场由活跃节点的连接数（度）与局部空间主峰合成，兼顾群峰起伏与山谷分隔；
   * - fwidth 抗锯齿等高线/坐标网格、贴地接触阴影与溢光、贴地高亮圈、鼠标地表探针与周期雷达扫波；
   * - 社区贴地平滑凸包疆界 + 可点击交互社区标签 + 缩放 LOD 标签避让；
   * - 阻尼浮点轨道相机、指向光标缩放、侧栏视椎偏置、筛选山体平滑沉降与可选轻量移轴/Bloom 后处理；
   * - 保持零外部 3D 库依赖与向后兼容的控制器 API。
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

  var FOV_DEG = 35;
  var FOV_HALF_RAD = (FOV_DEG * 0.5 * Math.PI) / 180;
  var FOV_TAN_HALF = Math.tan(FOV_HALF_RAD);
  var POINT_SCALE_FACTOR = 1 / FOV_TAN_HALF;

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

  function clampFloat(raw, fallback, min, max) {
    var n = (typeof raw === 'number') ? raw : (raw !== null && raw !== '' && raw !== undefined ? Number(raw) : NaN);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
  }

  function wrapYawDeg(yaw) {
    return ((yaw + 540) % 360) - 180;
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

  /* ── 连接度统计与多峰 L4 地形高度场（O(G^2 + N)）── */

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
      var r3d = Math.max(11, r2d * 1.65);
      return 0.056 * (r3d / 49.5);
    }
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

  function cpuHash2(ix, iz) {
    var n = Math.sin(ix * 127.1 + iz * 311.7) * 43758.5453123;
    return n - Math.floor(n);
  }

  function cpuValueNoise(x, z) {
    var ix = Math.floor(x);
    var iz = Math.floor(z);
    var fx = x - ix;
    var fz = z - iz;
    var ux = fx * fx * (3 - 2 * fx);
    var uz = fz * fz * (3 - 2 * fz);
    var a = cpuHash2(ix, iz);
    var b = cpuHash2(ix + 1, iz);
    var c = cpuHash2(ix, iz + 1);
    var d = cpuHash2(ix + 1, iz + 1);
    return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
  }

  function cpuFbmPlains(wx, wz) {
    return cpuValueNoise(wx * 3.1, wz * 3.1) * 0.62 +
      cpuValueNoise(wx * 6.5 + 7.1, wz * 6.5 - 3.4) * 0.28 +
      cpuValueNoise(wx * 13.2 - 2.3, wz * 13.2 + 5.7) * 0.10;
  }

  /**
   * 为大型图谱挑选多山系驱动峰：结合全局高连接枢纽与 22×22 空间局部极大值，
   * 避免数千低度节点在图心堆成单座平顶大包，同时保留各外围社区独立峰系。
   */
  function selectPeakDrivers(metaEntries, extent) {
    if (metaEntries.length <= 180) return metaEntries;
    var cellN = 22;
    var cellBest = Object.create(null);
    var i;
    for (i = 0; i < metaEntries.length; i++) {
      var item = metaEntries[i];
      if (!item.active && item.degNorm < 0.35) continue;
      var cx = Math.max(0, Math.min(cellN - 1, Math.floor(((item.wx + extent) / (2 * extent)) * cellN)));
      var cz = Math.max(0, Math.min(cellN - 1, Math.floor(((item.wz + extent) / (2 * extent)) * cellN)));
      var key = cz * cellN + cx;
      var score = item.degNorm * (item.active ? 1.0 : 0.12);
      if (!cellBest[key] || score > cellBest[key].score) {
        cellBest[key] = { item: item, score: score };
      }
    }
    var seen = Object.create(null);
    var drivers = [];
    var cellKeys = Object.keys(cellBest);
    for (i = 0; i < cellKeys.length; i++) {
      var picked = cellBest[cellKeys[i]].item;
      if (picked.degNorm > 0.08 || picked.active) {
        seen[picked.id] = true;
        drivers.push(picked);
      }
    }
    var sorted = metaEntries.slice().sort(function (a, b) {
      var sa = a.degNorm * (a.active ? 1.0 : 0.08);
      var sb = b.degNorm * (b.active ? 1.0 : 0.08);
      return sb - sa;
    });
    for (i = 0; i < sorted.length && drivers.length < 320; i++) {
      var topItem = sorted[i];
      if (seen[topItem.id]) continue;
      seen[topItem.id] = true;
      drivers.push(topItem);
    }
    return drivers;
  }

  function computeTerrainField(nodes, degreeInfo, opts) {
    opts = opts || {};
    var G = Math.max(32, Math.min(254, opts.gridSize || 168));
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

    var nodeMeta = Object.create(null);
    var metaEntries = [];
    var activeRadials = [];
    for (i = 0; i < list.length; i++) {
      var ndMap = list[i];
      if (!ndMap || !ndMap.id || ndMap.x == null || ndMap.y == null) continue;
      var wx0 = ((ndMap.x - cx) / halfSpan) * coreSpan;
      var wz0 = ((ndMap.y - cy) / halfSpan) * coreSpan;
      var deg0 = degCounts[ndMap.id] || 0;
      var degNorm0 = Math.pow(Math.log(1 + deg0) / logDen, 1.08);
      var isIncluded0 = !activeSet || activeSet.has(ndMap.id);
      var entry = {
        id: ndMap.id,
        wx: wx0,
        wz: wz0,
        deg: deg0,
        degNorm: degNorm0,
        active: isIncluded0,
      };
      nodeMeta[ndMap.id] = entry;
      metaEntries.push(entry);
      if (isIncluded0) activeRadials.push(Math.hypot(wx0, wz0));
    }

    if (!opts.extent && activeRadials.length) {
      activeRadials.sort(function (a, b) { return a - b; });
      var p99R = activeRadials[Math.min(activeRadials.length - 1, Math.floor((activeRadials.length - 1) * 0.99))];
      extent = Math.max(1.45, Math.min(2.6, p99R * 1.24));
    }

    var broadBuf = new Float32Array(G * G);
    var peakSumBuf = new Float32Array(G * G);
    var tmpBuf = new Float32Array(G * G);
    var cellScale = (G - 1) / (2 * extent);

    for (i = 0; i < metaEntries.length; i++) {
      var metaN = metaEntries[i];
      var contrib = metaN.active ? metaN.degNorm : metaN.degNorm * 0.04;
      if (!(contrib > 0.001)) continue;
      var gxF = (metaN.wx + extent) * cellScale;
      var gzF = (metaN.wz + extent) * cellScale;
      if (gxF < 0 || gxF >= G - 1 || gzF < 0 || gzF >= G - 1) continue;
      var gx0 = Math.floor(gxF);
      var gz0 = Math.floor(gzF);
      var tx = gxF - gx0;
      var tz = gzF - gz0;
      var broadW = Math.pow(contrib, 1.48);
      broadBuf[gz0 * G + gx0] += broadW * (1 - tx) * (1 - tz);
      broadBuf[gz0 * G + gx0 + 1] += broadW * tx * (1 - tz);
      broadBuf[(gz0 + 1) * G + gx0] += broadW * (1 - tx) * tz;
      broadBuf[(gz0 + 1) * G + gx0 + 1] += broadW * tx * tz;
    }

    var isPreviewField = activeList.length <= 180;
    var baseSigma = Math.max(1.8, G * (isPreviewField ? 0.038 : 0.022));
    var drivers = selectPeakDrivers(metaEntries, extent);
    var dx;
    var dz;
    for (i = 0; i < drivers.length; i++) {
      var drv = drivers[i];
      var dContrib = drv.active ? drv.degNorm : drv.degNorm * 0.05;
      if (!(dContrib > 0.004)) continue;
      var dGxF = (drv.wx + extent) * cellScale;
      var dGzF = (drv.wz + extent) * cellScale;
      if (dGxF < -4 || dGxF >= G + 3 || dGzF < -4 || dGzF >= G + 3) continue;
      var sigma = baseSigma * (0.58 + 0.68 * drv.degNorm);
      var rCells = Math.min(18, Math.ceil(sigma * 2.5));
      var inv2Sig2 = 1 / (2 * sigma * sigma);
      var gxC = Math.round(dGxF);
      var gzC = Math.round(dGzF);
      var peakAmp = Math.pow(dContrib, 1.38);
      for (dz = -rCells; dz <= rCells; dz++) {
        var pz = gzC + dz;
        if (pz < 0 || pz >= G) continue;
        var offZ = (pz - dGzF);
        var offZ2 = offZ * offZ;
        for (dx = -rCells; dx <= rCells; dx++) {
          var px = gxC + dx;
          if (px < 0 || px >= G) continue;
          var offX = px - dGxF;
          var d2 = offX * offX + offZ2;
          var t = peakAmp * Math.exp(-d2 * inv2Sig2);
          if (t < 0.015) continue;
          var t2 = t * t;
          // L4 范数叠加：保持各山峰独立高度，避免群峰相交处无限堆高成单一平顶大包
          peakSumBuf[pz * G + px] += t2 * t2;
        }
      }
    }

    var blurR = Math.max(2, Math.round(G * 0.032));
    separableBoxBlur(broadBuf, tmpBuf, G, blurR);
    separableBoxBlur(broadBuf, tmpBuf, G, Math.max(2, Math.round(blurR * 0.65)));

    var peakBuf = new Float32Array(G * G);
    var maxPeak = 1e-6;
    var maxBroad = 1e-6;
    for (i = 0; i < G * G; i++) {
      var pVal = peakSumBuf[i] > 1e-12 ? Math.pow(peakSumBuf[i], 0.25) : 0;
      peakBuf[i] = pVal;
      if (pVal > maxPeak) maxPeak = pVal;
      if (broadBuf[i] > maxBroad) maxBroad = broadBuf[i];
    }

    var heights = new Float32Array(G * G);
    for (dz = 0; dz < G; dz++) {
      var wzCell = -extent + (2 * extent * dz) / (G - 1);
      for (dx = 0; dx < G; dx++) {
        var wxCell = -extent + (2 * extent * dx) / (G - 1);
        var idx = dz * G + dx;
        var bNorm = broadBuf[idx] / maxBroad;
        bNorm = (bNorm / (bNorm + 0.55)) * 1.55;
        var pNorm = peakBuf[idx] / maxPeak;
        var radial = Math.hypot(wxCell, wzCell);
        var envMask = 1 - smoothstep(extent * 0.70, extent * 0.94, radial);
        var baseH = Math.min(1, Math.max(0, 0.14 * Math.pow(bNorm, 1.05) + 0.86 * Math.pow(pNorm, 1.04)));
        var keepNoise = (1 - smoothstep(0.05, 0.24, baseH)) * (1 - smoothstep(extent * 0.48, extent * 0.86, radial));
        var plainFbm = (cpuFbmPlains(wxCell, wzCell) - 0.35) * 0.042 * Math.max(0, keepNoise);
        heights[idx] = Math.max(0, Math.pow(baseH, 1.12) + plainFbm) * envMask;
      }
    }

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
      var isPreviewOverlap = list.length <= 160;
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
          baseRadius: computeSphereRadius3D(mItem.deg, maxDeg, isPreviewOverlap),
        });
      }
      resolveSphereOverlaps3D(overlapList, null, 0, {
        gap: opts.overlapGap != null ? opts.overlapGap : (isPreviewOverlap ? 0.003 : 0.0008),
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

  function sampleTerrainGrad(field, wx, wz) {
    if (!field || !field.grads) return [0, 0];
    var G = field.gridSize;
    var ext = field.extent;
    var u = ((wx + ext) / (2 * ext)) * (G - 1);
    var v = ((wz + ext) / (2 * ext)) * (G - 1);
    if (u < 0 || u > G - 1 || v < 0 || v > G - 1) return [0, 0];
    var x0 = Math.max(0, Math.min(G - 2, Math.floor(u)));
    var z0 = Math.max(0, Math.min(G - 2, Math.floor(v)));
    var tx = u - x0;
    var tz = v - z0;
    var g = field.grads;
    var i00 = (z0 * G + x0) * 2;
    var i10 = (z0 * G + x0 + 1) * 2;
    var i01 = ((z0 + 1) * G + x0) * 2;
    var i11 = ((z0 + 1) * G + x0 + 1) * 2;
    return [
      (g[i00] * (1 - tx) + g[i10] * tx) * (1 - tz) + (g[i01] * (1 - tx) + g[i11] * tx) * tz,
      (g[i00 + 1] * (1 - tx) + g[i10 + 1] * tx) * (1 - tz) + (g[i01 + 1] * (1 - tx) + g[i11 + 1] * tx) * tz,
    ];
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

  function perspectiveMat4(fovRad, aspect, near, far, offsetNdcX) {
    var f = 1 / Math.tan(fovRad / 2);
    var nf = 1 / (near - far);
    var out = new Float32Array(16);
    out[0] = f / Math.max(0.1, aspect);
    out[5] = f;
    out[8] = offsetNdcX || 0;
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

  function buildCameraMatrices(params, camState, aspect, viewportWidth) {
    var pitchRad = ((params.pitch != null ? params.pitch : 36) * Math.PI) / 180;
    var yawRad = ((params.yaw != null ? params.yaw : -18) * Math.PI) / 180;
    var dist = Math.max(0.55, Math.min(4.6, (camState && camState.distance) || 1.92));
    var effDist = dist / Math.min(1, Math.max(0.42, (aspect || 1) / 1.12));
    var tx = (camState && camState.targetX) || 0;
    var tz = (camState && camState.targetZ) || 0;
    var ty = (camState && camState.targetY) || 0.08;
    var offsetPx = (camState && camState.offsetPx) || 0;
    var vpW = Math.max(1, viewportWidth || 1200);
    var offsetNdcX = (2 * offsetPx) / vpW;

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
    var proj = perspectiveMat4((FOV_DEG * Math.PI) / 180, aspect, 0.04, 40, offsetNdcX);
    var viewProj = mulMat4(proj, view);
    return {
      eye: eye,
      target: [tx, ty, tz],
      view: view,
      proj: proj,
      viewProj: viewProj,
      offsetPx: offsetPx,
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

  function screenRayFromCam(localX, localY, cam, width, height) {
    if (!cam || !cam.eye || !cam.view || !width || !height) return null;
    var aspect = width / Math.max(1, height);
    var offsetPx = cam.offsetPx || 0;
    var ndcX = ((localX + offsetPx) / width) * 2 - 1;
    var ndcY = 1 - (localY / height) * 2;
    var v = cam.view;
    var rx = v[0]; var ry = v[4]; var rz = v[8];
    var ux = v[1]; var uy = v[5]; var uz = v[9];
    var fx = -v[2]; var fy = -v[6]; var fz = -v[10];
    var sx = ndcX * aspect * FOV_TAN_HALF;
    var sy = ndcY * FOV_TAN_HALF;
    var dx = fx + rx * sx + ux * sy;
    var dy = fy + ry * sx + uy * sy;
    var dz = fz + rz * sx + uz * sy;
    var len = Math.hypot(dx, dy, dz);
    if (!(len > 1e-6)) return null;
    return {
      ox: cam.eye[0], oy: cam.eye[1], oz: cam.eye[2],
      dx: dx / len, dy: dy / len, dz: dz / len,
    };
  }

  function unprojectScreenToPlaneY(localX, localY, planeY, cam, width, height) {
    var ray = screenRayFromCam(localX, localY, cam, width, height);
    if (!ray || Math.abs(ray.dy) < 1e-4) return null;
    var t = (planeY - ray.oy) / ray.dy;
    if (!(t > 0.05 && t < 20)) return null;
    return [ray.ox + t * ray.dx, planeY, ray.oz + t * ray.dz];
  }

  function raycastTerrainAtScreen(localX, localY, field, heightAmp, cam, width, height) {
    if (!field) return null;
    var ray = screenRayFromCam(localX, localY, cam, width, height);
    if (!ray || ray.dy > -0.02) return null;
    var y = Math.min(heightAmp * 0.45, Math.max(0.02, ray.oy - 0.1));
    var wx = 0;
    var wz = 0;
    for (var i = 0; i < 6; i++) {
      var t = (y - ray.oy) / ray.dy;
      if (!(t > 0.04 && t < 20)) return null;
      wx = ray.ox + ray.dx * t;
      wz = ray.oz + ray.dz * t;
      y = sampleTerrainHeight(field, wx, wz) * heightAmp;
    }
    var extLim = (field.extent || 1.35) * 0.96;
    if (Math.hypot(wx, wz) > extLim) return null;
    return [wx, y, wz];
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

  /* ── 社区贴地平滑凸包生成（Monotone Chain + Chaikin + 地形高度采样）── */

  function convexHull2D(pts) {
    if (!pts || pts.length < 3) return pts ? pts.slice() : [];
    var sorted = pts.slice().sort(function (a, b) {
      return a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1];
    });
    function cross(o, a, b) {
      return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    }
    var lo = [];
    var i;
    for (i = 0; i < sorted.length; i++) {
      while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], sorted[i]) <= 0) lo.pop();
      lo.push(sorted[i]);
    }
    var up = [];
    for (i = sorted.length - 1; i >= 0; i--) {
      while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], sorted[i]) <= 0) up.pop();
      up.push(sorted[i]);
    }
    lo.pop();
    up.pop();
    return lo.concat(up);
  }

  function chaikinSmoothLoop(ring) {
    if (!ring || ring.length < 3) return ring;
    var out = [];
    var n = ring.length;
    for (var i = 0; i < n; i++) {
      var a = ring[i];
      var b = ring[(i + 1) % n];
      out.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25]);
      out.push([a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75]);
    }
    return out;
  }

  function buildCommunityHullSegments(nodes3d, communityMeta, field, heightAmp, activeCommunityId, hoveredCommunityId) {
    if (!field || !communityMeta || !communityMeta.length || !nodes3d || !nodes3d.length) {
      return { positions: [], colors: [], hullCount: 0 };
    }
    var metaById = Object.create(null);
    var i;
    for (i = 0; i < communityMeta.length; i++) {
      var cm = communityMeta[i];
      if (cm && cm.id && cm.id !== 'community-other') metaById[cm.id] = cm;
    }
    var byComm = Object.create(null);
    for (i = 0; i < nodes3d.length; i++) {
      var n3 = nodes3d[i];
      if (!n3 || !n3.visible || !n3.node || !n3.node.community) continue;
      var cid = n3.node.community;
      if (!metaById[cid]) continue;
      if (!byComm[cid]) byComm[cid] = [];
      byComm[cid].push(n3);
    }
    var cids = Object.keys(byComm).filter(function (cid) { return byComm[cid].length >= 3; });
    cids.sort(function (a, b) {
      var pa = (a === activeCommunityId || a === hoveredCommunityId) ? 100000 : 0;
      var pb = (b === activeCommunityId || b === hoveredCommunityId) ? 100000 : 0;
      return (pb + byComm[b].length) - (pa + byComm[a].length);
    });
    var cap = Math.min(cids.length, 10);
    var pos = [];
    var col = [];
    var hullCount = 0;

    for (var cIdx = 0; cIdx < cap; cIdx++) {
      var id = cids[cIdx];
      var members = byComm[id];
      var sx = 0;
      var sz = 0;
      for (i = 0; i < members.length; i++) {
        sx += members[i].wx;
        sz += members[i].wz;
      }
      var cx = sx / members.length;
      var cz = sz / members.length;

      // 裁掉最外 12% 极端引用离群点，避免单篇偏远节点把边界拉成覆盖半张图的三角形
      var ptsWithR = [];
      for (i = 0; i < members.length; i++) {
        var m = members[i];
        ptsWithR.push({ wx: m.wx, wz: m.wz, r: Math.hypot(m.wx - cx, m.wz - cz) });
      }
      ptsWithR.sort(function (a, b) { return a.r - b.r; });
      var keepCount = Math.max(3, Math.floor(ptsWithR.length * 0.88));
      var corePts = [];
      for (i = 0; i < keepCount; i++) {
        corePts.push([ptsWithR[i].wx, ptsWithR[i].wz]);
      }
      var hull = convexHull2D(corePts);
      if (hull.length < 3) continue;

      var pad = 0.042;
      var padded = [];
      for (i = 0; i < hull.length; i++) {
        var dx = hull[i][0] - cx;
        var dz = hull[i][1] - cz;
        var d = Math.hypot(dx, dz) || 1;
        padded.push([hull[i][0] + (dx / d) * pad, hull[i][1] + (dz / d) * pad]);
      }
      var ring = chaikinSmoothLoop(chaikinSmoothLoop(padded));
      var samples = [];
      for (i = 0; i < ring.length; i++) {
        var p0 = ring[i];
        var p1 = ring[(i + 1) % ring.length];
        var segLen = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
        var steps = Math.max(1, Math.min(8, Math.ceil(segLen / 0.04)));
        for (var s = 0; s < steps; s++) {
          var t = s / steps;
          var x = p0[0] + (p1[0] - p0[0]) * t;
          var z = p0[1] + (p1[1] - p0[1]) * t;
          var y = sampleTerrainHeight(field, x, z) * heightAmp + 0.011;
          samples.push([x, y, z]);
        }
      }
      if (samples.length < 4) continue;
      var isHi = (id === activeCommunityId || id === hoveredCommunityId);
      var rgb = parseCssColor(metaById[id].bgColor || '#5b9cf6', [0.36, 0.61, 0.96, 1]);
      var alpha = isHi ? 0.90 : ((activeCommunityId || hoveredCommunityId) ? 0.14 : 0.30);
      var stride = isHi ? 1 : 2; // 激活态绘制完整实线轮廓，常规态虚线步进
      for (i = 0; i < samples.length; i += stride) {
        if (!isHi && (i % 2 === 1)) continue;
        var a = samples[i];
        var b = samples[(i + 1) % samples.length];
        pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
        col.push(rgb[0], rgb[1], rgb[2], alpha, rgb[0], rgb[1], rgb[2], alpha);
      }
      hullCount += 1;
    }

    return { positions: pos, colors: col, hullCount: hullCount };
  }

  /**
   * 3D 地形节点防重叠解算（O(N) 空间网格 + 对称阻尼 Jacobi 松弛）：
   * 每轮累计所有相邻球体的合力后按 kPush / sqrt(冲突度) 同步更新，保证位移随轮次严格单调衰减。
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

  function buildShaders(isWebGL2, hasDerivatives, shaderOpts) {
    var softPeak = !!(shaderOpts && shaderOpts.softPeakHighlight);
    var headV = isWebGL2 ? '#version 300 es\nprecision highp float;\n' : 'precision highp float;\n';
    var headF = isWebGL2
      ? '#version 300 es\nprecision highp float;\n'
      : (hasDerivatives ? '#extension GL_OES_standard_derivatives : enable\nprecision highp float;\n' : 'precision highp float;\n');
    var attr = isWebGL2 ? 'in ' : 'attribute ';
    var varyV = isWebGL2 ? 'out ' : 'varying ';
    var varyF = isWebGL2 ? 'in ' : 'varying ';
    var outDecl = isWebGL2 ? 'out vec4 fragColor;\n' : '';
    var outAssign = isWebGL2 ? 'fragColor' : 'gl_FragColor';
    var texFn = isWebGL2 ? 'texture' : 'texture2D';
    var hasDeriv = isWebGL2 || hasDerivatives;
    var derivFn = hasDeriv
      ? 'float gradMag(float val, vec2 fb) { return max(fwidth(val), 0.00025); }\n' +
        'float aaLine(float v, float w) { float fw = max(fwidth(v), 0.0002); return clamp(w * 0.5 + 0.5 - abs(fract(v - 0.5) - 0.5) / fw, 0.0, 1.0) * (1.0 - smoothstep(0.24, 0.58, fw)); }\n' +
        'float aaCircle(float d, float r) { return clamp(1.2 - abs(d - r) / max(fwidth(d), 0.0008), 0.0, 1.0); }\n'
      : 'float gradMag(float val, vec2 fb) { return max(length(fb) * 0.018, 0.010); }\n' +
        'float aaLine(float v, float w) { float fw = 0.025; return clamp(w * 0.5 + 0.5 - abs(fract(v - 0.5) - 0.5) / fw, 0.0, 1.0); }\n' +
        'float aaCircle(float d, float r) { return 1.0 - smoothstep(0.004, 0.014, abs(d - r)); }\n';

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
      attr + 'float aPrevH;\n' +
      attr + 'vec2 aPrevGrad;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uHeightAmp;\n' +
      'uniform float uHeightScale;\n' +
      'uniform float uForceSettle;\n' +
      'uniform float uMorph;\n' +
      'uniform float uExtent;\n' +
      varyV + 'vec3 vWorld;\n' +
      varyV + 'float vH;\n' +
      varyV + 'vec2 vGrad;\n' +
      varyV + 'float vMeshRadial;\n' +
      varyV + 'float vHeightScale;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  float effAmp = uHeightAmp * uHeightScale;\n' +
      '  float mH = mix(aPrevH, aH, uMorph);\n' +
      '  vec2 mG = mix(aPrevGrad, aGrad, uMorph);\n' +
      '  float y = mH * effAmp;\n' +
      '  vec2 mXZ = applyForceMotion(aXZ, uForceSettle, 0.0);\n' +
      '  vWorld = vec3(mXZ.x, y, mXZ.y);\n' +
      '  vH = mH;\n' +
      '  vGrad = mG * effAmp;\n' +
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
      'uniform float uRadarR;\n' +
      'uniform vec3 uProbe;\n' +
      'uniform sampler2D uContactTex;\n' +
      'uniform vec4 uHL[16];\n' +
      'uniform int uHLCount;\n' +
      'vec3 terrainRampDark(float h) {\n' +
      '  vec3 c0 = vec3(0.095, 0.112, 0.138);\n' +
      '  vec3 c1 = vec3(0.115, 0.220, 0.340);\n' +
      '  vec3 c2 = vec3(0.245, 0.495, 0.820);\n' +
      '  vec3 c3 = ' + (softPeak ? 'vec3(0.950, 0.780, 0.300)' : 'vec3(0.915, 0.615, 0.275)') + ';\n' +
      '  vec3 c4 = ' + (softPeak ? 'vec3(0.980, 0.740, 0.260)' : 'vec3(0.990, 0.455, 0.245)') + ';\n' +
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
      '  vec3 lit = ramp * mix(0.64, ' + (softPeak ? '1.06' : '1.12') + ', diff) + ramp * rim * (uIsDark > 0.5 ? ' + (softPeak ? '0.24' : '0.34') + ' : 0.14);\n' +
      '\n' +
      '  // 接触阴影 / 溢光（让浮空节点球在山体坡面留下软投影或高亮峰底暖晕）\n' +
      '  vec2 cUv = clamp(vWorld.xz / (2.0 * uExtent) + 0.5, 0.0, 1.0);\n' +
      '  float cSigned = (' + texFn + '(uContactTex, cUv).r - 0.5) * 2.0;\n' +
      '  float cShadow = clamp(-cSigned, 0.0, 1.0);\n' +
      '  float cSpill = clamp(cSigned, 0.0, 1.4);\n' +
      '  lit *= 1.0 - cShadow * 0.28 * (1.0 - smoothstep(0.68, 0.95, vH));\n' +
      '  vec3 spillTint = (uIsDark > 0.5) ? ' + (softPeak ? 'vec3(0.96, 0.78, 0.32)' : 'vec3(0.95, 0.62, 0.30)') + ' : vec3(0.24, 0.52, 0.95);\n' +
      '  lit += spillTint * cSpill * (uIsDark > 0.5 ? ' + (softPeak ? '0.06' : '0.26') + ' : 0.14);\n' +
      '\n' +
      '  // fwidth 抗锯齿等高线（陡坡自动淡出防摩尔纹，主/次等高线分层）\n' +
      '  float lvl = vH * uContours;\n' +
      '  float cMinor = aaLine(lvl, 0.95) * smoothstep(0.012, 0.06, vH);\n' +
      '  float cMajor = aaLine(lvl * 0.25, 1.75) * smoothstep(0.02, 0.08, vH);\n' +
      '  float cLine = max(cMinor * 0.58, cMajor);\n' +
      '  float isMajor = cMajor;\n' +
      '  float dmProbe = length(vWorld.xz - uProbe.xy);\n' +
      '  float probeGlow = exp(-dmProbe * dmProbe / (2.0 * 0.085 * 0.085)) * uProbe.z;\n' +
      '  float microLine = aaLine(lvl * 3.0, 0.9) * probeGlow * 0.52;\n' +
      '  float rSweep = (uRadarR >= 0.0) ? exp(-pow((length(vWorld.xz) - uRadarR) / 0.095, 2.0)) : 0.0;\n' +
      '  vec3 contourDark = mix(vec3(0.42, 0.65, 0.92), ' + (softPeak ? 'vec3(1.0, 0.94, 0.74)' : 'vec3(1.0, 0.72, 0.42)') + ', smoothstep(0.48, 0.92, vH));\n' +
      '  vec3 contourLight = mix(vec3(0.24, 0.45, 0.74), vec3(0.84, 0.42, 0.12), smoothstep(0.55, 0.95, vH));\n' +
      '  vec3 contourCol = (uIsDark > 0.5) ? contourDark : contourLight;\n' +
      '  float lineAlpha = max(cLine * mix(0.32, 0.92, smoothstep(0.05, 0.85, vH)) * (1.0 + rSweep * 0.9), microLine);\n' +
      '  lit = mix(lit, contourCol, min(0.95, lineAlpha));\n' +
      '\n' +
      '  // fwidth 抗锯齿基准坐标网格（低谷与平原区域显现）\n' +
      '  vec2 gCoord = vWorld.xz * 7.5;\n' +
      '  float gx = aaLine(gCoord.x, 1.0);\n' +
      '  float gz = aaLine(gCoord.y, 1.0);\n' +
      '  float riseVis = smoothstep(0.18, 0.68, vHeightScale);\n' +
      '  float gridFade = (1.0 - smoothstep(0.28, 0.76, vMeshRadial)) * riseVis;\n' +
      '  float gridLine = max(gx, gz) * (1.0 - smoothstep(0.02, 0.32, vH)) * gridFade;\n' +
      '  vec3 gridCol = (uIsDark > 0.5) ? vec3(0.38, 0.54, 0.78) : vec3(0.28, 0.44, 0.68);\n' +
      '  lit = mix(lit, gridCol, gridLine * (uIsDark > 0.5 ? 0.18 : 0.15));\n' +
      '\n' +
      '  float radial = max(vMeshRadial, length(vWorld.xz) / uExtent);\n' +
      '  vec3 glintTint = (uIsDark > 0.5)\n' +
      '    ? mix(vec3(0.74, 0.90, 1.0), vec3(1.0, 0.88, 0.66), smoothstep(0.38, 0.88, vH))\n' +
      '    : mix(vec3(0.38, 0.68, 1.0), vec3(1.0, 0.72, 0.34), smoothstep(0.42, 0.90, vH));\n' +
      '  float contourReflect = cLine * (0.34 * rim + 0.92 * specKey + 1.25 * specHorizon) * mix(0.75, 1.48, isMajor);\n' +
      '  float sweepWave = (uGlintPhase >= -0.25 && uGlintPhase <= 1.25)\n' +
      '    ? exp(-pow((vH - uGlintPhase) * 6.0, 2.0)) : 0.0;\n' +
      '  lit += glintTint * (contourReflect * (uIsDark > 0.5 ? ' + (softPeak ? '0.30' : '0.78') + ' : 0.48)\n' +
      '    + specHorizon * smoothstep(0.06, 0.85, vH) * (uIsDark > 0.5 ? ' + (softPeak ? '0.14' : '0.40') + ' : 0.20)\n' +
      '    + sweepWave * (0.20 + cLine * 1.15) * (uIsDark > 0.5 ? 0.92 : 0.54));\n' +
      '\n' +
      '  // 贴地高亮圆环（悬停 / 选中 / 搜索命中节点下方）与鼠标地表探针圈\n' +
      '  vec3 ringTint = (uIsDark > 0.5) ? vec3(0.64, 0.86, 1.0) : vec3(0.18, 0.44, 0.88);\n' +
      '  float hlRing = 0.0;\n' +
      '  for (int hi = 0; hi < 16; hi++) {\n' +
      '    if (hi >= uHLCount) break;\n' +
      '    float dHL = length(vWorld.xz - uHL[hi].xy);\n' +
      '    float ringM = aaCircle(dHL, uHL[hi].z) + aaCircle(dHL, uHL[hi].z * 1.42) * 0.45;\n' +
      '    hlRing = max(hlRing, clamp(ringM, 0.0, 1.0) * uHL[hi].w);\n' +
      '  }\n' +
      '  float probeRing = aaCircle(dmProbe, 0.088) * uProbe.z * 0.62;\n' +
      '  float ringMix = clamp(max(hlRing, probeRing), 0.0, 1.0);\n' +
      '  lit = mix(lit, ringTint * (uIsDark > 0.5 ? 1.18 : 1.0), ringMix);\n' +
      '\n' +
      (softPeak
        ? '  float peakGlow = smoothstep(0.82, 1.0, vH);\n' +
          '  if (uIsDark > 0.5) {\n' +
          '    lit += vec3(0.06, 0.04, 0.01) * peakGlow;\n' +
          '    float peakMax = max(lit.r, max(lit.g, lit.b));\n' +
          '    if (peakMax > 0.86) {\n' +
          '      float compressed = 0.86 + 0.12 * (1.0 - exp(-(peakMax - 0.86) / 0.35));\n' +
          '      lit *= compressed / peakMax;\n' +
          '    }\n' +
          '  }\n'
        : '  float peakGlow = smoothstep(0.76, 1.0, vH);\n' +
          '  if (uIsDark > 0.5) lit += vec3(0.28, 0.13, 0.04) * peakGlow;\n') +
      '\n' +
      '  float radialFade = 1.0 - smoothstep(0.52, 0.94, radial);\n' +
      '  float viewDist = length(uEye - vWorld);\n' +
      '  float viewFog = exp(-max(0.0, viewDist - 4.0) * 0.22);\n' +
      '  float vis = clamp(radialFade * viewFog, 0.0, 1.0);\n' +
      '  float valleySoft = clamp(smoothstep(0.008, 0.09, vH) + gridLine * 0.55 + cLine * 0.75 + ringMix * 0.8 + microLine * 0.6, 0.0, 1.0);\n' +
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
      'uniform float uBreathAmp;\n' +
      'uniform float uTime;\n' +
      varyV + 'vec2 vUv;\n' +
      varyV + 'vec4 vColor;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  vec2 mXZ = applyForceMotion(aCenter.xz, uForceSettle, 0.026);\n' +
      '  float bob = uBreathAmp * sin(uTime * 0.95 + aCenter.x * 11.3 + aCenter.z * 7.9) * uForceSettle;\n' +
      '  vec3 centerPos = vec3(mXZ.x, aCenter.y * uHeightScale + bob, mXZ.y);\n' +
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

    var ringVS = headV +
      attr + 'vec3 aCenter;\n' +
      attr + 'vec4 aRingParam;\n' +
      attr + 'vec4 aRingColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uHeightScale;\n' +
      'uniform float uForceSettle;\n' +
      'uniform float uTime;\n' +
      'uniform float uReduced;\n' +
      varyV + 'vec2 vUv;\n' +
      varyV + 'vec4 vColor;\n' +
      forceMotionGLSL +
      'void main() {\n' +
      '  vec2 mXZ = applyForceMotion(aCenter.xz, uForceSettle, 0.026);\n' +
      '  float phase = aRingParam.w > 0.5 && uReduced < 0.5 ? fract(uTime * 0.55 + aCenter.x * 3.1) : 0.0;\n' +
      '  float rad = aRingParam.z * (aRingParam.w > 0.5 ? (1.0 + phase * 1.35) : 1.0);\n' +
      '  vec3 wp = vec3(mXZ.x + aRingParam.x * rad, aCenter.y * uHeightScale, mXZ.y + aRingParam.y * rad);\n' +
      '  float fade = aRingParam.w > 0.5 && uReduced < 0.5 ? (1.0 - phase) : 1.0;\n' +
      '  vUv = aRingParam.xy;\n' +
      '  vColor = vec4(aRingColor.rgb, aRingColor.a * fade * uForceSettle);\n' +
      '  gl_Position = uViewProj * vec4(wp, 1.0);\n' +
      '}\n';

    var ringFS = headF + outDecl +
      varyF + 'vec2 vUv;\n' +
      varyF + 'vec4 vColor;\n' +
      'void main() {\n' +
      '  float r = length(vUv);\n' +
      '  if (r > 1.0 || r < 0.72) discard;\n' +
      '  float band = smoothstep(0.74, 0.84, r) * (1.0 - smoothstep(0.92, 1.0, r));\n' +
      '  ' + outAssign + ' = vec4(vColor.rgb, vColor.a * band);\n' +
      '}\n';

    var ambientVS = headV +
      attr + 'vec3 aPos;\n' +
      attr + 'float aSize;\n' +
      attr + 'vec4 aColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uPointScale;\n' +
      varyV + 'vec4 vColor;\n' +
      'void main() {\n' +
      '  vColor = aColor;\n' +
      '  gl_Position = uViewProj * vec4(aPos, 1.0);\n' +
      '  gl_PointSize = clamp(aSize * uPointScale / max(0.5, gl_Position.w), 1.5, 14.0);\n' +
      '}\n';

    var ambientFS = headF + outDecl +
      varyF + 'vec4 vColor;\n' +
      'void main() {\n' +
      '  vec2 pc = gl_PointCoord * 2.0 - 1.0;\n' +
      '  float r2 = dot(pc, pc);\n' +
      '  if (r2 > 1.0) discard;\n' +
      '  float soft = (1.0 - smoothstep(0.15, 1.0, r2));\n' +
      '  ' + outAssign + ' = vec4(vColor.rgb, vColor.a * soft);\n' +
      '}\n';

    var quadVS = headV +
      attr + 'vec2 aQuadPos;\n' +
      varyV + 'vec2 vUv;\n' +
      'void main() {\n' +
      '  vUv = aQuadPos * 0.5 + 0.5;\n' +
      '  gl_Position = vec4(aQuadPos, 0.0, 1.0);\n' +
      '}\n';

    var brightFS = headF + outDecl +
      varyF + 'vec2 vUv;\n' +
      'uniform sampler2D tScene;\n' +
      'uniform float uThreshold;\n' +
      'void main() {\n' +
      '  vec4 s = ' + texFn + '(tScene, vUv);\n' +
      '  float m = max(s.r, max(s.g, s.b));\n' +
      '  float w = smoothstep(uThreshold, uThreshold + 0.35, m);\n' +
      '  ' + outAssign + ' = vec4(s.rgb * w, s.a * w);\n' +
      '}\n';

    var blurFS = headF + outDecl +
      varyF + 'vec2 vUv;\n' +
      'uniform sampler2D tTex;\n' +
      'uniform vec2 uDir;\n' +
      'void main() {\n' +
      '  vec4 acc = ' + texFn + '(tTex, vUv) * 0.227027;\n' +
      '  acc += ' + texFn + '(tTex, vUv + uDir * 1.384615) * 0.316216;\n' +
      '  acc += ' + texFn + '(tTex, vUv - uDir * 1.384615) * 0.316216;\n' +
      '  acc += ' + texFn + '(tTex, vUv + uDir * 3.230769) * 0.070270;\n' +
      '  acc += ' + texFn + '(tTex, vUv - uDir * 3.230769) * 0.070270;\n' +
      '  ' + outAssign + ' = acc;\n' +
      '}\n';

    var postFS = headF + outDecl +
      varyF + 'vec2 vUv;\n' +
      'uniform sampler2D tScene;\n' +
      'uniform sampler2D tBloom;\n' +
      'uniform vec2 uRes;\n' +
      'uniform float uTime;\n' +
      'uniform float uIsDark;\n' +
      'uniform float uBlurAmp;\n' +
      'uniform float uBloomAmp;\n' +
      'uniform float uGrainAmp;\n' +
      'float hash12(vec2 p) {\n' +
      '  p = fract(p * vec2(123.34, 456.21));\n' +
      '  p += dot(p, p + 45.32);\n' +
      '  return fract(p.x * p.y);\n' +
      '}\n' +
      'void main() {\n' +
      '  float band = smoothstep(0.18, 0.52, abs(vUv.y - 0.53)) * uBlurAmp;\n' +
      '  vec4 col = vec4(0.0);\n' +
      '  if (band > 0.08) {\n' +
      '    for (int i = 0; i < 8; i++) {\n' +
      '      float ang = float(i) * 2.399963;\n' +
      '      float rr = sqrt((float(i) + 0.5) / 8.0) * band;\n' +
      '      col += ' + texFn + '(tScene, clamp(vUv + vec2(cos(ang), sin(ang)) * rr / uRes, 0.0, 1.0));\n' +
      '    }\n' +
      '    col *= 0.125;\n' +
      '  } else {\n' +
      '    col = ' + texFn + '(tScene, vUv);\n' +
      '  }\n' +
      '  vec4 blm = ' + texFn + '(tBloom, vUv);\n' +
      '  vec3 rgb = col.rgb + blm.rgb * uBloomAmp;\n' +
      '  float a = clamp(max(col.a, max(blm.r, max(blm.g, blm.b)) * uBloomAmp * 0.85), 0.0, 1.0);\n' +
      '  float mx = max(rgb.r, max(rgb.g, rgb.b));\n' +
      (softPeak
        ? '  if (uIsDark > 0.5 && mx > 0.86) {\n' +
          '    float nm = 0.86 + 0.10 * (1.0 - exp(-(mx - 0.86) / 0.65));\n' +
          '    rgb = rgb * (nm / mx);\n' +
          '  }\n'
        : '  if (uIsDark > 0.5 && mx > 0.84) {\n' +
          '    float nm = 0.84 + 0.16 * (1.0 - exp(-(mx - 0.84) / 0.85));\n' +
          '    rgb = rgb * (nm / mx);\n' +
          '  }\n') +
      '  float grain = (hash12(vUv * uRes + fract(uTime) * 73.0) - 0.5) * uGrainAmp * a;\n' +
      '  rgb = clamp(rgb + grain, 0.0, 1.0);\n' +
      '  ' + outAssign + ' = vec4(rgb, a);\n' +
      '}\n';

    return {
      terrainVS: terrainVS,
      terrainFS: terrainFS,
      lineVS: lineVS,
      lineFS: lineFS,
      solidVS: solidVS,
      solidFS: solidFS,
      ringVS: ringVS,
      ringFS: ringFS,
      pointVS: pointVS,
      pointFS: pointFS,
      ambientVS: ambientVS,
      ambientFS: ambientFS,
      quadVS: quadVS,
      brightFS: brightFS,
      blurFS: blurFS,
      postFS: postFS,
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

    var softPeakHighlight = !!(opts && opts.softPeakHighlight);
    var labelCtx = labelCanvas ? labelCanvas.getContext('2d') : null;
    var shaders = buildShaders(isWebGL2, hasDerivatives, { softPeakHighlight: softPeakHighlight });
    var terrainProg = createProgram(gl, shaders.terrainVS, shaders.terrainFS);
    var lineProg = createProgram(gl, shaders.lineVS, shaders.lineFS);
    var solidProg = createProgram(gl, shaders.solidVS, shaders.solidFS);
    var ringProg = createProgram(gl, shaders.ringVS, shaders.ringFS);
    var pointProg = createProgram(gl, shaders.pointVS, shaders.pointFS);
    var ambientProg = createProgram(gl, shaders.ambientVS, shaders.ambientFS);
    var brightProg = createProgram(gl, shaders.quadVS, shaders.brightFS);
    var blurProg = createProgram(gl, shaders.quadVS, shaders.blurFS);
    var postProg = createProgram(gl, shaders.quadVS, shaders.postFS);

    var terrainXZBuf = gl.createBuffer();
    var terrainHBuf = gl.createBuffer();
    var terrainGradBuf = gl.createBuffer();
    var terrainPrevHBuf = gl.createBuffer();
    var terrainPrevGradBuf = gl.createBuffer();
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

    var hullPosBuf = gl.createBuffer();
    var hullColBuf = gl.createBuffer();
    var hullVertCount = 0;
    var communityHullCount = 0;

    var hiLinePosBuf = gl.createBuffer();
    var hiLineColBuf = gl.createBuffer();
    var hiLineVertCount = 0;
    var hiEdgeCount = 0;

    var solidPosBuf = gl.createBuffer();
    var solidQuadBuf = gl.createBuffer();
    var solidInstColBuf = gl.createBuffer();
    var solidOpaqueVertCount = 0;
    var solidDimVertCount = 0;

    var ringCenterBuf = gl.createBuffer();
    var ringParamBuf = gl.createBuffer();
    var ringColBuf = gl.createBuffer();
    var ringVertCount = 0;

    var ptPosBuf = gl.createBuffer();
    var ptSizeBuf = gl.createBuffer();
    var ptColBuf = gl.createBuffer();
    var ptOpaqueCount = 0;
    var ptDimCount = 0;

    var ambPosBuf = gl.createBuffer();
    var ambSizeBuf = gl.createBuffer();
    var ambColBuf = gl.createBuffer();

    var screenQuadBuf = gl.createBuffer();
    if (screenQuadBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, screenQuadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    }

    // 128×128 接触阴影/溢光贴图（兼容 JSDOM mock GL）
    var contactTex = typeof gl.createTexture === 'function' ? gl.createTexture() : null;
    if (contactTex && typeof gl.texImage2D === 'function') {
      gl.bindTexture(gl.TEXTURE_2D, contactTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      var neutralTex = new Uint8Array(128 * 128 * 4);
      for (var ni = 0; ni < neutralTex.length; ni += 4) {
        neutralTex[ni] = 128;
        neutralTex[ni + 1] = 128;
        neutralTex[ni + 2] = 128;
        neutralTex[ni + 3] = 255;
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 128, 128, 0, gl.RGBA, gl.UNSIGNED_BYTE, neutralTex);
    }

    // 离屏 FBO 后处理管线（full 档位按需分配，mock 环境或不支持时自动直出）
    var postState = {
      available: typeof gl.FRAMEBUFFER_COMPLETE === 'number' &&
        typeof gl.createFramebuffer === 'function' &&
        typeof gl.createTexture === 'function' &&
        typeof gl.createRenderbuffer === 'function' &&
        typeof gl.checkFramebufferStatus === 'function',
      ready: false,
      w: 0,
      h: 0,
      bw: 0,
      bh: 0,
      sceneFbo: null,
      sceneTex: null,
      depthRb: null,
      bloomFboA: null,
      bloomTexA: null,
      bloomFboB: null,
      bloomTexB: null,
    };

    function createRenderTex(w, h) {
      var tex = gl.createTexture();
      if (!tex) return null;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      return tex;
    }

    function disposePostTargets() {
      if (!postState.ready) return;
      [postState.sceneFbo, postState.bloomFboA, postState.bloomFboB].forEach(function (f) {
        if (f && gl.deleteFramebuffer) gl.deleteFramebuffer(f);
      });
      [postState.sceneTex, postState.bloomTexA, postState.bloomTexB].forEach(function (t) {
        if (t && gl.deleteTexture) gl.deleteTexture(t);
      });
      if (postState.depthRb && gl.deleteRenderbuffer) gl.deleteRenderbuffer(postState.depthRb);
      postState.ready = false;
      postState.sceneFbo = postState.bloomFboA = postState.bloomFboB = null;
      postState.sceneTex = postState.bloomTexA = postState.bloomTexB = null;
      postState.depthRb = null;
    }

    function ensurePostTargets(pxW, pxH) {
      if (!postState.available) return false;
      if (postState.ready && postState.w === pxW && postState.h === pxH) return true;
      disposePostTargets();
      try {
        var bw = Math.max(64, Math.floor(pxW * 0.25));
        var bh = Math.max(64, Math.floor(pxH * 0.25));
        var sceneTex = createRenderTex(pxW, pxH);
        var sceneFbo = gl.createFramebuffer();
        var depthRb = gl.createRenderbuffer();
        if (!sceneTex || !sceneFbo || !depthRb) { postState.available = false; return false; }

        gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, pxW, pxH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTex, 0);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          postState.available = false;
          disposePostTargets();
          return false;
        }

        var bloomTexA = createRenderTex(bw, bh);
        var bloomFboA = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboA);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bloomTexA, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          postState.available = false;
          disposePostTargets();
          return false;
        }

        var bloomTexB = createRenderTex(bw, bh);
        var bloomFboB = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboB);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bloomTexB, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          postState.available = false;
          disposePostTargets();
          return false;
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        postState.ready = true;
        postState.w = pxW;
        postState.h = pxH;
        postState.bw = bw;
        postState.bh = bh;
        postState.sceneFbo = sceneFbo;
        postState.sceneTex = sceneTex;
        postState.depthRb = depthRb;
        postState.bloomFboA = bloomFboA;
        postState.bloomTexA = bloomTexA;
        postState.bloomFboB = bloomFboB;
        postState.bloomTexB = bloomTexB;
        return true;
      } catch (_e) {
        postState.available = false;
        disposePostTargets();
        return false;
      }
    }

    var width = 1200;
    var height = 800;
    var dpr = 1;
    var currentField = null;
    var cachedNodes3d = [];
    var cachedPolyNodes = [];
    var cachedPulseArcs = [];
    var cachedDust = null;
    var cachedDustExtent = 0;
    var lastViewProj = null;
    var lastCam = null;
    var lastHeightScale = 1;
    var lastForceSettle = 1;
    var lastMorph = 1;
    var lastCommunityLabelCount = 0;
    var lastRouteBadgeCount = 0;
    var lastCommunityHitboxes = [];
    var lastHighlightUniforms = new Float32Array(16 * 4);
    var lastHighlightCount = 0;
    var lastHighlightCenterId = null;

    function resolveLocs(prog, attrs, uniforms) {
      var a = {};
      var u = {};
      var i;
      for (i = 0; i < attrs.length; i++) a[attrs[i]] = gl.getAttribLocation(prog, attrs[i]);
      for (i = 0; i < uniforms.length; i++) u[uniforms[i]] = gl.getUniformLocation(prog, uniforms[i]);
      return { a: a, u: u };
    }

    var tLoc = resolveLocs(
      terrainProg,
      ['aXZ', 'aH', 'aGrad', 'aPrevH', 'aPrevGrad'],
      ['uViewProj', 'uHeightAmp', 'uHeightScale', 'uForceSettle', 'uMorph', 'uEye', 'uBg', 'uIsDark', 'uContours', 'uExtent', 'uGlintPhase', 'uRadarR', 'uProbe', 'uContactTex', 'uHL', 'uHLCount']
    );
    var lLoc = resolveLocs(lineProg, ['aPos', 'aColor'], ['uViewProj', 'uExtent', 'uHeightScale', 'uForceSettle', 'uFadeStart']);
    var sLoc = resolveLocs(solidProg, ['aCenter', 'aQuad', 'aInstColor'], ['uViewProj', 'uCamRight', 'uCamUp', 'uLightDirView', 'uHeightScale', 'uForceSettle', 'uBreathAmp', 'uTime']);
    var rLoc = resolveLocs(ringProg, ['aCenter', 'aRingParam', 'aRingColor'], ['uViewProj', 'uHeightScale', 'uForceSettle', 'uTime', 'uReduced']);
    var pLoc = resolveLocs(pointProg, ['aPos', 'aSize', 'aColor'], ['uViewProj', 'uPointScale', 'uLightDirView', 'uHeightScale', 'uForceSettle']);
    var aLoc = resolveLocs(ambientProg, ['aPos', 'aSize', 'aColor'], ['uViewProj', 'uPointScale']);
    var brLoc = resolveLocs(brightProg, ['aQuadPos'], ['tScene', 'uThreshold']);
    var blLoc = resolveLocs(blurProg, ['aQuadPos'], ['tTex', 'uDir']);
    var poLoc = resolveLocs(postProg, ['aQuadPos'], ['tScene', 'tBloom', 'uRes', 'uTime', 'uIsDark', 'uBlurAmp', 'uBloomAmp', 'uGrainAmp']);

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

    function buildPreviousSampleArrays(prevField, nextField) {
      var G = nextField.gridSize;
      if (!prevField) {
        return { heights: nextField.heights, grads: nextField.grads };
      }
      if (prevField.gridSize === G && Math.abs(prevField.extent - nextField.extent) < 1e-4) {
        return { heights: prevField.heights, grads: prevField.grads };
      }
      var ext = nextField.extent;
      var pH = new Float32Array(G * G);
      var pG = new Float32Array(G * G * 2);
      for (var z = 0; z < G; z++) {
        var wz = -ext + (2 * ext * z) / (G - 1);
        for (var x = 0; x < G; x++) {
          var wx = -ext + (2 * ext * x) / (G - 1);
          var idx = z * G + x;
          pH[idx] = sampleTerrainHeight(prevField, wx, wz);
          var gr = sampleTerrainGrad(prevField, wx, wz);
          pG[idx * 2] = gr[0];
          pG[idx * 2 + 1] = gr[1];
        }
      }
      return { heights: pH, grads: pG };
    }

    function uploadTerrainMesh(field, prevField) {
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
      var prevSamples = buildPreviousSampleArrays(prevField, field);

      gl.bindBuffer(gl.ARRAY_BUFFER, terrainHBuf);
      gl.bufferData(gl.ARRAY_BUFFER, field.heights, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, terrainGradBuf);
      gl.bufferData(gl.ARRAY_BUFFER, field.grads, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, terrainPrevHBuf);
      gl.bufferData(gl.ARRAY_BUFFER, prevSamples.heights, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, terrainPrevGradBuf);
      gl.bufferData(gl.ARRAY_BUFFER, prevSamples.grads, gl.STATIC_DRAW);
    }

    function updateContactTexture(field, polyNodes) {
      if (!contactTex || !field || typeof gl.texImage2D !== 'function') return;
      var sz = 128;
      var ext = field.extent || 1.35;
      var buf = new Float32Array(sz * sz);
      var count = Math.min(polyNodes.length, 150);
      var scale = (sz - 1) / (2 * ext);
      for (var i = 0; i < count; i++) {
        var pn = polyNodes[i];
        if (!pn.visible && !pn.focused) continue;
        var isGlow = pn.hovered || pn.selected || pn.degNorm >= 0.56;
        var amp = isGlow ? (0.30 + 0.65 * pn.degNorm) : -(0.34 + 0.22 * (1 - pn.degNorm));
        var wx = isGlow ? pn.wx : (pn.wx - 0.016);
        var wz = isGlow ? pn.wz : (pn.wz - 0.015);
        var worldSig = isGlow ? Math.max(0.032, pn.baseRadius * 2.4) : Math.max(0.022, pn.baseRadius * 1.95);
        var sigCells = worldSig * scale;
        var rad = Math.min(12, Math.ceil(sigCells * 2.4));
        var inv2s2 = 1 / (2 * sigCells * sigCells);
        var gxF = (wx + ext) * scale;
        var gzF = (wz + ext) * scale;
        var gxC = Math.round(gxF);
        var gzC = Math.round(gzF);
        for (var dz = -rad; dz <= rad; dz++) {
          var pz = gzC + dz;
          if (pz < 0 || pz >= sz) continue;
          var oz = pz - gzF;
          for (var dx = -rad; dx <= rad; dx++) {
            var px = gxC + dx;
            if (px < 0 || px >= sz) continue;
            var ox = px - gxF;
            var val = amp * Math.exp(-(ox * ox + oz * oz) * inv2s2);
            buf[pz * sz + px] += val;
          }
        }
      }
      var rgba = new Uint8Array(sz * sz * 4);
      for (var j = 0; j < sz * sz; j++) {
        var enc = Math.round(Math.max(0, Math.min(255, 128 + buf[j] * 110)));
        var o = j * 4;
        rgba[o] = enc;
        rgba[o + 1] = enc;
        rgba[o + 2] = enc;
        rgba[o + 3] = 255;
      }
      gl.bindTexture(gl.TEXTURE_2D, contactTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sz, sz, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    }

    function ensureDustCache(extent) {
      if (cachedDust && Math.abs(cachedDustExtent - extent) < 0.05) return cachedDust;
      cachedDustExtent = extent;
      cachedDust = [];
      for (var i = 0; i < 220; i++) {
        var h1 = cpuHash2(i * 1.7 + 3.1, i * 2.9);
        var h2 = cpuHash2(i * 4.3, i * 1.1 + 9.7);
        var h3 = cpuHash2(i * 2.1 + 5.3, i * 3.7);
        var ang = h1 * Math.PI * 2;
        var dist = Math.sqrt(h2) * extent * 1.05;
        cachedDust.push({
          x: Math.cos(ang) * dist,
          y: 0.05 + h3 * 0.62,
          z: Math.sin(ang) * dist,
          phase: h1 * 6.28318,
        });
      }
      return cachedDust;
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
      var isNodeNew = data.isNodeNew || function () { return false; };
      var showEdges = !!data.showEdges;
      var isLite = !!data.isLite;

      var metaMap = currentField.nodeMeta;
      var highlightCenterId = (hoverNode && hoverNode.id) || (sidebarNode && sidebarNode.id) || routeFocusId || null;
      lastHighlightCenterId = highlightCenterId;
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
          isNew: !!isNodeNew(n),
        });
      }

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

      // 接触阴影/溢光贴图同步
      updateContactTexture(currentField, cachedPolyNodes);

      // 贴地高亮圆环 uniforms（悬停/选中/路线/搜索命中，最多 16 个）
      var hlList = [];
      var hlSeen = Object.create(null);
      for (i = 0; i < cachedPolyNodes.length && hlList.length < 16; i++) {
        var hn = cachedPolyNodes[i];
        if (!hn.hovered && !hn.selected && !(data.searchMatchIds && data.searchMatchIds.has(hn.node.id))) continue;
        if (hlSeen[hn.node.id]) continue;
        hlSeen[hn.node.id] = true;
        var ringR = Math.max(0.038, hn.baseRadius * 2.5);
        hlList.push([hn.wx, hn.wz, ringR, (hn.hovered || hn.selected) ? 1.0 : 0.72]);
      }
      lastHighlightCount = hlList.length;
      for (i = 0; i < 16; i++) {
        var o4 = i * 4;
        if (i < hlList.length) {
          lastHighlightUniforms[o4] = hlList[i][0];
          lastHighlightUniforms[o4 + 1] = hlList[i][1];
          lastHighlightUniforms[o4 + 2] = hlList[i][2];
          lastHighlightUniforms[o4 + 3] = hlList[i][3];
        } else {
          lastHighlightUniforms[o4] = 0;
          lastHighlightUniforms[o4 + 1] = 0;
          lastHighlightUniforms[o4 + 2] = 0;
          lastHighlightUniforms[o4 + 3] = 0;
        }
      }

      // 世界空间节点外环（近期新增 / 悬停 / 选中 / 路线当前节点）
      var ringC = [];
      var ringP = [];
      var ringCol = [];
      var ringQuad = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
      for (i = 0; i < cachedPolyNodes.length; i++) {
        var rn = cachedPolyNodes[i];
        var isCurrRoute = !!(routeFocusId && rn.node.id === routeFocusId);
        if (!rn.isNew && !rn.hovered && !rn.selected && !isCurrRoute) continue;
        var rScale = Math.max(0.016, rn.baseRadius * (rn.hovered || rn.selected ? 1.85 : 1.75));
        var pulseFlag = (rn.isNew || isCurrRoute) ? 1.0 : 0.0;
        var rRgb = isCurrRoute ? [1.0, 0.72, 0.28] : rn.rgb;
        var rAlpha = (rn.hovered || rn.selected || isCurrRoute) ? 0.92 : 0.68;
        for (var rq = 0; rq < ringQuad.length; rq += 2) {
          ringC.push(rn.wx, rn.wy, rn.wz);
          ringP.push(ringQuad[rq], ringQuad[rq + 1], rScale, pulseFlag);
          ringCol.push(rRgb[0], rRgb[1], rRgb[2], rAlpha);
        }
      }
      ringVertCount = ringC.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, ringCenterBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(ringC), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ringParamBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(ringP), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ringColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(ringCol), gl.STATIC_DRAW);

      // 背景低度点精灵缓冲（实心不透明球优先写入深度，半透明暗化球在次轮叠加）
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

      // 枢纽球体缓冲
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

      // 垂直连接杆 + 峰顶光束 + 拱形边 + 大枢纽高亮边 Top-K 限流
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

      var maxBgEdges = isLite ? 1200 : 3600;
      var bgEdgeDrawn = 0;
      hiEdgeCount = 0;
      var baseEdgeRgb = isDarkTheme ? [0.68, 0.78, 0.92] : [0.24, 0.36, 0.56];
      var hiCandidates = [];
      cachedPulseArcs = [];

      function pushRibbonStrip(pts, nx, ny, nz, halfW, rgb, alphaCenter, alphaEdge) {
        for (var k = 0; k < pts.length - 1; k++) {
          var p0 = pts[k];
          var p1 = pts[k + 1];
          var t0 = Math.min(1, k / 2, (pts.length - 1 - k) / 2);
          var t1 = Math.min(1, (k + 1) / 2, (pts.length - 2 - k) / 2);
          var w0 = halfW * (0.55 + 0.45 * t0);
          var w1 = halfW * (0.55 + 0.45 * t1);
          var ox0 = nx * w0; var oy0 = ny * w0; var oz0 = nz * w0;
          var ox1 = nx * w1; var oy1 = ny * w1; var oz1 = nz * w1;
          var r = rgb[0]; var g = rgb[1]; var bl = rgb[2];
          hlPos.push(
            p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p1[0] + ox1, p1[1] + oy1, p1[2] + oz1,
            p0[0], p0[1], p0[2], p1[0] + ox1, p1[1] + oy1, p1[2] + oz1, p0[0] + ox0, p0[1] + oy0, p0[2] + oz0
          );
          hlCol.push(
            r, g, bl, alphaCenter, r, g, bl, alphaCenter, r, g, bl, alphaEdge,
            r, g, bl, alphaCenter, r, g, bl, alphaEdge, r, g, bl, alphaEdge
          );
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

      var centerObj = highlightCenterId ? node3dMap[highlightCenterId] : null;
      var centerComm = centerObj && centerObj.node ? centerObj.node.community : null;

      if (hiAdjList) {
        for (i = 0; i < hiAdjList.length; i++) {
          var heItem = hiAdjList[i];
          var aHi = node3dMap[heItem.s];
          var bHi = node3dMap[heItem.t];
          if (!aHi || !bHi) continue;
          var other = (heItem.s === highlightCenterId) ? bHi : aHi;
          var sameComm = centerComm && other.node && other.node.community === centerComm;
          var pri = (other.focused ? 20000 : 0) + (other.visible ? 10000 : 0) + (sameComm ? 4000 : 0) + Math.min(3999, other.deg);
          hiCandidates.push({ a: aHi, b: bHi, pri: pri });
        }
      }

      if (showEdges) {
        var bgEdgeAlpha = highlightCenterId
          ? (nodes.length <= 160 ? 0.04 : 0.028)
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
          if (cachedPulseArcs.length < 18 && a.degNorm > 0.45 && b.degNorm > 0.35) {
            cachedPulseArcs.push({ a: a, b: b, mx: mx, my: arcPeak, mz: mz, col: baseEdgeRgb });
          }
          bgEdgeDrawn += 1;
        }
      }

      if (hiCandidates.length) {
        hiCandidates.sort(function (u1, u2) { return u2.pri - u1.pri; });
        var maxHiRibbons = isLite ? 32 : (isPreview ? 90 : 64);
        var maxSecondaryThin = isLite ? 48 : 120;
        var col = centerObj ? centerObj.rgb : baseEdgeRgb;
        var coreMix = isDarkTheme ? 0.36 : 0.08;
        var coreCol = [
          col[0] + (1 - col[0]) * coreMix,
          col[1] + (1 - col[1]) * coreMix,
          col[2] + (1 - col[2]) * coreMix,
        ];
        cachedPulseArcs = [];
        for (i = 0; i < hiCandidates.length && i < maxHiRibbons + maxSecondaryThin; i++) {
          var ha = hiCandidates[i].a;
          var hb = hiCandidates[i].b;
          var hmx = (ha.wx + hb.wx) * 0.5;
          var hmz = (ha.wz + hb.wz) * 0.5;
          var hdist = Math.hypot(hb.wx - ha.wx, hb.wz - ha.wz);
          var hTerrainMid = sampleTerrainHeight(currentField, hmx, hmz) * heightAmp;
          var hArcPeak = Math.max(ha.wy, hb.wy, hTerrainMid + 0.055) + Math.min(0.18, hdist * 0.22 + 0.032);
          if (i < maxHiRibbons) {
            hiEdgeCount += 1;
            var perpLen = Math.max(1e-4, hdist);
            var sx = -(hb.wz - ha.wz) / perpLen;
            var sz = (hb.wx - ha.wx) / perpLen;
            var hiSegs = 14;
            var arcPts = [];
            for (var hsg = 0; hsg <= hiSegs; hsg++) {
              var hu = hsg / hiSegs;
              var homt = 1 - hu;
              arcPts.push([
                homt * homt * ha.wx + 2 * homt * hu * hmx + hu * hu * hb.wx,
                homt * homt * ha.wy + 2 * homt * hu * hArcPeak + hu * hu * hb.wy,
                homt * homt * ha.wz + 2 * homt * hu * hmz + hu * hu * hb.wz,
              ]);
            }
            var wScale = isPreview ? 1.0 : 0.62;
            pushRibbonStrip(arcPts, sx, 0, sz, 0.0030 * wScale, col, isDarkTheme ? 0.24 : 0.20, 0.0);
            pushRibbonStrip(arcPts, sx, 0, sz, 0.0012 * wScale, coreCol, 0.94, 0.15);
            pushRibbonStrip(arcPts, 0, 1, 0, 0.0024 * wScale, col, isDarkTheme ? 0.20 : 0.16, 0.0);
            pushRibbonStrip(arcPts, 0, 1, 0, 0.0010 * wScale, coreCol, 0.88, 0.12);
            if (cachedPulseArcs.length < 18) {
              cachedPulseArcs.push({ a: ha, b: hb, mx: hmx, my: hArcPeak, mz: hmz, col: coreCol });
            }
          } else {
            var spx0 = ha.wx;
            var spy0 = ha.wy;
            var spz0 = ha.wz;
            for (var ssg = 1; ssg <= 4; ssg++) {
              var su = ssg / 4;
              var somt = 1 - su;
              var sqx = somt * somt * ha.wx + 2 * somt * su * hmx + su * su * hb.wx;
              var sqy = somt * somt * ha.wy + 2 * somt * su * hArcPeak + su * su * hb.wy;
              var sqz = somt * somt * ha.wz + 2 * somt * su * hmz + su * su * hb.wz;
              lPos.push(spx0, spy0, spz0, sqx, sqy, sqz);
              lCol.push(col[0], col[1], col[2], 0.11, col[0], col[1], col[2], 0.11);
              spx0 = sqx; spy0 = sqy; spz0 = sqz;
            }
          }
        }
      }

      // 社区贴地平滑凸包疆界线
      var hulls = buildCommunityHullSegments(
        cachedNodes3d,
        data.communityLabels || [],
        currentField,
        heightAmp,
        data.activeCommunityId || null,
        data.hoveredCommunityId || null
      );
      communityHullCount = hulls.hullCount;
      hullVertCount = hulls.positions.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, hullPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hulls.positions), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, hullColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hulls.colors), gl.STATIC_DRAW);

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

    var maxAttribs = 5;
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
      lastCommunityHitboxes = [];
      if (renderOpts.showLabels === false) return;

      labelCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var isDarkTheme = !!renderOpts.isDark;
      var getLabelText = renderOpts.getLabelText || function (n) { return n.label || n.id; };
      var camDist = renderOpts.distance || 1.92;
      var heightScale = renderOpts.heightScale != null ? renderOpts.heightScale : 1;
      var forceSettle = renderOpts.forceSettle != null ? renderOpts.forceSettle : 1;
      var heightAmp = renderOpts.heightAmp != null ? renderOpts.heightAmp : 0.46;
      var labelFade = renderOpts.labelFade != null ? renderOpts.labelFade : 1;
      var communityLabels = Array.isArray(renderOpts.communityLabels) ? renderOpts.communityLabels : [];
      var routeBadges = Array.isArray(renderOpts.routeBadges) ? renderOpts.routeBadges : [];
      var activeCommunityId = renderOpts.activeCommunityId || null;
      var hoveredCommunityId = renderOpts.hoveredCommunityId || null;
      lastCommunityLabelCount = 0;
      lastRouteBadgeCount = 0;
      if (labelFade < 0.04) return;

      // 缩放 LOD：全景收敛保持山体通透，拉近逐级解锁局部节点标签
      var zoomIn = Math.max(0, Math.min(1, (1.55 - camDist) / 0.85));
      var hasHoverFocus = !!renderOpts.hasHoverHighlight;
      var maxLabels;
      if (hasHoverFocus) {
        maxLabels = renderOpts.isLite ? 12 : 20;
      } else if (cachedNodes3d.length <= 160) {
        maxLabels = Math.round((renderOpts.isLite ? 7 : 9) + zoomIn * 15);
      } else {
        maxLabels = Math.round((renderOpts.isLite ? 7 : 10) + zoomIn * 24);
      }

      var placed = [];
      var i;
      var j;

      // 0. 路线序号徽标
      if (routeBadges.length && cachedNodes3d.length) {
        var node3dById = Object.create(null);
        for (i = 0; i < cachedNodes3d.length; i++) {
          var cn = cachedNodes3d[i];
          if (cn.node && cn.node.id) node3dById[cn.node.id] = cn;
        }
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

      // 1. 可点击社区聚类中心胶囊标签（留足间距不压住整个山顶）
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
            wx: cWx,
            wz: cWz,
            count: Number(cl.count) || 0,
            depth: cPt[2],
          });
        }
        commCandidates.sort(function (a, b) {
          var aHi = (a.cl.id === activeCommunityId || a.cl.id === hoveredCommunityId) ? 100000 : 0;
          var bHi = (b.cl.id === activeCommunityId || b.cl.id === hoveredCommunityId) ? 100000 : 0;
          if (bHi !== aHi) return bHi - aHi;
          if (b.count !== a.count) return b.count - a.count;
          return a.depth - b.depth;
        });

        var commCap = renderOpts.isLite ? 8 : 11;
        var commAlpha = (hasHoverFocus ? 0.30 : 0.90) * labelFade;
        labelCtx.textAlign = 'center';
        labelCtx.textBaseline = 'middle';
        for (i = 0; i < commCandidates.length && lastCommunityLabelCount < commCap; i++) {
          var cc = commCandidates[i];
          var isCommActive = cc.cl.id === activeCommunityId;
          var isCommHover = cc.cl.id === hoveredCommunityId;
          var fs = Math.max(9.5, Math.min(14, (Number(cc.cl.fontSize) || 11.5) * 0.9));
          if (renderOpts.isLite) fs = Math.max(9, fs * 0.88);
          labelCtx.font = '700 ' + fs.toFixed(1) + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
          var cTxt = '◈ ' + String(cc.cl.label);
          var cPadX = Math.max(7, fs * 0.65);
          var cPadY = Math.max(3.5, fs * 0.32);
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
            if (cBx0 < cPb[2] + 10 && cBx0 + cTw > cPb[0] - 10 && cBy0 < cPb[3] + 8 && cBy0 + cTh > cPb[1] - 8) {
              cOverlap = true;
              break;
            }
          }
          if (cOverlap) continue;
          placed.push([cBx0, cBy0, cBx0 + cTw, cBy0 + cTh]);
          lastCommunityHitboxes.push({
            id: cc.cl.id,
            label: cc.cl.label,
            x0: cBx0,
            y0: cBy0,
            x1: cBx0 + cTw,
            y1: cBy0 + cTh,
            wx: cc.wx,
            wz: cc.wz,
          });

          labelCtx.save();
          labelCtx.globalAlpha = (isCommActive || isCommHover) ? labelFade : commAlpha;
          labelCtx.shadowColor = isDarkTheme ? 'rgba(0, 0, 0, 0.42)' : 'rgba(15, 23, 42, 0.18)';
          labelCtx.shadowBlur = isDarkTheme ? 7 : 5;
          labelCtx.shadowOffsetY = 1;
          labelCtx.fillStyle = (isCommActive || isCommHover)
            ? (cc.cl.bgColor || '#5b9cf6')
            : (isDarkTheme ? 'rgba(17, 21, 28, 0.78)' : 'rgba(255, 255, 255, 0.86)');
          labelCtx.beginPath();
          if (typeof labelCtx.roundRect === 'function') {
            labelCtx.roundRect(cBx0, cBy0, cTw, cTh, cTh / 2);
          } else {
            labelCtx.rect(cBx0, cBy0, cTw, cTh);
          }
          labelCtx.fill();
          labelCtx.shadowColor = 'transparent';
          labelCtx.shadowBlur = 0;
          labelCtx.lineWidth = (isCommActive || isCommHover) ? 1.6 : 1;
          labelCtx.strokeStyle = cc.cl.bgColor || '#5b9cf6';
          labelCtx.stroke();
          labelCtx.globalAlpha = (hasHoverFocus && !isCommActive && !isCommHover ? 0.36 : 0.98) * labelFade;
          labelCtx.fillStyle = (isCommActive || isCommHover)
            ? (cc.cl.textColor || '#ffffff')
            : (isDarkTheme ? '#e6edf5' : '#1f2328');
          labelCtx.fillText(cTxt, cDrawX, cc.sy + 0.5);
          labelCtx.restore();
          lastCommunityLabelCount += 1;
        }
      }

      // 2. 节点标签：仅真正 hovered / selected / routeFocus 主节点免碰撞，所有 1-hop 邻居参与完整避让！
      var primaryId = lastHighlightCenterId;
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
        var isPrimaryFocus = !!(primaryId && n3.node.id === primaryId);
        candidates.push({
          n3: n3,
          sx: pt[0],
          sy: pt[1] - 11,
          isPrimaryFocus: isPrimaryFocus,
          priority: (isPrimaryFocus ? 2000000 : (n3.focused ? 300000 : (n3.neighbor ? 100000 : 0))) + n3.deg * 10 - pt[2],
        });
      }
      candidates.sort(function (a, b) { return b.priority - a.priority; });

      labelCtx.textAlign = 'center';
      labelCtx.textBaseline = 'middle';

      var drawn = 0;
      for (i = 0; i < candidates.length && drawn < maxLabels; i++) {
        var c = candidates[i];
        var isPrim = c.isPrimaryFocus;
        var rawTxt = String(getLabelText(c.n3.node) || c.n3.node.id);
        var txt = rawTxt.length > 18 ? rawTxt.slice(0, 17) + '…' : rawTxt;
        var fontPx = isPrim ? 12 : (c.n3.degNorm > 0.72 ? 11.5 : 10.5);
        labelCtx.font = (isPrim || c.n3.degNorm > 0.72 ? '700 ' : '500 ') + fontPx + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
        var tw = labelCtx.measureText(txt).width + 10;
        var th = isPrim ? 20 : 17;
        var drawX = Math.max(tw / 2 + 12, Math.min(width - tw / 2 - 12, c.sx));
        if (Math.abs(drawX - c.sx) > 16) continue;
        var bx0 = drawX - tw / 2;
        var by0 = c.sy - th / 2;
        if (by0 < 14 || by0 + th > height - 28) continue;
        if (by0 < 50 && bx0 + tw > width - 144) continue;
        var padCollisionX = hasHoverFocus ? 5 : 9;
        var padCollisionY = hasHoverFocus ? 4 : 7;
        var overlaps = false;
        if (!isPrim) {
          for (j = 0; j < placed.length; j++) {
            var pb = placed[j];
            if (bx0 < pb[2] + padCollisionX && bx0 + tw > pb[0] - padCollisionX && by0 < pb[3] + padCollisionY && by0 + th > pb[1] - padCollisionY) {
              overlaps = true;
              break;
            }
          }
        }
        if (overlaps) continue;
        placed.push([bx0, by0, bx0 + tw, by0 + th]);

        labelCtx.save();
        labelCtx.globalAlpha = labelFade * (isPrim ? 1.0 : 0.90);
        labelCtx.fillStyle = isPrim
          ? (isDarkTheme ? 'rgba(15, 19, 25, 0.88)' : 'rgba(255, 255, 255, 0.94)')
          : (isDarkTheme ? 'rgba(15, 19, 25, 0.56)' : 'rgba(255, 255, 255, 0.74)');
        labelCtx.beginPath();
        if (typeof labelCtx.roundRect === 'function') {
          labelCtx.roundRect(bx0, by0, tw, th, 4);
        } else {
          labelCtx.rect(bx0, by0, tw, th);
        }
        labelCtx.fill();
        if (isPrim) {
          labelCtx.strokeStyle = 'rgba(' + Math.round(c.n3.rgb[0] * 255) + ',' + Math.round(c.n3.rgb[1] * 255) + ',' + Math.round(c.n3.rgb[2] * 255) + ', 0.90)';
          labelCtx.lineWidth = 1.25;
          labelCtx.stroke();
        }
        labelCtx.fillStyle = isPrim
          ? (isDarkTheme ? '#ffffff' : '#141412')
          : (isDarkTheme ? 'rgba(232, 237, 243, 0.92)' : 'rgba(30, 32, 36, 0.90)');
        labelCtx.fillText(txt, drawX, c.sy + 0.5);
        labelCtx.restore();
        drawn += 1;
      }
      labelCtx.globalAlpha = 1;
    }

    function renderAmbientParticles(viewProj, renderOpts, timeSec) {
      if (!renderOpts.ambientEnabled || !currentField) return;
      var heightScale = renderOpts.heightScale != null ? renderOpts.heightScale : 1;
      var forceSettle = renderOpts.forceSettle != null ? renderOpts.forceSettle : 1;
      var isDarkTheme = !!renderOpts.isDark;
      var pos = [];
      var sz = [];
      var col = [];
      var i;

      // 1. 弧线流动脉冲粒子
      for (i = 0; i < cachedPulseArcs.length; i++) {
        var arc = cachedPulseArcs[i];
        for (var k = 0; k < 2; k++) {
          var t = (timeSec * 0.24 + i * 0.19 + k * 0.5) % 1;
          var u = 1 - t;
          var ax = u * u * arc.a.wx + 2 * u * t * arc.mx + t * t * arc.b.wx;
          var ay = (u * u * arc.a.wy + 2 * u * t * arc.my + t * t * arc.b.wy) * heightScale;
          var az = u * u * arc.a.wz + 2 * u * t * arc.mz + t * t * arc.b.wz;
          var mXZ = evalForceMotionXZ(ax, az, forceSettle, 0.026);
          pos.push(mXZ[0], ay, mXZ[1]);
          sz.push(0.011);
          col.push(arc.col[0], arc.col[1], arc.col[2], Math.sin(Math.PI * t) * 0.88 * forceSettle);
        }
      }

      // 2. 高峰顶部升腾微粒
      var peakEmitted = 0;
      for (i = 0; i < cachedPolyNodes.length && peakEmitted < 12; i++) {
        var pn = cachedPolyNodes[i];
        if (!pn.visible || pn.degNorm < 0.55) continue;
        peakEmitted += 1;
        for (var e = 0; e < 7; e++) {
          var phase = (timeSec * (0.14 + (e % 3) * 0.04) + e * 0.143 + i * 0.31) % 1;
          var ang = timeSec * 0.6 + e * 1.3 + i;
          var spread = (0.008 + phase * 0.036);
          var mxz = evalForceMotionXZ(pn.wx + Math.cos(ang) * spread, pn.wz + Math.sin(ang) * spread, forceSettle, 0.026);
          var py = pn.wy * heightScale + Math.pow(phase, 0.85) * (0.12 + pn.degNorm * 0.12);
          pos.push(mxz[0], py, mxz[1]);
          sz.push(0.009 * (1 - 0.4 * phase));
          col.push(pn.rgb[0], pn.rgb[1], pn.rgb[2], Math.pow(Math.sin(Math.PI * phase), 0.7) * (isDarkTheme ? 0.72 : 0.42) * forceSettle);
        }
      }

      // 3. 空间漂浮微尘（增强旋转/缩放立体视差）
      var dust = ensureDustCache(currentField.extent || 1.35);
      for (i = 0; i < dust.length; i++) {
        var d = dust[i];
        pos.push(
          d.x + Math.sin(timeSec * 0.12 + d.phase) * 0.028,
          d.y + Math.sin(timeSec * 0.17 + d.phase * 1.9) * 0.014,
          d.z + Math.cos(timeSec * 0.10 + d.phase) * 0.028
        );
        sz.push(0.0065);
        col.push(isDarkTheme ? 0.72 : 0.32, isDarkTheme ? 0.82 : 0.44, isDarkTheme ? 0.96 : 0.62, (isDarkTheme ? 0.18 : 0.14) * forceSettle);
      }

      var count = pos.length / 3;
      if (!count) return;
      var activeAttrs = {};
      gl.useProgram(ambientProg);
      gl.uniformMatrix4fv(aLoc.u.uViewProj, false, viewProj);
      gl.uniform1f(aLoc.u.uPointScale, height * dpr * POINT_SCALE_FACTOR);
      gl.bindBuffer(gl.ARRAY_BUFFER, ambPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ambSizeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sz), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ambColBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(col), gl.STATIC_DRAW);
      bindAttrib(aLoc.a.aPos, ambPosBuf, 3, activeAttrs);
      bindAttrib(aLoc.a.aSize, ambSizeBuf, 1, activeAttrs);
      bindAttrib(aLoc.a.aColor, ambColBuf, 4, activeAttrs);
      useAttrs(activeAttrs);
      gl.drawArrays(gl.POINTS, 0, count);
    }

    function drawFullscreenPostQuad(locPos) {
      if (locPos < 0 || !screenQuadBuf) return;
      var activeAttrs = {};
      bindAttrib(locPos, screenQuadBuf, 2, activeAttrs);
      useAttrs(activeAttrs);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    function render(params, camState, renderOpts) {
      if (!currentField) return;
      var aspect = width / Math.max(1, height);
      var cam = buildCameraMatrices(params, camState, aspect, width);
      lastViewProj = cam.viewProj;
      lastCam = cam;
      var heightAmp = (params.height / 100) * 0.46;
      var heightScale = renderOpts.heightScale != null ? renderOpts.heightScale : 1;
      var forceSettle = renderOpts.forceSettle != null ? renderOpts.forceSettle : 1;
      var morphT = renderOpts.morph != null ? renderOpts.morph : 1;
      var glintPhase = renderOpts.glintPhase != null ? renderOpts.glintPhase : -1;
      var timeSec = renderOpts.timeSec || 0;
      var radarR = renderOpts.radarEnabled ? (((timeSec / 11) % 1) * currentField.extent * 1.15) : -1;
      var probe = renderOpts.probe || { wx: 0, wz: 0, strength: 0 };
      lastHeightScale = heightScale;
      lastForceSettle = forceSettle;
      lastMorph = morphT;
      var isDarkTheme = !!renderOpts.isDark;
      var isClassicUi = typeof document !== 'undefined' && document.documentElement.getAttribute('data-ui') === 'classic';
      var fallbackBg = isClassicUi
        ? (isDarkTheme ? '#191919' : '#f7f6f3')
        : (isDarkTheme ? '#0a0b0d' : '#f5f4f0');
      var bgArr = parseCssColorFast(renderOpts.bgColor || fallbackBg, [0.039, 0.043, 0.051, 1]);

      var pxW = Math.max(1, Math.round(width * dpr));
      var pxH = Math.max(1, Math.round(height * dpr));
      var usePost = !!renderOpts.postFxEnabled && ensurePostTargets(pxW, pxH);

      if (usePost) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, postState.sceneFbo);
        gl.viewport(0, 0, pxW, pxH);
      } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, pxW, pxH);
      }

      gl.depthMask(true);
      gl.disable(gl.CULL_FACE);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      // 1. 地形曲面网格
      var activeAttrs = {};
      gl.useProgram(terrainProg);
      gl.uniformMatrix4fv(tLoc.u.uViewProj, false, cam.viewProj);
      gl.uniform1f(tLoc.u.uHeightAmp, heightAmp);
      gl.uniform1f(tLoc.u.uHeightScale, heightScale);
      gl.uniform1f(tLoc.u.uForceSettle, forceSettle);
      gl.uniform1f(tLoc.u.uMorph, morphT);
      gl.uniform3f(tLoc.u.uEye, cam.eye[0], cam.eye[1], cam.eye[2]);
      gl.uniform3f(tLoc.u.uBg, bgArr[0], bgArr[1], bgArr[2]);
      gl.uniform1f(tLoc.u.uIsDark, isDarkTheme ? 1 : 0);
      gl.uniform1f(tLoc.u.uContours, params.contours || 16);
      gl.uniform1f(tLoc.u.uExtent, currentField.extent);
      gl.uniform1f(tLoc.u.uGlintPhase, glintPhase);
      gl.uniform1f(tLoc.u.uRadarR, radarR);
      gl.uniform3f(tLoc.u.uProbe, probe.wx, probe.wz, probe.strength);
      if (tLoc.u.uHL != null) gl.uniform4fv(tLoc.u.uHL, lastHighlightUniforms);
      if (tLoc.u.uHLCount != null) gl.uniform1i(tLoc.u.uHLCount, lastHighlightCount);
      if (contactTex && gl.activeTexture && gl.bindTexture && tLoc.u.uContactTex != null) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, contactTex);
        gl.uniform1i(tLoc.u.uContactTex, 0);
      }
      bindAttrib(tLoc.a.aXZ, terrainXZBuf, 2, activeAttrs);
      bindAttrib(tLoc.a.aH, terrainHBuf, 1, activeAttrs);
      bindAttrib(tLoc.a.aGrad, terrainGradBuf, 2, activeAttrs);
      bindAttrib(tLoc.a.aPrevH, terrainPrevHBuf, 1, activeAttrs);
      bindAttrib(tLoc.a.aPrevGrad, terrainPrevGradBuf, 2, activeAttrs);
      useAttrs(activeAttrs);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, terrainIdxBuf);
      gl.drawElements(gl.TRIANGLES, terrainIdxCount, gl.UNSIGNED_SHORT, 0);

      // 2. 实心不透明枢纽球与背景圆球（共同先写入深度，材质与 3D 立体视图 Lambert 着光一致）
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
        gl.uniform1f(sLoc.u.uBreathAmp, renderOpts.ambientEnabled ? 0.0022 : 0.0);
        gl.uniform1f(sLoc.u.uTime, timeSec);
        bindAttrib(sLoc.a.aCenter, solidPosBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aQuad, solidQuadBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aInstColor, solidInstColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, 0, solidOpaqueVertCount);
      }

      // 2a. 不透明低度圆球点精灵（与大枢纽球同轮写深度）
      if (ptOpaqueCount > 0) {
        activeAttrs = {};
        gl.useProgram(pointProg);
        gl.uniformMatrix4fv(pLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(pLoc.u.uPointScale, height * dpr * POINT_SCALE_FACTOR);
        gl.uniform3f(pLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(pLoc.u.uHeightScale, heightScale);
        gl.uniform1f(pLoc.u.uForceSettle, forceSettle);
        bindAttrib(pLoc.a.aPos, ptPosBuf, 3, activeAttrs);
        bindAttrib(pLoc.a.aSize, ptSizeBuf, 1, activeAttrs);
        bindAttrib(pLoc.a.aColor, ptColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.POINTS, 0, ptOpaqueCount);
      }

      gl.depthMask(false);

      // 2b. 社区贴地平滑凸包疆界线
      if (hullVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(lineProg);
        gl.uniformMatrix4fv(lLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(lLoc.u.uExtent, currentField.extent);
        gl.uniform1f(lLoc.u.uHeightScale, heightScale);
        gl.uniform1f(lLoc.u.uForceSettle, forceSettle);
        gl.uniform1f(lLoc.u.uFadeStart, 0.88);
        bindAttrib(lLoc.a.aPos, hullPosBuf, 3, activeAttrs);
        bindAttrib(lLoc.a.aColor, hullColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.LINES, 0, hullVertCount);
      }

      // 3. 垂线支撑杆、峰顶光柱与背景连线
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

      // 4. 暗化低度圆球点精灵（焦点/筛选态半透明层）
      if (ptDimCount > 0) {
        activeAttrs = {};
        gl.useProgram(pointProg);
        gl.uniformMatrix4fv(pLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(pLoc.u.uPointScale, height * dpr * POINT_SCALE_FACTOR);
        gl.uniform3f(pLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(pLoc.u.uHeightScale, heightScale);
        gl.uniform1f(pLoc.u.uForceSettle, forceSettle);
        bindAttrib(pLoc.a.aPos, ptPosBuf, 3, activeAttrs);
        bindAttrib(pLoc.a.aSize, ptSizeBuf, 1, activeAttrs);
        bindAttrib(pLoc.a.aColor, ptColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.POINTS, ptOpaqueCount, ptDimCount);
      }

      // 4b. 暗化枢纽球
      if (solidDimVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(solidProg);
        gl.uniformMatrix4fv(sLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform3f(sLoc.u.uCamRight, camRight[0], camRight[1], camRight[2]);
        gl.uniform3f(sLoc.u.uCamUp, camUp[0], camUp[1], camUp[2]);
        gl.uniform3f(sLoc.u.uLightDirView, lightDirView[0], lightDirView[1], lightDirView[2]);
        gl.uniform1f(sLoc.u.uHeightScale, heightScale);
        gl.uniform1f(sLoc.u.uForceSettle, forceSettle);
        gl.uniform1f(sLoc.u.uBreathAmp, 0.0);
        gl.uniform1f(sLoc.u.uTime, timeSec);
        bindAttrib(sLoc.a.aCenter, solidPosBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aQuad, solidQuadBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aInstColor, solidInstColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, solidOpaqueVertCount, solidDimVertCount);
      }

      // 4c. 节点外围水平脉冲环
      if (ringVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(ringProg);
        gl.uniformMatrix4fv(rLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(rLoc.u.uHeightScale, heightScale);
        gl.uniform1f(rLoc.u.uForceSettle, forceSettle);
        gl.uniform1f(rLoc.u.uTime, timeSec);
        gl.uniform1f(rLoc.u.uReduced, renderOpts.isReduced ? 1 : 0);
        bindAttrib(rLoc.a.aCenter, ringCenterBuf, 3, activeAttrs);
        bindAttrib(rLoc.a.aRingParam, ringParamBuf, 4, activeAttrs);
        bindAttrib(rLoc.a.aRingColor, ringColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, 0, ringVertCount);
      }

      // 5. 高亮邻居发光丝带
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

      // 5b. 单 Draw Call 氛围粒子（弧线脉冲 + 峰顶升腾微粒 + 空间微尘）
      renderAmbientParticles(cam.viewProj, renderOpts, timeSec);

      gl.depthMask(true);

      // 5c. 离屏后处理 Pass：Bloom 提取 + 双遍模糊 + 移轴柔焦 + 微量颗粒防色带
      if (usePost) {
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);

        // Bright extract -> bloomFboA
        gl.bindFramebuffer(gl.FRAMEBUFFER, postState.bloomFboA);
        gl.viewport(0, 0, postState.bw, postState.bh);
        gl.useProgram(brightProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, postState.sceneTex);
        gl.uniform1i(brLoc.u.tScene, 0);
        gl.uniform1f(brLoc.u.uThreshold, isDarkTheme ? (softPeakHighlight ? 0.88 : 0.62) : 0.88);
        drawFullscreenPostQuad(brLoc.a.aQuadPos);

        // Horizontal blur -> bloomFboB
        gl.bindFramebuffer(gl.FRAMEBUFFER, postState.bloomFboB);
        gl.useProgram(blurProg);
        gl.bindTexture(gl.TEXTURE_2D, postState.bloomTexA);
        gl.uniform1i(blLoc.u.tTex, 0);
        gl.uniform2f(blLoc.u.uDir, 1.35 / postState.bw, 0);
        drawFullscreenPostQuad(blLoc.a.aQuadPos);

        // Vertical blur -> bloomFboA
        gl.bindFramebuffer(gl.FRAMEBUFFER, postState.bloomFboA);
        gl.bindTexture(gl.TEXTURE_2D, postState.bloomTexB);
        gl.uniform2f(blLoc.u.uDir, 0, 1.35 / postState.bh);
        drawFullscreenPostQuad(blLoc.a.aQuadPos);

        // Final composite -> backbuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, pxW, pxH);
        gl.useProgram(postProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, postState.sceneTex);
        gl.uniform1i(poLoc.u.tScene, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, postState.bloomTexA);
        gl.uniform1i(poLoc.u.tBloom, 1);
        gl.uniform2f(poLoc.u.uRes, pxW, pxH);
        gl.uniform1f(poLoc.u.uTime, timeSec);
        gl.uniform1f(poLoc.u.uIsDark, isDarkTheme ? 1 : 0);
        gl.uniform1f(poLoc.u.uBlurAmp, 2.2 * dpr);
        gl.uniform1f(poLoc.u.uBloomAmp, isDarkTheme ? (softPeakHighlight ? 0.06 : 0.32) : 0.10);
        gl.uniform1f(poLoc.u.uGrainAmp, isDarkTheme ? 0.008 : 0.012);
        drawFullscreenPostQuad(poLoc.a.aQuadPos);
        gl.activeTexture(gl.TEXTURE0);
      }

      // 6. HiDPI 屏幕空间标签叠加层（100% 保持锐利文字）
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
        activeCommunityId: renderOpts.activeCommunityId,
        hoveredCommunityId: renderOpts.hoveredCommunityId,
      });
    }

    function pickNodeAtScreen(localX, localY, extraSlop) {
      if (!lastViewProj || !cachedNodes3d.length) return null;
      var vp = lastViewProj;
      var m0 = vp[0]; var m1 = vp[1]; var m2 = vp[2]; var m3 = vp[3];
      var m4 = vp[4]; var m5 = vp[5]; var m6 = vp[6]; var m7 = vp[7];
      var m8 = vp[8]; var m9 = vp[9]; var m10 = vp[10]; var m11 = vp[11];
      var m12 = vp[12]; var m13 = vp[13]; var m14 = vp[14]; var m15 = vp[15];
      var slop = extraSlop > 0 ? extraSlop : 0;
      var settled = lastForceSettle >= 0.999;
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
          var rNorm = rad > 1.05 ? 1 : rad / 1.05;
          var delay = rNorm * 0.24;
          var u = Math.max(0, Math.min(1, (lastForceSettle - delay) / Math.max(0.01, 1 - delay)));
          var spring = 1 - Math.exp(-5.4 * u) * Math.cos(6.28318 * u);
          var blendEnd = smoothstep(0.84, 1.0, u);
          var springNorm = spring + (1 - spring) * blendEnd;
          var radialScale = 0.46 + 0.54 * springNorm;
          var swirl = (1 - springNorm) * (0.28 + 0.16 * rNorm);
          var cs = Math.cos(swirl);
          var sn = Math.sin(swirl);
          var ox = (wx * cs - wz * sn) * radialScale;
          var oz = (wx * sn + wz * cs) * radialScale;
          var env = Math.sin(Math.PI * u) * Math.exp(-2.6 * u) * (1 - blendEnd);
          var ph1 = wx * 17.3 + wz * 11.7;
          var ph2 = wx * 13.1 - wz * 19.4;
          wx = ox + Math.sin(ph1 + u * 10.5) * 0.026 * env;
          wz = oz + Math.cos(ph2 + u * 9.3) * 0.026 * env;
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

    function pickCommunityAtScreen(localX, localY) {
      for (var i = 0; i < lastCommunityHitboxes.length; i++) {
        var hb = lastCommunityHitboxes[i];
        if (localX >= hb.x0 && localX <= hb.x1 && localY >= hb.y0 && localY <= hb.y1) {
          return hb;
        }
      }
      return null;
    }

    function destroy() {
      if (gl.isContextLost && gl.isContextLost()) return;
      disposePostTargets();
      if (contactTex && gl.deleteTexture) gl.deleteTexture(contactTex);
      [terrainXZBuf, terrainHBuf, terrainGradBuf, terrainPrevHBuf, terrainPrevGradBuf, terrainIdxBuf,
        linePosBuf, lineColBuf, hullPosBuf, hullColBuf, hiLinePosBuf, hiLineColBuf,
        solidPosBuf, solidQuadBuf, solidInstColBuf, ringCenterBuf, ringParamBuf, ringColBuf,
        ptPosBuf, ptSizeBuf, ptColBuf, ambPosBuf, ambSizeBuf, ambColBuf, screenQuadBuf].forEach(function (b) {
        if (b) gl.deleteBuffer(b);
      });
      [terrainProg, lineProg, solidProg, ringProg, pointProg, ambientProg, brightProg, blurProg, postProg].forEach(function (p) {
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
      pickCommunityAtScreen: pickCommunityAtScreen,
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
      raycastTerrain: function (localX, localY, heightAmp) {
        return raycastTerrainAtScreen(localX, localY, currentField, heightAmp, lastCam, width, height);
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
      getCommunityHitboxes: function () {
        return lastCommunityHitboxes.slice();
      },
      getFieldStats: function () {
        return currentField ? {
          gridSize: currentField.gridSize,
          maxDegree: currentField.maxDegree,
          nodeCount: cachedNodes3d.length,
          polyCount: cachedPolyNodes.length,
          hiEdgeCount: hiEdgeCount,
          communityLabelCount: lastCommunityLabelCount,
          communityHullCount: communityHullCount,
          routeBadgeCount: lastRouteBadgeCount,
          forceSettle: Number(lastForceSettle.toFixed(3)),
          morph: Number(lastMorph.toFixed(3)),
        } : null;
      },
      isPostAvailable: function () { return postState.available; },
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
    var getActiveCommunityId = opts.getActiveCommunityId || function () { return activeCommunityOverride; };
    var getSearchMatchIds = opts.getSearchMatchIds || function () { return null; };
    var isNodeNew = opts.isNodeNew || function () { return false; };
    var isLabelsVisible = opts.isLabelsVisible || function () { return true; };
    var isEdgesVisible = opts.isEdgesVisible || function () { return false; };
    var onNodeHover = opts.onNodeHover || function () {};
    var onNodeClick = opts.onNodeClick || function () {};
    var onCommunityClick = opts.onCommunityClick || function () {};
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
    var defaultDist = opts.defaultDistance || 1.92;
    var targetCam = {
      pitch: params.pitch,
      yaw: params.yaw,
      distance: defaultDist,
      targetX: 0,
      targetY: 0.08,
      targetZ: 0,
      offsetPx: 0,
    };
    var smoothCam = {
      pitch: params.pitch,
      yaw: params.yaw,
      distance: defaultDist,
      targetX: 0,
      targetY: 0.08,
      targetZ: 0,
      offsetPx: 0,
    };

    var renderer = null;
    var glCanvas = null;
    var labelCanvas = null;
    var degreeInfo = null;
    var terrainField = null;
    var prevTerrainField = null;
    var activeDragBounds = null;
    var hoveredNode = null;
    var hoveredCommunityId = null;
    var activeCommunityOverride = null;
    var routeFocusNodeId = null;
    var batchesDirty = true;
    var terrainDirty = true;
    var morphAnim = null;
    var pendingMorphOnBuild = false;
    var renderRaf = 0;
    var lastFrameTime = 0;
    var startEpochMs = 0;
    var lastUserInputAt = 0;
    var degradedPerf = false;
    var slowFrameStreak = 0;
    var probeState = { wx: 0, wz: 0, strength: 0, targetStrength: 0 };

    function nowMs() {
      return (typeof performance !== 'undefined' && typeof performance.now === 'function')
        ? performance.now()
        : Date.now();
    }

    function markUserInput() {
      lastUserInputAt = nowMs();
    }

    function snapSmoothToTarget() {
      smoothCam.pitch = targetCam.pitch;
      smoothCam.yaw = targetCam.yaw;
      smoothCam.distance = targetCam.distance;
      smoothCam.targetX = targetCam.targetX;
      smoothCam.targetY = targetCam.targetY;
      smoothCam.targetZ = targetCam.targetZ;
      smoothCam.offsetPx = targetCam.offsetPx;
    }

    function syncRoundedParamsFromTarget(shouldPersist) {
      var next = clampParams({
        height: params.height,
        contours: params.contours,
        pitch: targetCam.pitch,
        yaw: targetCam.yaw,
      }, tier);
      if (next.pitch !== params.pitch || next.yaw !== params.yaw) {
        params = next;
        if (shouldPersist) saveParamsIfEnabled(params);
        onStateChange(getPublicState());
      }
    }

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
            dprCap: (tier === 'lite' || degradedPerf) ? 1.5 : 2,
            softPeakHighlight: !!opts.softPeakHighlight,
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
        var gridN = tier === 'lite' ? 112 : (nodes.length <= 160 ? 160 : 200);
        var isLiveMotion = liveSimSync || isDragOrRebound;
        var shouldMorph = pendingMorphOnBuild && !!terrainField && !isLiveMotion && tier !== 'reduced';
        pendingMorphOnBuild = false;
        prevTerrainField = shouldMorph ? terrainField : null;
        terrainField = computeTerrainField(nodes, degreeInfo, {
          gridSize: gridN,
          extent: (activeDragBounds && terrainField) ? terrainField.extent : undefined,
          activeIds: data.activeIds || null,
          fixedBounds: activeDragBounds,
          skipOverlapResolve: liveSimSync || (isLiveMotion && nodes.length > 160),
          overlapPasses: isLiveMotion ? 1 : (tier === 'lite' ? 4 : 6),
          pinnedId: activeDragId,
        });
        renderer.uploadTerrainMesh(terrainField, prevTerrainField);
        if (shouldMorph && typeof requestAnimationFrame === 'function') {
          morphAnim = { startTime: nowMs(), duration: 520 };
        } else {
          morphAnim = null;
        }
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
          isNodeNew: isNodeNew,
          searchMatchIds: getSearchMatchIds(),
          communityLabels: getCommunityLabels(),
          activeCommunityId: getActiveCommunityId(),
          hoveredCommunityId: hoveredCommunityId,
          showEdges: isEdgesVisible(),
          isLite: tier === 'lite',
        }, heightAmp, isDark());
        batchesDirty = false;
      }
    }

    var entryAnim = null;
    var reboundAnim = null;

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

    function startCameraTransition(fromPitch, fromYaw, fromDist, fromHeightScale, durationMs, sweepGlint, forceMotion, fromTargetX, fromTargetZ, fromTargetY) {
      if (tier === 'reduced' || typeof requestAnimationFrame !== 'function' || !durationMs || durationMs <= 0) {
        entryAnim = null;
        snapSmoothToTarget();
        renderNow();
        return;
      }
      entryAnim = {
        startTime: null,
        duration: durationMs,
        fromPitch: fromPitch,
        fromYaw: fromYaw,
        fromDist: fromDist,
        fromTargetX: fromTargetX != null ? fromTargetX : smoothCam.targetX,
        fromTargetY: fromTargetY != null ? fromTargetY : smoothCam.targetY,
        fromTargetZ: fromTargetZ != null ? fromTargetZ : smoothCam.targetZ,
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

    var tabBackgrounded = false;

    function isTabHidden() {
      return tabBackgrounded;
    }

    function isThemeSwitching() {
      return typeof document !== 'undefined' && document.documentElement &&
        document.documentElement.classList && document.documentElement.classList.contains('is-theme-switching');
    }

    function shouldIdleAutoRotate(now) {
      if (tier !== 'full' || degradedPerf || !enabled || isTabHidden() || isThemeSwitching()) return false;
      if (draggingSlider || dragState || pinchState || entryAnim || morphAnim || reboundAnim) return false;
      if (Math.abs(targetCam.offsetPx) > 1 || Math.abs(smoothCam.offsetPx) > 1) return false;
      return (now - lastUserInputAt) > 7000;
    }

    function renderNow() {
      renderRaf = 0;
      if (!enabled || tier === 'unsupported' || isTabHidden()) return;
      var r = ensureStageCanvases();
      if (!r) return;
      var now = nowMs();
      var switchingTheme = isThemeSwitching();
      if (!startEpochMs) startEpochMs = now;
      var dt = lastFrameTime ? Math.max(0.001, Math.min(0.08, (now - lastFrameTime) / 1000)) : 0.016;
      lastFrameTime = now;

      // 慢 GPU 帧率自适应退档保护
      if (tier === 'full' && !degradedPerf && !entryAnim && !reboundAnim && !switchingTheme) {
        if (dt > 0.045) slowFrameStreak += 1;
        else slowFrameStreak = Math.max(0, slowFrameStreak - 2);
        if (slowFrameStreak >= 90) degradedPerf = true;
      }

      var vp = getViewportSize();
      var baseDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      r.resize(vp.width || 1200, vp.height || 800, degradedPerf ? 1 : baseDpr);

      var keepGoing = false;
      if (reboundAnim && stepReboundAnim()) {
        keepGoing = true;
      }
      rebuildFieldAndBatches();

      // 地表探针阻尼
      var probeBlend = tier === 'reduced' ? 1 : Math.min(1, dt * 10);
      probeState.strength += (probeState.targetStrength - probeState.strength) * probeBlend;
      if (Math.abs(probeState.targetStrength - probeState.strength) > 0.01) keepGoing = true;

      // 山体平滑沉降/隆起形变动画
      var morphT = 1;
      if (morphAnim) {
        var mElapsed = Math.max(0, now - morphAnim.startTime);
        var mRaw = Math.min(1, mElapsed / morphAnim.duration);
        morphT = 1 - Math.pow(1 - mRaw, 3);
        if (mRaw < 1) keepGoing = true;
        else morphAnim = null;
      }

      var idleOrbiting = shouldIdleAutoRotate(now);
      if (idleOrbiting) {
        targetCam.yaw = wrapYawDeg(targetCam.yaw + dt * 2.1);
        smoothCam.yaw = targetCam.yaw;
      }

      var renderParams;
      var renderCam;
      var heightScale = 1;
      var forceSettle = 1;
      var glintPhase = -1;
      var labelFade = 1;

      if (entryAnim) {
        if (entryAnim.startTime == null) entryAnim.startTime = now;
        var elapsed = Math.max(0, now - entryAnim.startTime);
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
        var yawDiff = wrapYawDeg(targetCam.yaw - entryAnim.fromYaw);
        smoothCam.pitch = entryAnim.fromPitch + (targetCam.pitch - entryAnim.fromPitch) * camEase;
        smoothCam.yaw = wrapYawDeg(entryAnim.fromYaw + yawDiff * camEase);
        smoothCam.distance = entryAnim.fromDist + (targetCam.distance - entryAnim.fromDist) * camEase;
        smoothCam.targetX = entryAnim.fromTargetX + (targetCam.targetX - entryAnim.fromTargetX) * camEase;
        smoothCam.targetY = entryAnim.fromTargetY + (targetCam.targetY - entryAnim.fromTargetY) * camEase;
        smoothCam.targetZ = entryAnim.fromTargetZ + (targetCam.targetZ - entryAnim.fromTargetZ) * camEase;
        var dampOff = 1 - Math.exp(-dt * 12);
        smoothCam.offsetPx += (targetCam.offsetPx - smoothCam.offsetPx) * dampOff;

        heightScale = entryAnim.fromHeightScale + (1 - entryAnim.fromHeightScale) * riseEase;
        glintPhase = entryAnim.sweepGlint ? (rawT * 1.32 - 0.12) : -1;
        labelFade = (entryAnim.fromHeightScale < 0.5 || entryAnim.forceMotion)
          ? Math.max(0, Math.min(1, (rawT - 0.30) / 0.50))
          : 1;
        if (rawT < 1) {
          keepGoing = true;
        } else {
          entryAnim = null;
          snapSmoothToTarget();
          heightScale = 1;
          forceSettle = 1;
          glintPhase = -1;
          labelFade = 1;
        }
      } else {
        var dampK = (tier === 'reduced' || draggingSlider) ? 1 : (1 - Math.exp(-dt * 12));
        smoothCam.pitch += (targetCam.pitch - smoothCam.pitch) * dampK;
        smoothCam.yaw = wrapYawDeg(smoothCam.yaw + wrapYawDeg(targetCam.yaw - smoothCam.yaw) * dampK);
        smoothCam.distance += (targetCam.distance - smoothCam.distance) * dampK;
        smoothCam.targetX += (targetCam.targetX - smoothCam.targetX) * dampK;
        smoothCam.targetY += (targetCam.targetY - smoothCam.targetY) * dampK;
        smoothCam.targetZ += (targetCam.targetZ - smoothCam.targetZ) * dampK;
        smoothCam.offsetPx += (targetCam.offsetPx - smoothCam.offsetPx) * dampK;
        var camDelta = Math.abs(targetCam.pitch - smoothCam.pitch) +
          Math.abs(wrapYawDeg(targetCam.yaw - smoothCam.yaw)) +
          Math.abs(targetCam.distance - smoothCam.distance) * 30 +
          Math.abs(targetCam.targetX - smoothCam.targetX) * 40 +
          Math.abs(targetCam.targetY - smoothCam.targetY) * 40 +
          Math.abs(targetCam.targetZ - smoothCam.targetZ) * 40 +
          Math.abs(targetCam.offsetPx - smoothCam.offsetPx) * 0.2;
        if (camDelta > 0.04) {
          keepGoing = true;
        } else {
          snapSmoothToTarget();
        }
      }

      renderParams = {
        height: params.height,
        contours: params.contours,
        pitch: smoothCam.pitch,
        yaw: smoothCam.yaw,
      };
      renderCam = {
        distance: smoothCam.distance,
        targetX: smoothCam.targetX,
        targetY: smoothCam.targetY,
        targetZ: smoothCam.targetZ,
        offsetPx: smoothCam.offsetPx,
      };

      var ambientEnabled = tier === 'full' && !degradedPerf && !draggingSlider && !switchingTheme;
      var postFxEnabled = tier === 'full' && !degradedPerf && !draggingSlider && r.isPostAvailable();
      var timeSec = (now - startEpochMs) / 1000;
      if (ambientEnabled || idleOrbiting) keepGoing = true;

      r.render(renderParams, renderCam, {
        isDark: isDark(),
        bgColor: getBgColor(),
        isLite: tier === 'lite',
        isReduced: tier === 'reduced',
        showLabels: isLabelsVisible(),
        getLabelText: getLabelText,
        heightScale: heightScale,
        forceSettle: forceSettle,
        morph: morphT,
        glintPhase: glintPhase,
        radarEnabled: ambientEnabled,
        ambientEnabled: ambientEnabled,
        postFxEnabled: postFxEnabled,
        probe: probeState,
        timeSec: timeSec,
        labelFade: labelFade,
        hasHoverHighlight: !!hoveredNode,
        communityLabels: getCommunityLabels(),
        routeBadges: getRouteBadges(),
        activeCommunityId: getActiveCommunityId(),
        hoveredCommunityId: hoveredCommunityId,
      });
      if (keepGoing && typeof requestAnimationFrame === 'function') {
        renderRaf = requestAnimationFrame(renderNow);
      }
    }

    function scheduleRender() {
      if (!enabled || renderRaf || isTabHidden()) return;
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

    /* ── 鼠标 / 触控交互（轨道旋转、平移、光标指向缩放、节点拖拽、社区标签点击、地表探针）── */
    var dragState = null;
    var activePointers = new Map();
    var pinchState = null;
    var gestureSuppressedTap = false;
    var cachedStageRect = null;
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
      var commHit = renderer.pickCommunityAtScreen ? renderer.pickCommunityAtScreen(pt[0], pt[1]) : null;
      if (commHit) {
        if (hoveredCommunityId !== commHit.id || hoveredNode) {
          hoveredCommunityId = commHit.id;
          hoveredNode = null;
          probeState.targetStrength = 0;
          if (stageEl) stageEl.style.cursor = 'pointer';
          batchesDirty = true;
          scheduleRender();
          onNodeHover(null, ev);
        }
        return;
      }
      var commChanged = false;
      if (hoveredCommunityId) {
        hoveredCommunityId = null;
        if (stageEl) stageEl.style.cursor = '';
        batchesDirty = true;
        commChanged = true;
        scheduleRender();
      }
      var hit = renderer.pickNodeAtScreen(pt[0], pt[1]);
      var prevId = hoveredNode ? hoveredNode.id : null;
      var nextId = hit ? hit.id : null;
      if (prevId !== nextId) {
        hoveredNode = hit || null;
        if (stageEl) stageEl.style.cursor = (hit && canDragNodesNow()) ? 'grab' : '';
        batchesDirty = true;
        scheduleRender();
      }
      if (!hit && renderer.raycastTerrain) {
        var heightAmp = (params.height / 100) * 0.46;
        var terrainHit = renderer.raycastTerrain(pt[0], pt[1], heightAmp);
        if (terrainHit) {
          probeState.wx = terrainHit[0];
          probeState.wz = terrainHit[2];
          probeState.targetStrength = 1;
          scheduleRender();
        } else if (probeState.targetStrength > 0) {
          probeState.targetStrength = 0;
          scheduleRender();
        }
      } else if (probeState.targetStrength > 0) {
        probeState.targetStrength = 0;
        scheduleRender();
      }
      if (hit || prevId !== null || commChanged) onNodeHover(hit || null, ev);
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
      var yawRad = (targetCam.yaw * Math.PI) / 180;
      var speed = targetCam.distance * 0.0022;
      var rx = Math.cos(yawRad);
      var rz = -Math.sin(yawRad);
      var fx = Math.sin(yawRad);
      var fz = Math.cos(yawRad);
      var panMax = Math.max(1.8, (terrainField ? terrainField.extent : 1.45) * 1.25);
      targetCam.targetX = Math.max(-panMax, Math.min(panMax, targetCam.targetX - dx * rx * speed + dy * fx * speed));
      targetCam.targetZ = Math.max(-panMax, Math.min(panMax, targetCam.targetZ - dx * rz * speed + dy * fz * speed));
    }

    function onPointerDown(ev) {
      if (!enabled || !stageEl) return;
      cancelPendingHover();
      cachedStageRect = null;
      markUserInput();
      if (entryAnim) cancelEntryAnim();
      var isTouch = ev.pointerType === 'touch';
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
      var pt = getStageLocalPoint(ev);
      if (ev.button === 0 || isTouch) {
        var commHit = renderer && renderer.pickCommunityAtScreen ? renderer.pickCommunityAtScreen(pt[0], pt[1]) : null;
        if (commHit) {
          dragState = {
            id: ev.pointerId,
            startX: ev.clientX,
            startY: ev.clientY,
            lastX: ev.clientX,
            lastY: ev.clientY,
            moved: false,
            isTouch: isTouch,
            mode: 'community',
            communityId: commHit.id,
          };
          return;
        }
      }
      var isPan = ev.button === 2 || ev.button === 1 || ev.shiftKey;
      if (!isPan && (ev.button === 0 || isTouch) && canDragNodesNow() && renderer && terrainField) {
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
        markUserInput();
        var pair = getFirstTwoPointers();
        var nextDist = Math.max(12, Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y));
        var nextMidX = (pair[0].x + pair[1].x) * 0.5;
        var nextMidY = (pair[0].y + pair[1].y) * 0.5;
        var zoomRatio = pinchState.dist / nextDist;
        targetCam.distance = clampFloat(targetCam.distance * zoomRatio, targetCam.distance, 0.58, 5.5);
        panCameraByDelta(nextMidX - pinchState.midX, nextMidY - pinchState.midY);
        pinchState.dist = nextDist;
        pinchState.midX = nextMidX;
        pinchState.midY = nextMidY;
        scheduleRender();
        return;
      }
      if (dragState && dragState.id === ev.pointerId) {
        markUserInput();
        var dx = ev.clientX - dragState.lastX;
        var dy = ev.clientY - dragState.lastY;
        dragState.lastX = ev.clientX;
        dragState.lastY = ev.clientY;
        var moveThresh = dragState.isTouch ? 8 : 4;
        if (Math.hypot(ev.clientX - dragState.startX, ev.clientY - dragState.startY) > moveThresh) {
          dragState.moved = true;
        }
        if (dragState.mode === 'community') {
          if (dragState.moved) dragState.mode = 'orbit';
          else return;
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
              probeState.targetStrength = 0;
              if (stageEl) stageEl.style.cursor = 'grabbing';
              onNodeHover(null, ev);
              onNodeDragStart(dragState.node, ev);
            }
            var dragPt = getStageLocalPoint(ev);
            var hitWorld = renderer.unprojectToPlaneY(dragPt[0], dragPt[1], dragState.planeY);
            if (hitWorld && activeDragBounds) {
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
          if (hoveredNode || hoveredCommunityId) {
            hoveredNode = null;
            hoveredCommunityId = null;
            if (stageEl) stageEl.style.cursor = '';
            batchesDirty = true;
            onNodeHover(null, ev);
          }
          probeState.targetStrength = 0;
          targetCam.yaw = wrapYawDeg(targetCam.yaw - dx * 0.34);
          targetCam.pitch = clampFloat(targetCam.pitch + dy * 0.26, targetCam.pitch, BOUNDS.pitch.min, BOUNDS.pitch.max);
          syncRoundedParamsFromTarget(false);
          scheduleSaveParams(params);
          scheduleRender();
          return;
        }
        if (dragState.mode === 'pan' && dragState.moved) {
          if (hoveredNode || hoveredCommunityId) {
            hoveredNode = null;
            hoveredCommunityId = null;
            if (stageEl) stageEl.style.cursor = '';
            batchesDirty = true;
            onNodeHover(null, ev);
          }
          probeState.targetStrength = 0;
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
      if (wasDrag.mode === 'community' && !wasDrag.moved) {
        markUserInput();
        onCommunityClick(wasDrag.communityId, ev);
        return;
      }
      if (wasDrag.mode === 'node' && wasDrag.dragStarted) {
        markUserInput();
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
      markUserInput();
      var pt = getStageLocalPoint(ev);
      var commTap = renderer.pickCommunityAtScreen ? renderer.pickCommunityAtScreen(pt[0], pt[1]) : null;
      if (commTap) {
        onCommunityClick(commTap.id, ev);
        return;
      }
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
      var needRender = false;
      if (hoveredNode) {
        hoveredNode = null;
        batchesDirty = true;
        needRender = true;
      }
      if (hoveredCommunityId) {
        hoveredCommunityId = null;
        batchesDirty = true;
        needRender = true;
      }
      if (probeState.targetStrength > 0) {
        probeState.targetStrength = 0;
        needRender = true;
      }
      if (needRender) scheduleRender();
      onNodeHover(null, ev);
    }

    function onWheel(ev) {
      if (!enabled) return;
      markUserInput();
      if (entryAnim) cancelEntryAnim();
      ev.preventDefault();
      var factor = ev.deltaY > 0 ? 1.09 : 0.91;
      var oldDist = targetCam.distance;
      var nextDist = clampFloat(oldDist * factor, oldDist, 0.58, 5.5);
      if (renderer && renderer.raycastTerrain) {
        var pt = getStageLocalPoint(ev);
        var heightAmp = (params.height / 100) * 0.46;
        var hit = renderer.raycastTerrain(pt[0], pt[1], heightAmp);
        if (hit) {
          var zoomFrac = 1 - (nextDist / oldDist);
          var panMax = Math.max(1.8, (terrainField ? terrainField.extent : 1.45) * 1.25);
          targetCam.targetX = Math.max(-panMax, Math.min(panMax, targetCam.targetX + (hit[0] - targetCam.targetX) * zoomFrac * 0.72));
          targetCam.targetZ = Math.max(-panMax, Math.min(panMax, targetCam.targetZ + (hit[2] - targetCam.targetZ) * zoomFrac * 0.72));
        }
      }
      targetCam.distance = nextDist;
      scheduleRender();
    }

    function onContextMenu(ev) {
      if (enabled) ev.preventDefault();
    }

    function onVisibilityChange() {
      tabBackgrounded = !!(typeof document !== 'undefined' && document.hidden);
      if (!enabled) return;
      if (tabBackgrounded) {
        if (renderRaf && typeof cancelAnimationFrame === 'function') {
          cancelAnimationFrame(renderRaf);
          renderRaf = 0;
        }
      } else {
        lastFrameTime = nowMs();
        scheduleRender();
      }
    }

    function bindCanvasEvents() {
      if (stageEl && typeof stageEl.addEventListener === 'function') {
        stageEl.addEventListener('pointerdown', onPointerDown);
        stageEl.addEventListener('pointermove', onPointerMove);
        stageEl.addEventListener('pointerup', onPointerUp);
        stageEl.addEventListener('pointercancel', onPointerUp);
        stageEl.addEventListener('pointerleave', onPointerLeave);
        stageEl.addEventListener('wheel', onWheel, { passive: false });
        stageEl.addEventListener('contextmenu', onContextMenu);
      }
      if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
        document.addEventListener('visibilitychange', onVisibilityChange);
      }
      if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener('resize', invalidateStageRect, { passive: true });
        window.addEventListener('scroll', invalidateStageRect, { passive: true });
      }
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
          distance: Number(targetCam.distance.toFixed(3)),
          targetX: Number(targetCam.targetX.toFixed(3)),
          targetZ: Number(targetCam.targetZ.toFixed(3)),
          offsetPx: Number(targetCam.offsetPx.toFixed(1)),
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
        markUserInput();
        params = clampParams(params, tier);
        targetCam.pitch = params.pitch;
        targetCam.yaw = params.yaw;
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
          snapSmoothToTarget();
          renderNow();
        } else {
          startCameraTransition(78, 0, targetCam.distance * 1.14, 0.04, tier === 'lite' ? 950 : 1450, true, true);
        }
      } else {
        cancelPendingHover();
        flushSaveParams();
        invalidateStageRect();
        cancelEntryAnim();
        cancelReboundAnim(true);
        morphAnim = null;
        hoveredNode = null;
        hoveredCommunityId = null;
        dragState = null;
        activePointers.clear();
        pinchState = null;
        gestureSuppressedTap = false;
        activeDragBounds = null;
        probeState.targetStrength = 0;
        probeState.strength = 0;
        if (stageEl) stageEl.style.cursor = '';
        clearStageClasses();
      }
      onStateChange(getPublicState());
      return true;
    }

    function setParam(key, value) {
      if (!Object.prototype.hasOwnProperty.call(BOUNDS, key)) return;
      markUserInput();
      if (entryAnim) cancelEntryAnim();
      var next = Object.assign({}, params);
      next[key] = value;
      params = clampParams(next, tier);
      if (key === 'pitch') targetCam.pitch = params.pitch;
      if (key === 'yaw') targetCam.yaw = params.yaw;
      saveParamsIfEnabled(params);
      if (key === 'height') batchesDirty = true;
      if (enabled) scheduleRender();
      onStateChange(getPublicState());
    }

    function resetDefaults() {
      markUserInput();
      var prevPitch = smoothCam.pitch;
      var prevYaw = smoothCam.yaw;
      var prevDist = smoothCam.distance;
      var prevTx = smoothCam.targetX;
      var prevTy = smoothCam.targetY;
      var prevTz = smoothCam.targetZ;
      params = resolveDefaultForTier(tier);
      targetCam.pitch = params.pitch;
      targetCam.yaw = params.yaw;
      targetCam.distance = defaultDist;
      targetCam.targetX = 0;
      targetCam.targetY = 0.08;
      targetCam.targetZ = 0;
      saveParamsIfEnabled(params);
      batchesDirty = true;
      if (enabled) startCameraTransition(prevPitch, prevYaw, prevDist, 1, 480, true, false, prevTx, prevTz, prevTy);
      onStateChange(getPublicState());
    }

    function setTopView() {
      markUserInput();
      var prevPitch = smoothCam.pitch;
      var prevYaw = smoothCam.yaw;
      var prevDist = smoothCam.distance;
      var prevTx = smoothCam.targetX;
      var prevTy = smoothCam.targetY;
      var prevTz = smoothCam.targetZ;
      params = clampParams({
        height: params.height,
        contours: params.contours,
        pitch: TOP_VIEW_PARAMS.pitch,
        yaw: TOP_VIEW_PARAMS.yaw,
      }, tier);
      targetCam.pitch = params.pitch;
      targetCam.yaw = params.yaw;
      targetCam.targetX = 0;
      targetCam.targetY = 0.08;
      targetCam.targetZ = 0;
      saveParamsIfEnabled(params);
      if (enabled) startCameraTransition(prevPitch, prevYaw, prevDist, 1, 460, true, false, prevTx, prevTz, prevTy);
      onStateChange(getPublicState());
    }

    function syncData(rebuildField, syncOpts) {
      liveSimSync = !!(syncOpts && syncOpts.live);
      if (rebuildField !== false) {
        terrainDirty = true;
        if (syncOpts && syncOpts.morph) pendingMorphOnBuild = true;
      }
      batchesDirty = true;
      if (enabled) scheduleRender();
    }

    function refreshTheme() {
      batchesDirty = true;
      if (enabled) renderNow();
    }

    function fitToView() {
      markUserInput();
      var prevDist = smoothCam.distance;
      var prevTx = smoothCam.targetX;
      var prevTy = smoothCam.targetY;
      var prevTz = smoothCam.targetZ;
      targetCam.distance = defaultDist;
      targetCam.targetX = 0;
      targetCam.targetY = 0.08;
      targetCam.targetZ = 0;
      if (enabled) {
        startCameraTransition(smoothCam.pitch, smoothCam.yaw, prevDist, 1, 420, false, false, prevTx, prevTz, prevTy);
      }
    }

    function focusNode(nodeId) {
      markUserInput();
      routeFocusNodeId = nodeId || null;
      batchesDirty = true;
      if (!enabled) return;
      if (!routeFocusNodeId) {
        targetCam.targetY = 0.08;
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
      var extLim = (terrainField.extent || 1.35) * 0.85;
      var hAmpFocus = (params.height / 100) * 0.46;
      var worldFocusY = (m.elevH || m.surfaceH || 0.1) * hAmpFocus;
      targetCam.targetX = Math.max(-extLim, Math.min(extLim, m.wx));
      targetCam.targetY = Math.max(0.08, Math.min(0.42, worldFocusY * 0.68));
      targetCam.targetZ = Math.max(-extLim, Math.min(extLim, m.wz));
      targetCam.distance = Math.min(targetCam.distance, 1.08 + (m.degNorm || 0) * 0.22);
      if (tier === 'reduced' || typeof requestAnimationFrame !== 'function') {
        snapSmoothToTarget();
        renderNow();
      } else {
        scheduleRender();
      }
    }

    function frameNodes(nodeIdsOrSet) {
      markUserInput();
      if (!nodeIdsOrSet) return;
      var ids = Array.isArray(nodeIdsOrSet) ? nodeIdsOrSet : Array.from(nodeIdsOrSet);
      if (!ids.length) return;
      if (enabled) rebuildFieldAndBatches();
      if (!terrainField || !terrainField.nodeMeta) return;
      var x0 = Infinity;
      var x1 = -Infinity;
      var z0 = Infinity;
      var z1 = -Infinity;
      var elevSum = 0;
      var count = 0;
      for (var i = 0; i < ids.length; i++) {
        var m = terrainField.nodeMeta[ids[i]];
        if (!m) continue;
        if (m.wx < x0) x0 = m.wx;
        if (m.wx > x1) x1 = m.wx;
        if (m.wz < z0) z0 = m.wz;
        if (m.wz > z1) z1 = m.wz;
        elevSum += (m.surfaceH || 0);
        count += 1;
      }
      if (!count) return;
      var cx = (x0 + x1) * 0.5;
      var cz = (z0 + z1) * 0.5;
      var span = Math.max(0.24, Math.hypot(x1 - x0, z1 - z0));
      var extLim = (terrainField.extent || 1.35) * 0.85;
      var hAmpFrame = (params.height / 100) * 0.46;
      targetCam.targetX = Math.max(-extLim, Math.min(extLim, cx));
      targetCam.targetY = Math.max(0.08, Math.min(0.28, (elevSum / count) * hAmpFrame * 0.55));
      targetCam.targetZ = Math.max(-extLim, Math.min(extLim, cz));
      targetCam.distance = clampFloat(span * 1.45 + 0.58, defaultDist, 0.72, Math.max(defaultDist, 2.6));
      if (enabled) {
        if (tier === 'reduced' || typeof requestAnimationFrame !== 'function') {
          snapSmoothToTarget();
          renderNow();
        } else {
          scheduleRender();
        }
      }
    }

    function setSidebarOffset(px) {
      var nextOff = clampFloat(Number(px) || 0, 0, -400, 400);
      if (Math.abs(targetCam.offsetPx - nextOff) < 0.5) return;
      targetCam.offsetPx = nextOff;
      if (enabled) {
        if (tier === 'reduced' || typeof requestAnimationFrame !== 'function') {
          smoothCam.offsetPx = nextOff;
          renderNow();
        } else {
          scheduleRender();
        }
      }
      onStateChange(getPublicState());
    }

    function setActiveCommunity(communityId) {
      activeCommunityOverride = communityId || null;
      batchesDirty = true;
      if (enabled) scheduleRender();
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
      morphAnim = null;
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
      if (typeof document !== 'undefined' && typeof document.removeEventListener === 'function') {
        document.removeEventListener('visibilitychange', onVisibilityChange);
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
      setDragging: function (active) { draggingSlider = !!active; markUserInput(); },
      resetDefaults: resetDefaults,
      setFlatView: setTopView,
      setTopView: setTopView,
      syncData: syncData,
      refreshTheme: refreshTheme,
      replayForceBloom: function () {
        if (!enabled || tier === 'unsupported') return;
        terrainDirty = true;
        batchesDirty = true;
        startCameraTransition(smoothCam.pitch, smoothCam.yaw, smoothCam.distance, 0.12, tier === 'lite' ? 850 : 1200, true, true);
      },
      fitToView: fitToView,
      focusNode: focusNode,
      frameNodes: frameNodes,
      setSidebarOffset: setSidebarOffset,
      setActiveCommunity: setActiveCommunity,
      syncViewport: syncViewport,
      refreshStageRect: function () { invalidateStageRect(); },
      getScreenTargets: function (limit) {
        return renderer && typeof renderer.getTopScreenTargets === 'function'
          ? renderer.getTopScreenTargets(limit)
          : [];
      },
      getCommunityHitboxes: function () {
        return renderer && typeof renderer.getCommunityHitboxes === 'function'
          ? renderer.getCommunityHitboxes()
          : [];
      },
      getDebugState: function () {
        var now = nowMs();
        return {
          supported: tier !== 'unsupported',
          tier: tier,
          enabled: enabled,
          isWebGL2: renderer ? renderer.isWebGL2 : false,
          postFxEnabled: tier === 'full' && !degradedPerf && !!(renderer && renderer.isPostAvailable()),
          degradedPerf: degradedPerf,
          params: Object.assign({}, params),
          camera: {
            distance: Number(targetCam.distance.toFixed(3)),
            targetX: Number(targetCam.targetX.toFixed(3)),
            targetZ: Number(targetCam.targetZ.toFixed(3)),
            offsetPx: Number(targetCam.offsetPx.toFixed(1)),
          },
          fieldStats: renderer ? renderer.getFieldStats() : null,
          hoveredNodeId: hoveredNode ? hoveredNode.id : null,
          hoveredCommunityId: hoveredCommunityId,
          routeFocusNodeId: routeFocusNodeId,
          draggingNodeId: (dragState && dragState.mode === 'node' && dragState.dragStarted && dragState.node) ? dragState.node.id : null,
          rebounding: !!reboundAnim,
          morphing: !!morphAnim,
          probeActive: probeState.strength > 0.05,
          idleAutoRotate: shouldIdleAutoRotate(now),
          animating: !!entryAnim || !!reboundAnim || !!morphAnim || !!draggingSlider || !!renderRaf,
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
    buildCommunityHullSegments: buildCommunityHullSegments,
    createController: createController,
  };
});
