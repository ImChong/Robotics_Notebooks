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

  var TOP_VIEW_PARAMS = { pitch: 76, yaw: 0 };

  var TIER_BADGE_TEXT = {
    full: '',
    lite: '轻量模式',
    reduced: '减弱动效',
    unsupported: '降级 2D',
  };

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

  /* ── 颜色解析 ── */

  function parseCssColor(str, fallback) {
    var fb = fallback || [0.36, 0.61, 0.96, 1];
    if (!str || typeof str !== 'string') return fb.slice();
    var s = str.trim();
    if (s.charAt(0) === '#') {
      var hex = s.slice(1);
      if (hex.length === 3) {
        hex = hex.charAt(0) + hex.charAt(0) + hex.charAt(1) + hex.charAt(1) + hex.charAt(2) + hex.charAt(2);
      }
      if (hex.length >= 6) {
        var n = parseInt(hex.slice(0, 6), 16);
        if (Number.isFinite(n)) {
          return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
        }
      }
      return fb.slice();
    }
    var m = s.match(/^rgba?\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*[,\s]\s*([\d.]+)(?:\s*[,/]\s*([\d.]+))?\s*\)/i);
    if (m) {
      return [
        Math.min(1, Math.max(0, Number(m[1]) / 255)),
        Math.min(1, Math.max(0, Number(m[2]) / 255)),
        Math.min(1, Math.max(0, Number(m[3]) / 255)),
        m[4] !== undefined ? Math.min(1, Math.max(0, Number(m[4]))) : 1,
      ];
    }
    return fb.slice();
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
      for (i = 0; i < list.length; i++) {
        var n2 = list[i];
        if (!n2 || n2.x == null || n2.y == null) continue;
        activeList.push(n2);
        xs.push(n2.x);
        ys.push(n2.y);
      }
    }

    var xB = quantileBounds(xs, 0.02, 0.98);
    var yB = quantileBounds(ys, 0.02, 0.98);
    var cx = (xB[0] + xB[1]) / 2;
    var cy = (yB[0] + yB[1]) / 2;
    var halfSpan = Math.max(40, Math.max(xB[1] - xB[0], yB[1] - yB[0]) * 0.55);
    var coreSpan = 0.78;

    var broadBuf = new Float32Array(G * G);
    var peakBuf = new Float32Array(G * G);
    var tmpBuf = new Float32Array(G * G);
    var cellScale = (G - 1) / (2 * extent);

    var nodeMeta = Object.create(null);
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
      if (!node || !node.id || node.x == null || node.y == null) continue;
      var rx = (node.x - cx) / halfSpan;
      var rz = (node.y - cy) / halfSpan;
      // Softly compress far outliers so mountains stay well within the misty horizon
      var wx = (Math.abs(rx) > 1 ? Math.sign(rx) * (1 + 0.16 * Math.tanh(Math.abs(rx) - 1)) : rx) * coreSpan;
      var wz = (Math.abs(rz) > 1 ? Math.sign(rz) * (1 + 0.16 * Math.tanh(Math.abs(rz) - 1)) : rz) * coreSpan;
      var deg = degCounts[node.id] || 0;
      var degNorm = Math.pow(Math.log(1 + deg) / logDen, 1.08);
      var isIncluded = !activeSet || activeSet.has(node.id);
      var contrib = isIncluded ? degNorm : degNorm * 0.06;

      nodeMeta[node.id] = {
        wx: wx,
        wz: wz,
        deg: deg,
        degNorm: degNorm,
        active: isIncluded,
      };

      if (!(contrib > 0.001)) continue;
      var gxF = (wx + extent) * cellScale;
      var gzF = (wz + extent) * cellScale;
      var gx0 = Math.max(0, Math.min(G - 2, Math.floor(gxF)));
      var gz0 = Math.max(0, Math.min(G - 2, Math.floor(gzF)));
      var tx = Math.max(0, Math.min(1, gxF - gx0));
      var tz = Math.max(0, Math.min(1, gzF - gz0));

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
        var envMask = 1 - smoothstep(0.76, extent * 0.88, radial);
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
        var maskEnd = 1 - smoothstep(0.78, extent * 0.90, Math.hypot(wxEnv, wzEnv));
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
      center: { x: cx, y: cy, halfSpan: halfSpan },
    };

    var ids = Object.keys(nodeMeta);
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
    var proj = perspectiveMat4((42 * Math.PI) / 180, aspect, 0.04, 24);
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

    var terrainVS = headV +
      attr + 'vec2 aXZ;\n' +
      attr + 'float aH;\n' +
      attr + 'vec2 aGrad;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uHeightAmp;\n' +
      varyV + 'vec3 vWorld;\n' +
      varyV + 'float vH;\n' +
      varyV + 'vec2 vGrad;\n' +
      'void main() {\n' +
      '  float y = aH * uHeightAmp;\n' +
      '  vWorld = vec3(aXZ.x, y, aXZ.y);\n' +
      '  vH = aH;\n' +
      '  vGrad = aGrad * uHeightAmp;\n' +
      '  gl_Position = uViewProj * vec4(vWorld, 1.0);\n' +
      '}\n';

    var terrainFS = headF + outDecl + derivFn + '\n' +
      varyF + 'vec3 vWorld;\n' +
      varyF + 'float vH;\n' +
      varyF + 'vec2 vGrad;\n' +
      'uniform vec3 uEye;\n' +
      'uniform vec3 uBg;\n' +
      'uniform float uIsDark;\n' +
      'uniform float uContours;\n' +
      'uniform float uExtent;\n' +
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
      '  float diff = clamp(dot(N, L), 0.0, 1.0);\n' +
      '  float rim = pow(1.0 - max(dot(N, V), 0.0), 2.4);\n' +
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
      '  float gridLine = max(gx, gz) * (1.0 - smoothstep(0.02, 0.34, vH));\n' +
      '  vec3 gridCol = (uIsDark > 0.5) ? vec3(0.38, 0.54, 0.78) : vec3(0.28, 0.44, 0.68);\n' +
      '  lit = mix(lit, gridCol, gridLine * (uIsDark > 0.5 ? 0.16 : 0.14));\n' +
      '\n' +
      '  float peakGlow = smoothstep(0.76, 1.0, vH);\n' +
      '  if (uIsDark > 0.5) lit += vec3(0.28, 0.13, 0.04) * peakGlow;\n' +
      '\n' +
      '  float radial = length(vWorld.xz) / uExtent;\n' +
      '  float radialFade = 1.0 - smoothstep(0.46, 0.92, radial);\n' +
      '  float viewDist = length(uEye - vWorld);\n' +
      '  float viewFog = exp(-max(0.0, viewDist - 2.6) * 0.35);\n' +
      '  float vis = clamp(radialFade * viewFog, 0.0, 1.0);\n' +
      '  float valleySoft = mix(0.48, 1.0, smoothstep(0.0, 0.09, vH) + gridLine * 0.55 + cLine * 0.65);\n' +
      '  ' + outAssign + ' = vec4(mix(uBg, lit, 0.92), vis * valleySoft);\n' +
      '}\n';

    var lineVS = headV +
      attr + 'vec3 aPos;\n' +
      attr + 'vec4 aColor;\n' +
      'uniform mat4 uViewProj;\n' +
      varyV + 'vec4 vColor;\n' +
      varyV + 'float vRadial;\n' +
      'uniform float uExtent;\n' +
      'void main() {\n' +
      '  vColor = aColor;\n' +
      '  vRadial = length(aPos.xz) / uExtent;\n' +
      '  gl_Position = uViewProj * vec4(aPos, 1.0);\n' +
      '}\n';

    var lineFS = headF + outDecl +
      varyF + 'vec4 vColor;\n' +
      varyF + 'float vRadial;\n' +
      'void main() {\n' +
      '  float fade = 1.0 - smoothstep(0.60, 0.96, vRadial);\n' +
      '  ' + outAssign + ' = vec4(vColor.rgb, vColor.a * fade);\n' +
      '}\n';

    var solidVS = headV +
      attr + 'vec3 aCenter;\n' +
      attr + 'vec3 aQuad;\n' +
      attr + 'vec4 aInstColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform vec3 uCamRight;\n' +
      'uniform vec3 uCamUp;\n' +
      varyV + 'vec2 vUv;\n' +
      varyV + 'vec4 vColor;\n' +
      'void main() {\n' +
      '  vec3 worldPos = aCenter + (uCamRight * aQuad.x + uCamUp * aQuad.y) * aQuad.z;\n' +
      '  vUv = aQuad.xy;\n' +
      '  vColor = aInstColor;\n' +
      '  gl_Position = uViewProj * vec4(worldPos, 1.0);\n' +
      '}\n';

    var solidFS = headF + outDecl + derivFn + '\n' +
      varyF + 'vec2 vUv;\n' +
      varyF + 'vec4 vColor;\n' +
      'uniform float uIsDark;\n' +
      'void main() {\n' +
      '  float r2 = dot(vUv, vUv);\n' +
      '  if (r2 > 1.0) discard;\n' +
      '  float r = sqrt(r2);\n' +
      '  float z = sqrt(max(0.0, 1.0 - r2));\n' +
      '  vec3 N = vec3(vUv.x, vUv.y, z);\n' +
      '  vec3 L = normalize(vec3(0.34, 0.56, 0.75));\n' +
      '  vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));\n' +
      '  float diff = clamp(dot(N, L), 0.0, 1.0);\n' +
      '  float spec = pow(max(dot(N, H), 0.0), 26.0);\n' +
      '  float rim = pow(1.0 - z, 2.3);\n' +
      '  vec3 baseCol = vColor.rgb;\n' +
      '  vec3 lit = baseCol * mix(0.60, 1.15, diff)\n' +
      '    + mix(baseCol, vec3(1.0), 0.78) * spec * (uIsDark > 0.5 ? 0.36 : 0.28)\n' +
      '    + mix(baseCol, vec3(1.0), 0.45) * rim * (uIsDark > 0.5 ? 0.22 : 0.12);\n' +
      '  float aa = max(gradMag(r, vec2(0.025)), 0.014);\n' +
      '  float edgeAlpha = 1.0 - smoothstep(1.0 - aa * 1.6, 1.0, r);\n' +
      '  float ring = smoothstep(0.84, 0.94, r) * (1.0 - smoothstep(0.97, 1.0, r));\n' +
      '  lit = mix(lit, mix(baseCol, vec3(1.0), uIsDark > 0.5 ? 0.46 : 0.22), ring * 0.30);\n' +
      '  ' + outAssign + ' = vec4(lit, vColor.a * edgeAlpha);\n' +
      '}\n';

    var pointVS = headV +
      attr + 'vec3 aPos;\n' +
      attr + 'float aSize;\n' +
      attr + 'vec4 aColor;\n' +
      'uniform mat4 uViewProj;\n' +
      'uniform float uDpr;\n' +
      varyV + 'vec4 vColor;\n' +
      'void main() {\n' +
      '  vColor = aColor;\n' +
      '  gl_Position = uViewProj * vec4(aPos, 1.0);\n' +
      '  gl_PointSize = clamp(aSize * uDpr / max(0.45, gl_Position.w), 2.0, 28.0);\n' +
      '}\n';

    var pointFS = headF + outDecl +
      varyF + 'vec4 vColor;\n' +
      'void main() {\n' +
      '  vec2 pc = gl_PointCoord * 2.0 - 1.0;\n' +
      '  float r2 = dot(pc, pc);\n' +
      '  if (r2 > 1.0) discard;\n' +
      '  float z = sqrt(max(0.0, 1.0 - r2));\n' +
      '  vec3 N = vec3(pc.x, -pc.y, z);\n' +
      '  float diff = clamp(dot(N, normalize(vec3(0.34, 0.56, 0.75))), 0.0, 1.0);\n' +
      '  float edge = 1.0 - smoothstep(0.76, 1.0, r2);\n' +
      '  vec3 col = vColor.rgb * mix(0.68, 1.14, diff) + vec3(0.18) * pow(max(0.0, diff), 8.0);\n' +
      '  ' + outAssign + ' = vec4(col, vColor.a * edge);\n' +
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

    var linePosBuf = gl.createBuffer();
    var lineColBuf = gl.createBuffer();
    var lineVertCount = 0;

    var hiLinePosBuf = gl.createBuffer();
    var hiLineColBuf = gl.createBuffer();
    var hiLineVertCount = 0;

    var solidPosBuf = gl.createBuffer();
    var solidQuadBuf = gl.createBuffer();
    var solidInstColBuf = gl.createBuffer();
    var solidOpaqueVertCount = 0;
    var solidDimVertCount = 0;

    var ptPosBuf = gl.createBuffer();
    var ptSizeBuf = gl.createBuffer();
    var ptColBuf = gl.createBuffer();
    var ptCount = 0;

    var width = 1200;
    var height = 800;
    var dpr = 1;
    var currentField = null;
    var cachedNodes3d = [];
    var cachedPolyNodes = [];
    var lastViewProj = null;

    function resolveLocs(prog, attrs, uniforms) {
      var a = {};
      var u = {};
      var i;
      for (i = 0; i < attrs.length; i++) a[attrs[i]] = gl.getAttribLocation(prog, attrs[i]);
      for (i = 0; i < uniforms.length; i++) u[uniforms[i]] = gl.getUniformLocation(prog, uniforms[i]);
      return { a: a, u: u };
    }

    var tLoc = resolveLocs(terrainProg, ['aXZ', 'aH', 'aGrad'], ['uViewProj', 'uHeightAmp', 'uEye', 'uBg', 'uIsDark', 'uContours', 'uExtent']);
    var lLoc = resolveLocs(lineProg, ['aPos', 'aColor'], ['uViewProj', 'uExtent']);
    var sLoc = resolveLocs(solidProg, ['aCenter', 'aQuad', 'aInstColor'], ['uViewProj', 'uCamRight', 'uCamUp', 'uIsDark']);
    var pLoc = resolveLocs(pointProg, ['aPos', 'aSize', 'aColor'], ['uViewProj', 'uDpr']);

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
      gl.bindBuffer(gl.ARRAY_BUFFER, terrainHBuf);
      gl.bufferData(gl.ARRAY_BUFFER, field.heights, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, terrainGradBuf);
      gl.bufferData(gl.ARRAY_BUFFER, field.grads, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, terrainIdxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    }

    function buildDynamicBatches(data, heightAmp, isDarkTheme) {
      if (!currentField) return;
      var nodes = data.nodes || [];
      var edges = data.edges || [];
      var activeIds = data.activeIds || null;
      var focusSet = data.focusIds || null;
      var hoverNode = data.hoverNode || null;
      var sidebarNode = data.sidebarNode || null;
      var getNodeColor = data.getNodeColor || function () { return '#5b9cf6'; };
      var getNodeAlpha = data.getNodeAlpha || function () { return 1; };
      var showEdges = !!data.showEdges;
      var isLite = !!data.isLite;

      var metaMap = currentField.nodeMeta;
      cachedNodes3d = [];
      var i;
      for (i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        if (!n || !n.id) continue;
        var m = metaMap[n.id];
        if (!m) continue;
        var visible = !activeIds || activeIds.has(n.id);
        var alpha = getNodeAlpha(n);
        if (!visible && alpha <= 0.08 && (!focusSet || !focusSet.has(n.id))) continue;
        var rgb = parseCssColor(getNodeColor(n), [0.36, 0.61, 0.96, 1]);
        var isFocused = !!(focusSet && focusSet.has(n.id));
        var isHovered = !!(hoverNode && hoverNode.id === n.id);
        var isSelected = !!(sidebarNode && sidebarNode.id === n.id);
        cachedNodes3d.push({
          node: n,
          wx: m.wx,
          wz: m.wz,
          wySurf: m.surfaceH * heightAmp,
          wy: m.elevH * heightAmp,
          degNorm: m.degNorm,
          deg: m.deg,
          rgb: rgb,
          alpha: visible ? Math.max(0.2, alpha) : 0.14,
          visible: visible,
          focused: isFocused || isHovered || isSelected,
        });
      }

      // Rank top hubs for 3D polyhedral crystals + vertical stems
      var stemCap = isLite ? 55 : (cachedNodes3d.length <= 160 ? cachedNodes3d.length : 170);
      var ranked = cachedNodes3d.slice().sort(function (a, b) {
        if (a.focused !== b.focused) return a.focused ? -1 : 1;
        if (a.visible !== b.visible) return a.visible ? -1 : 1;
        return b.deg - a.deg;
      });
      var polySet = Object.create(null);
      cachedPolyNodes = [];
      for (i = 0; i < ranked.length; i++) {
        var item = ranked[i];
        if (i < stemCap || item.focused) {
          polySet[item.node.id] = item;
          cachedPolyNodes.push(item);
        }
      }

      // Background points buffer (remaining lower-degree nodes on large graphs)
      var bgPts = [];
      var bgSizes = [];
      var bgCols = [];
      for (i = 0; i < cachedNodes3d.length; i++) {
        var bn = cachedNodes3d[i];
        if (polySet[bn.node.id]) continue;
        bgPts.push(bn.wx, bn.wy, bn.wz);
        bgSizes.push(bn.visible ? (4.5 + bn.degNorm * 7.5) : 3.2);
        bgCols.push(bn.rgb[0], bn.rgb[1], bn.rgb[2], bn.alpha * 0.85);
      }
      ptCount = bgPts.length / 3;
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
        var radius = (nodes.length <= 160 ? 0.022 : 0.012) + pn.degNorm * (nodes.length <= 160 ? 0.022 : 0.018);
        if (pn.focused) radius *= 1.24;
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

      var highlightCenterId = (hoverNode && hoverNode.id) || (sidebarNode && sidebarNode.id) || null;
      var maxBgEdges = isLite ? 1400 : 4200;
      var bgEdgeDrawn = 0;
      var baseEdgeRgb = isDarkTheme ? [0.68, 0.78, 0.92] : [0.24, 0.36, 0.56];

      for (i = 0; i < edges.length; i++) {
        var e = edges[i];
        if (!e) continue;
        var sId = edgeEndId(e.source);
        var tId = edgeEndId(e.target);
        var a = node3dMap[sId];
        var b = node3dMap[tId];
        if (!a || !b) continue;

        var isHi = highlightCenterId && (sId === highlightCenterId || tId === highlightCenterId);
        if (!isHi && (!showEdges || bgEdgeDrawn >= maxBgEdges)) continue;
        if (!isHi && (!a.visible || !b.visible)) continue;

        var mx = (a.wx + b.wx) * 0.5;
        var mz = (a.wz + b.wz) * 0.5;
        var dist = Math.hypot(b.wx - a.wx, b.wz - a.wz);
        var terrainMid = sampleTerrainHeight(currentField, mx, mz) * heightAmp;
        var arcPeak = Math.max((a.wy + b.wy) * 0.5, terrainMid + 0.03) + Math.min(0.14, dist * 0.16);

        var targetPos = isHi ? hlPos : lPos;
        var targetCol = isHi ? hlCol : lCol;
        var col = isHi ? (node3dMap[highlightCenterId] ? node3dMap[highlightCenterId].rgb : a.rgb) : baseEdgeRgb;
        var edgeAlpha = isHi ? 0.82 : (nodes.length <= 160 ? 0.16 : 0.10);

        var segs = isHi ? 6 : 3;
        var px0 = a.wx;
        var py0 = a.wy;
        var pz0 = a.wz;
        for (var sg = 1; sg <= segs; sg++) {
          var u = sg / segs;
          var omt = 1 - u;
          var qx = omt * omt * a.wx + 2 * omt * u * mx + u * u * b.wx;
          var qy = omt * omt * a.wy + 2 * omt * u * arcPeak + u * u * b.wy;
          var qz = omt * omt * a.wz + 2 * omt * u * mz + u * u * b.wz;
          targetPos.push(px0, py0, pz0, qx, qy, qz);
          targetCol.push(col[0], col[1], col[2], edgeAlpha, col[0], col[1], col[2], edgeAlpha);
          px0 = qx; py0 = qy; pz0 = qz;
        }
        if (!isHi) bgEdgeDrawn += 1;
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
      if (camDist < 1.1) maxLabels = Math.round(maxLabels * 1.45);

      var candidates = [];
      var i;
      for (i = 0; i < cachedPolyNodes.length; i++) {
        var n3 = cachedPolyNodes[i];
        if (!n3.visible && !n3.focused) continue;
        var pt = projectWorldToScreen(n3.wx, n3.wy + 0.025, n3.wz, viewProj, width, height);
        if (!pt) continue;
        if (pt[0] < 24 || pt[0] > width - 24 || pt[1] < 24 || pt[1] > height - 24 || pt[2] < -1 || pt[2] > 1) continue;
        candidates.push({
          n3: n3,
          sx: pt[0],
          sy: pt[1] - 11,
          priority: (n3.focused ? 1000000 : 0) + n3.deg * 10 - pt[2],
        });
      }
      candidates.sort(function (a, b) { return b.priority - a.priority; });

      labelCtx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
      labelCtx.textAlign = 'center';
      labelCtx.textBaseline = 'middle';

      var placed = [];
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
          for (var j = 0; j < placed.length; j++) {
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
    }

    function render(params, camState, renderOpts) {
      if (!currentField) return;
      var aspect = width / Math.max(1, height);
      var cam = buildCameraMatrices(params, camState, aspect);
      lastViewProj = cam.viewProj;
      var heightAmp = (params.height / 100) * 0.46;
      var isDarkTheme = !!renderOpts.isDark;
      var bgArr = parseCssColor(renderOpts.bgColor || (isDarkTheme ? '#191919' : '#f7f6f3'), [0.098, 0.098, 0.098, 1]);

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
      gl.uniform3f(tLoc.u.uEye, cam.eye[0], cam.eye[1], cam.eye[2]);
      gl.uniform3f(tLoc.u.uBg, bgArr[0], bgArr[1], bgArr[2]);
      gl.uniform1f(tLoc.u.uIsDark, isDarkTheme ? 1 : 0);
      gl.uniform1f(tLoc.u.uContours, params.contours || 16);
      gl.uniform1f(tLoc.u.uExtent, currentField.extent);
      bindAttrib(tLoc.a.aXZ, terrainXZBuf, 2, activeAttrs);
      bindAttrib(tLoc.a.aH, terrainHBuf, 1, activeAttrs);
      bindAttrib(tLoc.a.aGrad, terrainGradBuf, 2, activeAttrs);
      useAttrs(activeAttrs);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, terrainIdxBuf);
      gl.drawElements(gl.TRIANGLES, terrainIdxCount, gl.UNSIGNED_SHORT, 0);

      // 2. Round 3D Hub Spheres (opaque pass before translucent arcs)
      var camRight = [cam.view[0], cam.view[4], cam.view[8]];
      var camUp = [cam.view[1], cam.view[5], cam.view[9]];
      if (solidOpaqueVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(solidProg);
        gl.uniformMatrix4fv(sLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform3f(sLoc.u.uCamRight, camRight[0], camRight[1], camRight[2]);
        gl.uniform3f(sLoc.u.uCamUp, camUp[0], camUp[1], camUp[2]);
        gl.uniform1f(sLoc.u.uIsDark, isDarkTheme ? 1 : 0);
        bindAttrib(sLoc.a.aCenter, solidPosBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aQuad, solidQuadBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aInstColor, solidInstColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, 0, solidOpaqueVertCount);
      }

      // Translucent overlay layers share read-only depth test against terrain & solids
      gl.depthMask(false);

      // 3. Stems, Peak Beams & Background Edges
      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(lLoc.u.uViewProj, false, cam.viewProj);
      gl.uniform1f(lLoc.u.uExtent, currentField.extent);
      if (lineVertCount > 0) {
        activeAttrs = {};
        bindAttrib(lLoc.a.aPos, linePosBuf, 3, activeAttrs);
        bindAttrib(lLoc.a.aColor, lineColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.LINES, 0, lineVertCount);
      }

      // 4. Background Low-Degree Round Node Orbs
      if (ptCount > 0) {
        activeAttrs = {};
        gl.useProgram(pointProg);
        gl.uniformMatrix4fv(pLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(pLoc.u.uDpr, dpr);
        bindAttrib(pLoc.a.aPos, ptPosBuf, 3, activeAttrs);
        bindAttrib(pLoc.a.aSize, ptSizeBuf, 1, activeAttrs);
        bindAttrib(pLoc.a.aColor, ptColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.POINTS, 0, ptCount);
      }

      // 4b. Dimmed Background Hub Spheres (focus/filter mode)
      if (solidDimVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(solidProg);
        gl.uniformMatrix4fv(sLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform3f(sLoc.u.uCamRight, camRight[0], camRight[1], camRight[2]);
        gl.uniform3f(sLoc.u.uCamUp, camUp[0], camUp[1], camUp[2]);
        gl.uniform1f(sLoc.u.uIsDark, isDarkTheme ? 1 : 0);
        bindAttrib(sLoc.a.aCenter, solidPosBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aQuad, solidQuadBuf, 3, activeAttrs);
        bindAttrib(sLoc.a.aInstColor, solidInstColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.TRIANGLES, solidOpaqueVertCount, solidDimVertCount);
      }

      // 5. Focused / Hovered Highlight Neighbor Arcs
      if (hiLineVertCount > 0) {
        activeAttrs = {};
        gl.useProgram(lineProg);
        gl.uniformMatrix4fv(lLoc.u.uViewProj, false, cam.viewProj);
        gl.uniform1f(lLoc.u.uExtent, currentField.extent);
        bindAttrib(lLoc.a.aPos, hiLinePosBuf, 3, activeAttrs);
        bindAttrib(lLoc.a.aColor, hiLineColBuf, 4, activeAttrs);
        useAttrs(activeAttrs);
        gl.drawArrays(gl.LINES, 0, hiLineVertCount);
      }

      gl.depthMask(true);

      // 6. Crisp HiDPI Billboard Labels
      drawLabelsOverlay(cam.viewProj, {
        isDark: isDarkTheme,
        isLite: renderOpts.isLite,
        showLabels: renderOpts.showLabels,
        getLabelText: renderOpts.getLabelText,
        distance: camState.distance,
      });
    }

    function pickNodeAtScreen(localX, localY) {
      if (!lastViewProj || !cachedNodes3d.length) return null;
      var best = null;
      var bestScore = Infinity;
      for (var i = 0; i < cachedNodes3d.length; i++) {
        var n3 = cachedNodes3d[i];
        if (!n3.visible && !n3.focused) continue;
        var pt = projectWorldToScreen(n3.wx, n3.wy, n3.wz, lastViewProj, width, height);
        if (!pt || pt[2] < -1 || pt[2] > 1) continue;
        var hitRadius = (n3.degNorm > 0.35 ? 16 : 11) + (n3.focused ? 4 : 0);
        var distPx = Math.hypot(pt[0] - localX, pt[1] - localY);
        if (distPx <= hitRadius) {
          var score = distPx + pt[2] * 2;
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
      getTopScreenTargets: function (limit) {
        if (!lastViewProj) return [];
        var maxCount = limit || 12;
        var out = [];
        for (var i = 0; i < cachedPolyNodes.length && out.length < maxCount; i++) {
          var n3 = cachedPolyNodes[i];
          if (!n3.visible && !n3.focused) continue;
          var pt = projectWorldToScreen(n3.wx, n3.wy, n3.wz, lastViewProj, width, height);
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
    var getBgColor = opts.getBgColor || function () { return isDark() ? '#191919' : '#f7f6f3'; };
    var getLabelText = opts.getLabelText || function (d) { return (d && (d._info && d._info.title || d.label)) || ''; };
    var isLabelsVisible = opts.isLabelsVisible || function () { return true; };
    var isEdgesVisible = opts.isEdgesVisible || function () { return false; };
    var onNodeHover = opts.onNodeHover || function () {};
    var onNodeClick = opts.onNodeClick || function () {};
    var onBackgroundClick = opts.onBackgroundClick || function () {};
    var onStateChange = opts.onStateChange || function () {};
    var onUnsupported = opts.onUnsupported || function () {};

    var persist = opts.persist !== false;
    var allowTouchOrbit = opts.allowTouchOrbit !== false;
    var defaultOverrides = (opts.defaultParamsByTier && typeof opts.defaultParamsByTier === 'object')
      ? opts.defaultParamsByTier
      : null;

    function resolveDefaultForTier(t) {
      var base = getDefaultsForTier(t);
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
    var hoveredNode = null;
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

    function rebuildFieldAndBatches() {
      if (!renderer || !enabled) return;
      var data = getGraphData() || {};
      var nodes = data.nodes || [];
      var edges = data.edges || [];
      if (terrainDirty || !terrainField) {
        degreeInfo = computeNodeDegrees(nodes, edges);
        var gridN = tier === 'lite' ? 92 : (nodes.length <= 160 ? 120 : 148);
        terrainField = computeTerrainField(nodes, degreeInfo, {
          gridSize: gridN,
          extent: 1.35,
          activeIds: data.activeIds || null,
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
          getNodeColor: data.getNodeColor,
          getNodeAlpha: data.getNodeAlpha,
          showEdges: isEdgesVisible(),
          isLite: tier === 'lite',
        }, heightAmp, isDark());
        batchesDirty = false;
      }
    }

    function renderNow() {
      renderRaf = 0;
      if (!enabled || tier === 'unsupported') return;
      var r = ensureStageCanvases();
      if (!r) return;
      var vp = getViewportSize();
      r.resize(vp.width || 1200, vp.height || 800, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1);
      rebuildFieldAndBatches();
      r.render(params, camState, {
        isDark: isDark(),
        bgColor: getBgColor(),
        isLite: tier === 'lite',
        showLabels: isLabelsVisible(),
        getLabelText: getLabelText,
      });
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

    /* ── 鼠标 / 触控交互（轨道旋转、平移、滚轮缩放、节点拾取）── */
    var dragState = null;

    function getStageLocalPoint(ev) {
      if (!stageEl) return [0, 0];
      var rect = stageEl.getBoundingClientRect();
      return [ev.clientX - rect.left, ev.clientY - rect.top];
    }

    function onPointerDown(ev) {
      if (!enabled || !stageEl) return;
      if (ev.pointerType === 'touch' && !allowTouchOrbit) {
        dragState = {
          id: ev.pointerId,
          startX: ev.clientX,
          startY: ev.clientY,
          lastX: ev.clientX,
          lastY: ev.clientY,
          moved: false,
          mode: 'tap-only',
        };
        return;
      }
      var isPan = ev.button === 2 || ev.button === 1 || ev.shiftKey;
      dragState = {
        id: ev.pointerId,
        startX: ev.clientX,
        startY: ev.clientY,
        lastX: ev.clientX,
        lastY: ev.clientY,
        moved: false,
        mode: isPan ? 'pan' : 'orbit',
      };
      if (glCanvas && typeof glCanvas.setPointerCapture === 'function') {
        try { glCanvas.setPointerCapture(ev.pointerId); } catch (_e) { /* ignore */ }
      }
    }

    function onPointerMove(ev) {
      if (!enabled || !renderer) return;
      if (dragState && dragState.id === ev.pointerId) {
        var dx = ev.clientX - dragState.lastX;
        var dy = ev.clientY - dragState.lastY;
        dragState.lastX = ev.clientX;
        dragState.lastY = ev.clientY;
        if (Math.hypot(ev.clientX - dragState.startX, ev.clientY - dragState.startY) > 4) {
          dragState.moved = true;
        }
        if (dragState.mode === 'orbit' && dragState.moved) {
          var nextYaw = ((params.yaw - dx * 0.38 + 540) % 360) - 180;
          var nextPitch = Math.max(BOUNDS.pitch.min, Math.min(BOUNDS.pitch.max, params.pitch + dy * 0.28));
          params = clampParams({
            height: params.height,
            contours: params.contours,
            pitch: nextPitch,
            yaw: nextYaw,
          }, tier);
          saveParamsIfEnabled(params);
          onStateChange(getPublicState());
          scheduleRender();
          return;
        }
        if (dragState.mode === 'pan' && dragState.moved) {
          var yawRad = (params.yaw * Math.PI) / 180;
          var speed = camState.distance * 0.0022;
          var rx = Math.cos(yawRad);
          var rz = -Math.sin(yawRad);
          var fx = Math.sin(yawRad);
          var fz = Math.cos(yawRad);
          camState.targetX = Math.max(-0.9, Math.min(0.9, camState.targetX - dx * rx * speed + dy * fx * speed));
          camState.targetZ = Math.max(-0.9, Math.min(0.9, camState.targetZ - dx * rz * speed + dy * fz * speed));
          scheduleRender();
          return;
        }
      }
      if (ev.pointerType === 'touch') return;
      var pt = getStageLocalPoint(ev);
      var hit = renderer.pickNodeAtScreen(pt[0], pt[1]);
      var prevId = hoveredNode ? hoveredNode.id : null;
      var nextId = hit ? hit.id : null;
      if (prevId !== nextId) {
        hoveredNode = hit || null;
        batchesDirty = true;
        scheduleRender();
      }
      onNodeHover(hit || null, ev);
    }

    function onPointerUp(ev) {
      if (!enabled) return;
      var wasDrag = dragState && dragState.id === ev.pointerId ? dragState : null;
      dragState = null;
      if (!wasDrag || wasDrag.moved) return;
      if (ev.button === 2) return;
      if (!renderer) return;
      var pt = getStageLocalPoint(ev);
      var hit = renderer.pickNodeAtScreen(pt[0], pt[1]);
      if (hit) onNodeClick(hit, ev);
      else onBackgroundClick(ev);
    }

    function onWheel(ev) {
      if (!enabled) return;
      ev.preventDefault();
      var factor = ev.deltaY > 0 ? 1.09 : 0.91;
      camState.distance = Math.max(0.65, Math.min(3.8, camState.distance * factor));
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
      stageEl.addEventListener('wheel', onWheel, { passive: false });
      stageEl.addEventListener('contextmenu', onContextMenu);
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
        renderNow();
      } else {
        hoveredNode = null;
        dragState = null;
        clearStageClasses();
      }
      onStateChange(getPublicState());
      return true;
    }

    function setParam(key, value) {
      if (!Object.prototype.hasOwnProperty.call(BOUNDS, key)) return;
      var next = Object.assign({}, params);
      next[key] = value;
      params = clampParams(next, tier);
      saveParamsIfEnabled(params);
      if (key === 'height') batchesDirty = true;
      if (enabled) scheduleRender();
      onStateChange(getPublicState());
    }

    function resetDefaults() {
      params = resolveDefaultForTier(tier);
      camState.distance = opts.defaultDistance || 1.72;
      camState.targetX = 0;
      camState.targetZ = 0;
      saveParamsIfEnabled(params);
      batchesDirty = true;
      if (enabled) renderNow();
      onStateChange(getPublicState());
    }

    function setTopView() {
      params = clampParams({
        height: params.height,
        contours: params.contours,
        pitch: TOP_VIEW_PARAMS.pitch,
        yaw: TOP_VIEW_PARAMS.yaw,
      }, tier);
      camState.targetX = 0;
      camState.targetZ = 0;
      saveParamsIfEnabled(params);
      if (enabled) renderNow();
      onStateChange(getPublicState());
    }

    function syncData(rebuildField) {
      if (rebuildField !== false) terrainDirty = true;
      batchesDirty = true;
      if (enabled) scheduleRender();
    }

    function refreshTheme() {
      batchesDirty = true;
      if (enabled) scheduleRender();
    }

    function fitToView() {
      camState.distance = opts.defaultDistance || 1.72;
      camState.targetX = 0;
      camState.targetZ = 0;
      if (enabled) scheduleRender();
    }

    function syncViewport() {
      if (!enabled) return;
      renderNow();
    }

    function destroy() {
      if (renderRaf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(renderRaf);
      if (stageEl && typeof stageEl.removeEventListener === 'function') {
        stageEl.removeEventListener('pointerdown', onPointerDown);
        stageEl.removeEventListener('pointermove', onPointerMove);
        stageEl.removeEventListener('pointerup', onPointerUp);
        stageEl.removeEventListener('pointercancel', onPointerUp);
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
      fitToView: fitToView,
      syncViewport: syncViewport,
      refreshStageRect: function () {},
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
          animating: !!draggingSlider || !!renderRaf,
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
    getDefaultsForTier: getDefaultsForTier,
    clampParams: clampParams,
    resolveCapabilityTier: resolveCapabilityTier,
    detectEnvironment: detectEnvironment,
    parseCssColor: parseCssColor,
    computeNodeDegrees: computeNodeDegrees,
    computeTerrainField: computeTerrainField,
    sampleTerrainHeight: sampleTerrainHeight,
    buildCameraMatrices: buildCameraMatrices,
    projectWorldToScreen: projectWorldToScreen,
    createController: createController,
  };
});
