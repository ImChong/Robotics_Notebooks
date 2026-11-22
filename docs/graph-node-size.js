(function () {
  'use strict';

  /**
   * Graph view、首页知识图谱预览、详情页知识地图（1-hop 邻居）共用的节点尺寸标尺。
   * 半径按 sqrt(degree) 在 [R_MIN, R_MAX] 间归一化插值：
   * 度数 ≤1 的节点取 R_MIN（明显最小），全图最大度数的节点取 R_MAX（明显最大）。
   */
  var R_MIN = 4;
  var R_MAX = 30;

  /** 与 graph view 一致的度数统计：每条边给两端各计 1 度 */
  function computeDegreeMap(edges) {
    var map = {};
    (edges || []).forEach(function (e) {
      map[e.source] = (map[e.source] || 0) + 1;
      map[e.target] = (map[e.target] || 0) + 1;
    });
    return map;
  }

  function maxDegreeOf(degreeMap) {
    var max = 1;
    for (var id in degreeMap) {
      if (degreeMap[id] > max) max = degreeMap[id];
    }
    return max;
  }

  function radiusForDegree(degree, maxDegree) {
    var span = Math.sqrt(Math.max(maxDegree || 1, 2)) - 1;
    var t = (Math.sqrt(Math.max(degree || 0, 1)) - 1) / span;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return R_MIN + (R_MAX - R_MIN) * t;
  }

  // 小图的圆形节点防重叠，拖动中的节点保持固定。
  function resolveOverlaps2D(list, radiusOf, opts) {
    var n = list.length;
    if (n < 2) return 0;
    var cfg = opts || {};
    var pad = cfg.padding != null ? cfg.padding : 5;
    var maxPasses = cfg.maxPasses != null ? cfg.maxPasses : 40;
    var kPush = cfg.kPush != null ? cfg.kPush : 0.78;
    var radii = new Float32Array(n);
    var invMass = new Float32Array(n);
    var accX = new Float32Array(n);
    var accY = new Float32Array(n);
    var accW = new Float32Array(n);
    var i;
    for (i = 0; i < n; i++) {
      var r = radiusOf(list[i]);
      radii[i] = r;
      var fixed = list[i].fx != null && list[i].fy != null;
      invMass[i] = fixed ? 0 : 1 / Math.max(1, r * r);
    }
    var totalResolved = 0;
    for (var pass = 0; pass < maxPasses; pass++) {
      for (i = 0; i < n; i++) {
        accX[i] = 0;
        accY[i] = 0;
        accW[i] = 0;
      }
      var passOverlaps = 0;
      for (i = 0; i < n - 1; i++) {
        var a = list[i];
        var ra = radii[i];
        var invMa = invMass[i];
        for (var j = i + 1; j < n; j++) {
          var b = list[j];
          var invMb = invMass[j];
          var invSum = invMa + invMb;
          if (!(invSum > 0)) continue;
          var minDist = ra + radii[j] + pad;
          var dx = b.x - a.x;
          var dy = b.y - a.y;
          var d2 = dx * dx + dy * dy;
          if (d2 >= minDist * minDist) continue;
          var dist = Math.sqrt(d2);
          var ux;
          var uy;
          if (dist < 1e-4) {
            var ang = (((i * 2654435761 + j * 1597334677) >>> 0) % 1024) * (Math.PI * 2 / 1024);
            ux = Math.cos(ang);
            uy = Math.sin(ang);
          } else {
            ux = dx / dist;
            uy = dy / dist;
          }
          var pen = minDist - dist;
          var wa = invMa / invSum;
          var wb = invMb / invSum;
          accX[i] -= ux * pen * wa;
          accY[i] -= uy * pen * wa;
          accW[i] += 1;
          accX[j] += ux * pen * wb;
          accY[j] += uy * pen * wb;
          accW[j] += 1;
          passOverlaps += 1;
          totalResolved += 1;
        }
      }
      if (passOverlaps === 0) break;
      var maxStep = 0;
      for (i = 0; i < n; i++) {
        if (accW[i] <= 0) continue;
        var damp = kPush / Math.sqrt(accW[i]);
        var mx = accX[i] * damp;
        var my = accY[i] * damp;
        list[i].x += mx;
        list[i].y += my;
        var mv = Math.hypot(mx, my);
        if (mv > maxStep) maxStep = mv;
      }
      if (maxStep < 0.05) break;
    }
    return totalResolved;
  }

  window.RNGraphNodeSize = {
    R_MIN: R_MIN,
    R_MAX: R_MAX,
    computeDegreeMap: computeDegreeMap,
    maxDegreeOf: maxDegreeOf,
    radiusForDegree: radiusForDegree,
    resolveOverlaps2D: resolveOverlaps2D
  };
})();
