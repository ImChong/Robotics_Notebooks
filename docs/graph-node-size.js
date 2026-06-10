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

  window.RNGraphNodeSize = {
    R_MIN: R_MIN,
    R_MAX: R_MAX,
    computeDegreeMap: computeDegreeMap,
    maxDegreeOf: maxDegreeOf,
    radiusForDegree: radiusForDegree
  };
})();
