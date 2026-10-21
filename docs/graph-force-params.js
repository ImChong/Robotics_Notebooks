(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RNGraphForce = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * 2D 图谱力模拟参数与装配：graph.html（浏览器首屏 / 刷新布局）与
   * scripts/precompute_graph_layout.cjs（导出期预计算布局）共用同一份，避免两边漂移。
   * 语义见 graph.html 力模拟一节：alpha 单调衰减一次落稳；叶序铺开；中等摩擦 + 度数加权弹簧。
   */
  var PARAMS = {
    CENTER_STRENGTH: 0.05,
    VELOCITY_DECAY: 0.54,
    ALPHA_DECAY: 0.04,
    LINK_STRENGTH: 0.72,   // 实际强度 = 该值 / sqrt(min(deg_s, deg_t))
    LINK_DISTANCE: 80,
    COLLIDE_STRENGTH: 0.45,
    COLLIDE_PADDING: 5,
    CHARGE_STRENGTH: -100, // 排斥力默认值（页面滑块可改）
    SEED_RADIUS: 10,       // 叶序种子半径系数：略紧于完全舒展，给首帧可见扩张动能
  };

  function endpointId(v) { return typeof v === 'object' && v !== null ? v.id : v; }

  /** 度数加权弹簧强度 */
  function linkStrengthFor(degreeMap) {
    return function (link) {
      var deg = Math.min(degreeMap[endpointId(link.source)] || 1, degreeMap[endpointId(link.target)] || 1);
      return PARAMS.LINK_STRENGTH / Math.max(1, Math.sqrt(deg));
    };
  }

  /** 叶序铺开（确定性），以 (cx, cy) 为中心 */
  function seedPhyllotaxis(list, cx, cy) {
    var goldenAngle = Math.PI * (3 - Math.sqrt(5));
    for (var i = 0; i < list.length; i++) {
      var n = list[i];
      n.fx = null; n.fy = null; n.vx = 0; n.vy = 0;
      var r = PARAMS.SEED_RADIUS * Math.sqrt(0.5 + i);
      var a = i * goldenAngle;
      n.x = cx + r * Math.cos(a);
      n.y = cy + r * Math.sin(a);
    }
  }

  function makeCollide(d3, radiusOf) {
    return d3.forceCollide().radius(function (d) { return radiusOf(d) + PARAMS.COLLIDE_PADDING; }).strength(PARAMS.COLLIDE_STRENGTH);
  }

  /**
   * 装配力模拟（已 stop）。opts: { degreeMap, radiusOf, cx, cy, theta, alphaMin, chargeStrength }
   * 不设 distanceMax：硬截断会让越过阈值的节点瞬间失/得全部斥力，形成来回拉扯的极限环。
   */
  function buildSimulation(d3, nodes, edges, opts) {
    return d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(function (d) { return d.id; })
        .distance(PARAMS.LINK_DISTANCE).strength(linkStrengthFor(opts.degreeMap)))
      .force('charge', d3.forceManyBody()
        .strength(opts.chargeStrength != null ? opts.chargeStrength : PARAMS.CHARGE_STRENGTH).theta(opts.theta))
      .force('center', d3.forceCenter(opts.cx, opts.cy).strength(PARAMS.CENTER_STRENGTH))
      .force('collision', makeCollide(d3, opts.radiusOf))
      .force('x', d3.forceX(opts.cx).strength(PARAMS.CENTER_STRENGTH))
      .force('y', d3.forceY(opts.cy).strength(PARAMS.CENTER_STRENGTH))
      .velocityDecay(PARAMS.VELOCITY_DECAY)
      .alphaDecay(PARAMS.ALPHA_DECAY)
      .alphaMin(opts.alphaMin)
      .stop();
  }

  return {
    PARAMS: PARAMS,
    linkStrengthFor: linkStrengthFor,
    seedPhyllotaxis: seedPhyllotaxis,
    makeCollide: makeCollide,
    buildSimulation: buildSimulation,
  };
});
