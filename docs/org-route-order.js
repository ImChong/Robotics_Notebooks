// 公司 / 实验室路线共用时间排序：首页入口、路线导航、增长趋势保持一致。
(function (global) {
  'use strict';

  // 年 / 月精度补到年初 / 月初；忽略空值和无效日期，避免 NaN 扰乱排序。
  function normalizeDate(value) {
    var match = /^([1-9]\d{3})(?:-(0[1-9]|1[0-2])(?:-(0[1-9]|[12]\d|3[01]))?)?$/.exec(String(value || ''));
    if (!match) return '';
    var month = match[2] || '01';
    var day = match[3] || '01';
    var date = new Date(Date.UTC(Number(match[1]), Number(month) - 1, Number(day)));
    if (date.getUTCMonth() !== Number(month) - 1) return '';
    return match[1] + '-' + month + '-' + day;
  }

  function routeDate(org) {
    var founded = /^\d{4}$/.test(String(org.founded_year)) ? normalizeDate(org.founded_year) : '';
    if (founded) return founded;
    return (org.nodes || []).reduce(function (earliest, node) {
      var date = normalizeDate(node.date);
      return date && (!earliest || date < earliest) ? date : earliest;
    }, '');
  }

  function compare(a, b) {
    // 两种时间都缺失则放到最后；同时间保持输入顺序（合并视图同时间公司在前）。
    var left = routeDate(a) || '\uffff';
    var right = routeDate(b) || '\uffff';
    return left < right ? -1 : left > right ? 1 : 0;
  }

  global.OrgRouteOrder = {
    date: routeDate,
    compare: compare,
    sort: function (orgs) { return (orgs || []).slice().sort(compare); }
  };
})(window);
