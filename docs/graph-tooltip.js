(function () {
  'use strict';

  function createDismissGuard() {
    var lastDismissAt = 0;
    return function runDismiss(fn) {
      if (!fn()) return;
      var now = Date.now();
      if (now - lastDismissAt < 300) return;
      lastDismissAt = now;
      return true;
    };
  }

  /**
   * 移动端：点击画布空白处关闭已固定的节点浮窗。
   * 仅当 event.target 落在 canvasEl 内时生效。
   *
   * @param {Element} canvasEl - SVG 画布（接收 pointer/click 的容器）
   * @param {{ isMobile: boolean, getPinned: function, clearPin: function, hide: function }} tooltipApi
   * @param {{ nodeSelector?: string, tooltipEl?: Element }} [opts]
   */
  function bindGraphTooltipBlankDismiss(canvasEl, tooltipApi, opts) {
    if (!canvasEl || !tooltipApi) return;
    opts = opts || {};
    var nodeSelector = opts.nodeSelector || '.node-g';
    var tooltipEl = opts.tooltipEl || null;
    var guard = createDismissGuard();

    function shouldDismiss(ev) {
      if (!tooltipApi.isMobile || !tooltipApi.getPinned()) return false;
      var target = ev.target;
      if (!target || !canvasEl.contains(target)) return false;
      var closest = target.closest;
      if (!closest) return false;
      if (closest.call(target, nodeSelector)) return false;
      if (closest.call(target, '.tt-link')) return false;
      if (tooltipEl && tooltipEl.contains(target)) return false;
      return true;
    }

    function dismissFromBlank(ev) {
      if (!guard(function () { return shouldDismiss(ev); })) return;
      tooltipApi.clearPin();
      tooltipApi.hide();
    }

    function onClick(ev) {
      if (ev.defaultPrevented) return;
      dismissFromBlank(ev);
    }

    canvasEl.addEventListener('pointerup', dismissFromBlank);
    canvasEl.addEventListener('click', onClick);
  }

  /**
   * 移动端：点击画布以外区域关闭已固定的节点浮窗（如详情页正文、首页搜索区等）。
   *
   * @param {Element} canvasEl - SVG 画布，用于排除画布内点击
   * @param {{ isMobile: boolean, getPinned: function, clearPin: function, hide: function }} tooltipApi
   * @param {{ tooltipEl?: Element, dismissRootEl?: Element }} [opts]
   */
  function bindGraphTooltipOutsideDismiss(canvasEl, tooltipApi, opts) {
    if (!canvasEl || !tooltipApi) return;
    opts = opts || {};
    var tooltipEl = opts.tooltipEl || null;
    var dismissRootEl = opts.dismissRootEl || document.body;
    var guard = createDismissGuard();

    function shouldDismiss(ev) {
      if (!tooltipApi.isMobile || !tooltipApi.getPinned()) return false;
      var target = ev.target;
      if (!target) return false;
      if (canvasEl.contains(target)) return false;
      if (tooltipEl && tooltipEl.contains(target)) return false;
      if (dismissRootEl && !dismissRootEl.contains(target)) return false;
      return true;
    }

    function dismissFromOutside(ev) {
      if (!guard(function () { return shouldDismiss(ev); })) return;
      tooltipApi.clearPin();
      tooltipApi.hide();
    }

    function onClick(ev) {
      if (ev.defaultPrevented) return;
      dismissFromOutside(ev);
    }

    document.addEventListener('pointerup', dismissFromOutside);
    document.addEventListener('click', onClick);
  }

  /** 与详情页一致：仅含 type 元数据的摘要不在浮窗正文重复展示 */
  function isMetadataOnlySummary(summary) {
    return /^type:\s*[\w-]+[。.]?$/i.test(String(summary || '').trim());
  }

  /** 浮窗摘要：过滤 type 占位行，并按 maxLen 截断 */
  function formatTooltipSummary(raw, maxLen) {
    var s = String(raw || '').trim();
    if (!s || isMetadataOnlySummary(s)) return '';
    maxLen = maxLen == null ? 100 : maxLen;
    if (s.length > maxLen) return s.slice(0, maxLen) + '…';
    return s;
  }

  var GRAPH_NODE_TYPE_LABEL = {
    concept: '概念', method: '方法', task: '任务',
    entity: '工具', comparison: '对比', query: 'Query',
    formalization: '形式化', overview: '总览', reference: '参考',
    '': 'Wiki'
  };

  var GRAPH_NODE_TYPE_COLOR = {
    concept: '#60a5fa', method: '#34d399', task: '#f472b6',
    entity: '#fbbf24', comparison: '#c084fc', query: '#94a3b8',
    formalization: '#fb923c', overview: '#64748b', reference: '#64748b',
    '': '#64748b'
  };

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function colorBadgeHtml(className, label, bgColor, textColor) {
    var fg = textColor || '#0d1117';
    return '<span class="' + className + '" style="background:' + escapeHtml(String(bgColor)) +
      ';color:' + escapeHtml(String(fg)) + '">' + escapeHtml(String(label)) + '</span>';
  }

  /** 类型徽章用类型色；社区徽章用社区色 */
  function buildMetaBadgesHtml(opts) {
    opts = opts || {};
    var html = '';
    if (opts.showType !== false) {
      var typeKey = opts.type != null ? opts.type : '';
      var typeLabels = opts.typeLabels || GRAPH_NODE_TYPE_LABEL;
      var typeColors = opts.typeColors || GRAPH_NODE_TYPE_COLOR;
      var typeLabel = typeLabels[typeKey] || typeKey || 'Wiki';
      var typeColor = typeColors[typeKey] || typeColors[''] || '#64748b';
      html = colorBadgeHtml('tt-type', typeLabel, typeColor);
    }
    if (opts.communityLabel) {
      var communityColor = opts.communityColor || '#64748b';
      html += colorBadgeHtml('tt-community', opts.communityLabel, communityColor);
    }
    if (!html) return '';
    return '<div class="tt-meta-badges">' + html + '</div>';
  }

  function buildNodeTooltipHtml(opts) {
    opts = opts || {};
    var badges = buildMetaBadgesHtml(opts);
    var title = escapeHtml(String(opts.title || opts.id || ''));
    var summary = opts.summary
      ? '<div class="tt-summary">' + escapeHtml(String(opts.summary)) + '</div>'
      : '';
    var extra = opts.extraHtml || '';
    var link = opts.linkHtml || '';
    return badges +
      '<div class="tt-title">' + title + '</div>' +
      summary +
      extra +
      link;
  }

  window.RNGraphTooltip = {
    bindBlankDismiss: bindGraphTooltipBlankDismiss,
    bindOutsideDismiss: bindGraphTooltipOutsideDismiss,
    isMetadataOnlySummary: isMetadataOnlySummary,
    formatTooltipSummary: formatTooltipSummary,
    GRAPH_NODE_TYPE_LABEL: GRAPH_NODE_TYPE_LABEL,
    GRAPH_NODE_TYPE_COLOR: GRAPH_NODE_TYPE_COLOR,
    buildMetaBadgesHtml: buildMetaBadgesHtml,
    buildNodeTooltipHtml: buildNodeTooltipHtml
  };
})();
