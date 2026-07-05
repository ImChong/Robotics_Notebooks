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

  function formatNodeTypeLabel(typeKey, typeLabels) {
    if (typeLabels) {
      return typeLabels[typeKey] || typeKey || '知识页';
    }
    var api = window.RNWikiTypeLabels;
    if (api && api.formatBilingual) {
      return api.formatBilingual(typeKey);
    }
    return typeKey || '知识页';
  }

  var GRAPH_NODE_TYPE_COLOR = {
    concept: '#60a5fa', method: '#34d399', task: '#f472b6',
    entity: '#fbbf24', comparison: '#c084fc', query: '#94a3b8',
    formalization: '#fb923c', overview: '#64748b', reference: '#64748b',
    roadmap: '#22d3ee', roadmap_page: '#22d3ee',
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

  function shortenCommunityLabel(label) {
    if (!label) return '未分类';
    return String(label).replace(/\s*社区\s*$/, '').trim() || '未分类';
  }

  /** 与 graph.html COMMUNITY_COLORS 一致：Tableau10 + 扩展色 */
  var COMMUNITY_COLORS_FALLBACK = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
    '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
    '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5'
  ];

  function getCommunityColors() {
    if (typeof window !== 'undefined' && window.d3 && window.d3.schemeTableau10) {
      return window.d3.schemeTableau10.concat(COMMUNITY_COLORS_FALLBACK.slice(10));
    }
    return COMMUNITY_COLORS_FALLBACK.slice();
  }

  /** 与 graph.html getSortedCommunities 一致，保证徽标色与图谱社区色对齐 */
  function getSortedCommunities(communities) {
    var list = (communities || []).slice();
    return list.sort(function (a, b) {
      if (a.id === 'community-other') return 1;
      if (b.id === 'community-other') return -1;
      var sizeDiff = (b.size || 0) - (a.size || 0);
      if (sizeDiff !== 0) return sizeDiff;
      return String(a.label || a.id).localeCompare(String(b.label || b.id), 'zh-CN');
    });
  }

  function buildCommunityColorMap(communities) {
    var palette = getCommunityColors();
    var map = {};
    var namedColorIdx = 0;
    getSortedCommunities(communities).forEach(function (c) {
      map[c.id] = c.id === 'community-other'
        ? '#94a3b8'
        : palette[namedColorIdx++ % palette.length];
    });
    return map;
  }

  function hexToRgbChannels(hex) {
    var h = String(hex || '').replace('#', '').trim();
    if (h.length === 3) {
      h = h.split('').map(function (ch) { return ch + ch; }).join('');
    }
    if (h.length !== 6 || !/^[0-9a-fA-F]{6}$/.test(h)) return null;
    return {
      r: parseInt(h.slice(0, 2), 16),
      g: parseInt(h.slice(2, 4), 16),
      b: parseInt(h.slice(4, 6), 16)
    };
  }

  function communityBadgeStyleAttr(color) {
    var rgb = hexToRgbChannels(color);
    if (!rgb) return '';
    return ' style="--community-r:' + rgb.r + ';--community-g:' + rgb.g + ';--community-b:' + rgb.b + '"';
  }

  /** 浮窗/侧边栏类型徽章：badgeColor 由图谱 colorMode 决定（社区/类型/健康度） */
  function buildMetaBadgesHtml(opts) {
    opts = opts || {};
    var html = '';
    if (opts.showType !== false) {
      var typeKey = opts.type != null ? opts.type : '';
      var typeColors = opts.typeColors || GRAPH_NODE_TYPE_COLOR;
      var typeLabel = formatNodeTypeLabel(typeKey, opts.typeLabels);
      var badgeColor = opts.badgeColor != null ? opts.badgeColor
        : (opts.communityColor
          || typeColors[typeKey] || typeColors[''] || '#64748b');
      html = colorBadgeHtml('tt-type', typeLabel, badgeColor);
    }
    if (!html) return '';
    return '<div class="tt-meta-badges">' + html + '</div>';
  }

  /** 与详情页一致的社区跳转按钮；communityColor 与图谱节点社区色一致 */
  function buildCommunityBadgeHtml(communityId, communityLabel, communityColor) {
    if (!communityId || !communityLabel) return '';
    var label = shortenCommunityLabel(communityLabel);
    return '<a class="detail-meta-badge detail-community-badge"' + communityBadgeStyleAttr(communityColor) +
      ' href="graph.html?community=' + encodeURIComponent(communityId) +
      '" title="在知识图谱中查看「' + escapeHtml(label) + '」社区视图">' +
      '<span>🧭</span><span>' + escapeHtml(label) + '</span></a>';
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
    formatNodeTypeLabel: formatNodeTypeLabel,
    GRAPH_NODE_TYPE_COLOR: GRAPH_NODE_TYPE_COLOR,
    shortenCommunityLabel: shortenCommunityLabel,
    COMMUNITY_COLORS_FALLBACK: COMMUNITY_COLORS_FALLBACK,
    getCommunityColors: getCommunityColors,
    getSortedCommunities: getSortedCommunities,
    buildCommunityColorMap: buildCommunityColorMap,
    communityBadgeStyleAttr: communityBadgeStyleAttr,
    buildMetaBadgesHtml: buildMetaBadgesHtml,
    buildCommunityBadgeHtml: buildCommunityBadgeHtml,
    buildNodeTooltipHtml: buildNodeTooltipHtml
  };
})();
