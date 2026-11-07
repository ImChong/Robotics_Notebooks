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
    var fg = textColor || '#191919';
    return '<span class="' + className + '" style="background:' + escapeHtml(String(bgColor)) +
      ';color:' + escapeHtml(String(fg)) + '">' + escapeHtml(String(label)) + '</span>';
  }

  // 与 scripts/utils/community_labels.community_short_label 对齐：
  // 「中文（English） 社区」→「中文」
  function shortenCommunityLabel(label) {
    if (!label) return '未分类';
    var base = String(label).replace(/\s*社区\s*$/, '').trim();
    if (!base) return '未分类';
    var head = base.split('（', 1)[0].trim();
    return head || base;
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

  /** 浮窗/侧边栏类型徽章：badgeColor 由图谱 colorMode 决定（社区/类型/开源/健康度） */
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

  /**
   * 浮窗标题 / 摘要是纯文本，公式交给 KaTeX auto-render：
   * 先把 `$...$` 归一为 `\(...\)`（货币判定复用 main.js），`$$`、`\[`、`\(` 原样透传。
   * escapeHtml 后的 `&lt;` 等实体会在文本节点还原，不影响 KaTeX 解析。
   */
  function prepareMathText(text) {
    var api = window.RNMath;
    if (api && api.normalizeDollarMath) return api.normalizeDollarMath(text);
    return String(text || '');
  }

  /** 浮窗内容注入 DOM 后调用：按内容懒加载 KaTeX 并渲染，无公式则不请求组件。 */
  function renderTooltipMath(el) {
    var api = window.RNMath;
    if (el && api && api.render) api.render(el);
  }

  /*
   * 浮窗摘要的行内 Markdown（CommonMark 0.31 子集）：*强调* / _强调_ → <em>，**加粗** / __加粗__ → <strong>，
   * `代码` → <code>，反斜杠转义。摘要里有大量字面星号（A*、RRT*、rmw_*），不能用「两个星号之间」的正则硬配，
   * 按规范 §6.2 的左右侧接（flanking）规则判定能否开 / 闭强调，再按附录「process emphasis」配对。
   * 公式段 \(..\)、\[..\]、$$..$$ 原样保留给 KaTeX（$..$ 已由 prepareMathText 归一）。
   */
  var MD_PUNCT = /[\p{P}\p{S}]/u;
  var MD_SPACE = /\s/;
  var MD_ESCAPABLE = /[!-/:-@[-`{-~]/;
  var MD_MATH = /\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$/y;

  function codePointBefore(src, i) {
    if (i <= 0) return ' '; // 行首按空白处理
    var lo = src.charCodeAt(i - 1);
    return (lo >= 0xdc00 && lo <= 0xdfff && i >= 2) ? src.slice(i - 2, i) : src[i - 1];
  }

  function codePointAt(src, i) {
    if (i >= src.length) return ' '; // 行尾按空白处理
    return String.fromCodePoint(src.codePointAt(i));
  }

  /** 与 i 处同长的反引号串结束位置（找不到返回 -1） */
  function findCodeSpanClose(src, from, len) {
    var re = /`+/g;
    re.lastIndex = from;
    var m;
    while ((m = re.exec(src))) {
      if (m[0].length === len) return m.index;
    }
    return -1;
  }

  function renderInlineMarkdown(text) {
    var src = String(text || '');
    var nodes = [];
    var buf = '';
    function flushText() {
      if (buf) nodes.push({ html: escapeHtml(buf) });
      buf = '';
    }
    var i = 0;
    while (i < src.length) {
      var ch = src[i];
      MD_MATH.lastIndex = i;
      var math = (ch === '\\' || ch === '$') ? MD_MATH.exec(src) : null;
      if (math) {
        buf += math[0];
        i += math[0].length;
      } else if (ch === '\\' && MD_ESCAPABLE.test(src[i + 1] || '')) {
        buf += src[i + 1];
        i += 2;
      } else if (ch === '`') {
        var ticks = /`+/y;
        ticks.lastIndex = i;
        var openLen = ticks.exec(src)[0].length;
        var close = findCodeSpanClose(src, i + openLen, openLen);
        if (close < 0) {
          buf += src.substr(i, openLen);
          i += openLen;
          continue;
        }
        var code = src.slice(i + openLen, close).replace(/\n/g, ' ');
        if (/^ [\s\S]*[^ ][\s\S]* $/.test(code)) code = code.slice(1, -1);
        flushText();
        nodes.push({ html: '<code>' + escapeHtml(code) + '</code>' });
        i = close + openLen;
      } else if (ch === '*' || ch === '_') {
        var end = i;
        while (src[end] === ch) end++;
        var before = codePointBefore(src, i);
        var after = codePointAt(src, end);
        var beforeSpace = MD_SPACE.test(before);
        var afterSpace = MD_SPACE.test(after);
        var beforePunct = MD_PUNCT.test(before);
        var afterPunct = MD_PUNCT.test(after);
        var left = !afterSpace && (!afterPunct || beforeSpace || beforePunct);
        var right = !beforeSpace && (!beforePunct || afterSpace || afterPunct);
        flushText();
        nodes.push({
          delim: ch,
          count: end - i,
          origCount: end - i,
          canOpen: ch === '*' ? left : left && (!right || beforePunct),
          canClose: ch === '*' ? right : right && (!left || afterPunct),
          active: true,
          openTags: '',
          closeTags: ''
        });
        i = end;
      } else {
        buf += ch;
        i += 1;
      }
    }
    flushText();

    // process emphasis：逐个闭合符向前找同字符的开启符；「3 的倍数」规则防止 *foo**bar* 这类误配
    for (var ci = 0; ci < nodes.length; ci++) {
      var closer = nodes[ci];
      if (!closer.delim || !closer.canClose || !closer.active) continue;
      while (closer.count > 0) {
        var oi = ci - 1;
        for (; oi >= 0; oi--) {
          var o = nodes[oi];
          if (!o.delim || !o.active || !o.count || o.delim !== closer.delim || !o.canOpen) continue;
          var sum = o.origCount + closer.origCount;
          if ((o.canClose || closer.canOpen) && sum % 3 === 0 &&
              !(o.origCount % 3 === 0 && closer.origCount % 3 === 0)) continue;
          break;
        }
        if (oi < 0) break;
        var opener = nodes[oi];
        var use = (opener.count >= 2 && closer.count >= 2) ? 2 : 1;
        var tag = use === 2 ? 'strong' : 'em';
        opener.count -= use;
        closer.count -= use;
        opener.openTags = '<' + tag + '>' + opener.openTags;
        closer.closeTags += '</' + tag + '>';
        for (var k = oi + 1; k < ci; k++) {
          if (nodes[k].delim) nodes[k].active = false;
        }
      }
      if (!closer.count || !closer.canOpen) closer.active = false;
    }

    var html = '';
    for (var ni = 0; ni < nodes.length; ni++) {
      var node = nodes[ni];
      html += node.delim
        ? node.closeTags + new Array(node.count + 1).join(node.delim) + node.openTags
        : node.html;
    }
    return html;
  }

  /**
   * 浮窗社区标签：放在摘要下方单独一行（不进顶部类型徽章行，避免两枚徽章挤在一起），
   * 描边胶囊 + 🧭 与详情页「所属社区」按钮同一视觉语言，与实心类型徽章区分。
   */
  function buildTooltipCommunityHtml(communityLabel, communityColor) {
    if (!communityLabel) return '';
    return '<div class="tt-community"' + communityBadgeStyleAttr(communityColor) + '>' +
      '<span aria-hidden="true">🧭</span><span>' + escapeHtml(shortenCommunityLabel(communityLabel)) + '</span></div>';
  }

  /**
   * 浮窗健康度行：「健康度」+ 3 格小条（逐项说明在悬停小条弹出的信息框里）。node 需带 health_score（可选 health_gaps / type）；
   * 小条由 main.js 的 RNNodeHealth 提供（与列表行、页面信息同一套），未加载或无分数则不显示。
   */
  function buildTooltipHealthHtml(node) {
    var api = window.RNNodeHealth;
    if (!api || !node || typeof node.health_score !== 'number') return '';
    return '<div class="tt-health"><span class="tt-health-label">健康度</span>' + api.renderMeta(node) + '</div>';
  }

  function buildNodeTooltipHtml(opts) {
    opts = opts || {};
    var badges = buildMetaBadgesHtml(opts);
    var title = escapeHtml(prepareMathText(opts.title || opts.id || ''));
    // 有开源代码（关联 sources/repos/，has_repo）：标题末尾加 ⭐️，与列表行 / 详情页标题同标记
    if (opts.hasRepo) {
      title += '<span class="updates-item-opensource" aria-label="含开源仓库" title="含开源仓库">⭐️</span>';
    }
    var summary = opts.summary
      ? '<div class="tt-summary">' + renderInlineMarkdown(prepareMathText(opts.summary)) + '</div>'
      : '';
    var community = buildTooltipCommunityHtml(opts.communityLabel, opts.communityColor);
    var health = buildTooltipHealthHtml(opts.health);
    var extra = opts.extraHtml || '';
    var link = opts.linkHtml || '';
    return badges +
      '<div class="tt-title">' + title + '</div>' +
      summary +
      community +
      health +
      extra +
      link;
  }

  /**
   * 浮窗置于最上层：#graph-tooltip 原在 #graph-wrap（position: fixed，自成层叠上下文）里，
   * z-index 只在容器内生效，会被图谱工具栏盖住。统一挪到 body 直接下级，配合 CSS 的最高 z-index，
   * 浮窗总在所有画面之上。浮窗都按视口坐标（fixed）定位，挪位置不影响摆放。
   */
  function liftTooltipsToBody() {
    var els = document.querySelectorAll('#graph-tooltip, .graph-hover-tooltip');
    for (var i = 0; i < els.length; i++) {
      if (els[i].parentNode !== document.body) document.body.appendChild(els[i]);
    }
  }
  if (typeof document !== 'undefined' && typeof document.querySelectorAll === 'function') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', liftTooltipsToBody);
    } else {
      liftTooltipsToBody();
    }
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
    buildNodeTooltipHtml: buildNodeTooltipHtml,
    buildTooltipHealthHtml: buildTooltipHealthHtml,
    renderInlineMarkdown: renderInlineMarkdown,
    renderMath: renderTooltipMath
  };
})();
