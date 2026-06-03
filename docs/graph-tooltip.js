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

  window.RNGraphTooltip = {
    bindBlankDismiss: bindGraphTooltipBlankDismiss,
    bindOutsideDismiss: bindGraphTooltipOutsideDismiss,
    isMetadataOnlySummary: isMetadataOnlySummary,
    formatTooltipSummary: formatTooltipSummary
  };
})();
