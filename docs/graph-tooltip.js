(function () {
  'use strict';

  /**
   * 移动端：点击画布空白处关闭已固定的节点浮窗。
   * @param {Element} containerEl - SVG 或画布容器
   * @param {{ isMobile: boolean, getPinned: function, clearPin: function, hide: function }} tooltipApi
   * @param {{ nodeSelector?: string, tooltipEl?: Element }} [opts]
   */
  function bindGraphTooltipBlankDismiss(containerEl, tooltipApi, opts) {
    if (!containerEl || !tooltipApi) return;
    opts = opts || {};
    var nodeSelector = opts.nodeSelector || '.node-g';
    var tooltipEl = opts.tooltipEl || null;

    containerEl.addEventListener('click', function (ev) {
      if (!tooltipApi.isMobile || !tooltipApi.getPinned()) return;
      var closest = ev.target.closest;
      if (!closest) return;
      if (closest.call(ev.target, nodeSelector)) return;
      if (closest.call(ev.target, '.tt-link')) return;
      if (tooltipEl && tooltipEl.contains(ev.target)) return;
      tooltipApi.clearPin();
      tooltipApi.hide();
    });
  }

  window.RNGraphTooltip = {
    bindBlankDismiss: bindGraphTooltipBlankDismiss
  };
})();
