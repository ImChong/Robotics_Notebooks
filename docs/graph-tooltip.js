(function () {
  'use strict';

  /**
   * 移动端：点击画布空白处关闭已固定的节点浮窗。
   * 仅当 event.target 落在 canvasEl 内时生效，画布外点击不会关闭。
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
    var lastDismissAt = 0;

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
      if (!shouldDismiss(ev)) return;
      var now = Date.now();
      if (now - lastDismissAt < 300) return;
      lastDismissAt = now;
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

  window.RNGraphTooltip = {
    bindBlankDismiss: bindGraphTooltipBlankDismiss
  };
})();
