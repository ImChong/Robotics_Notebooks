(function () {
  'use strict';

  var script = document.getElementById('sponsor-js') || document.currentScript;
  if (!script || !script.src) return;
  var assetBase = script.src.replace(/[^/]*$/, '');

  function openDialog(dialog) {
    var qr = dialog.querySelector('.sponsor-qr');
    if (qr && !qr.getAttribute('src') && qr.getAttribute('data-src')) {
      qr.setAttribute('src', qr.getAttribute('data-src'));
    }
    dialog.hidden = false;
    document.body.classList.add('sponsor-dialog-open');
    var closeBtn = dialog.querySelector('.sponsor-dialog-close');
    if (closeBtn) closeBtn.focus();
  }

  function closeDialog(dialog) {
    dialog.hidden = true;
    document.body.classList.remove('sponsor-dialog-open');
    var toggle = document.getElementById('sponsorToggle');
    if (toggle) toggle.focus();
  }

  function init() {
    var headerRight = document.querySelector('.header-right');
    if (!headerRight || headerRight.querySelector('.sponsor-toggle')) return;

    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'sponsor-toggle';
    toggle.id = 'sponsorToggle';
    toggle.setAttribute('aria-label', '赞助我');
    toggle.setAttribute('title', '赞助我');
    toggle.setAttribute('aria-haspopup', 'dialog');
    var coffeeSvg =
      (window.RNIcons && window.RNIcons.svg && window.RNIcons.svg('coffee')) ||
      '<svg class="ui-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' +
      '<path d="M3 6.2h7.6v3.9a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V6.2Z"/>' +
      '<path d="M10.6 7.3h1.3a1.7 1.7 0 0 1 0 3.4h-1.3M5.3 3.2c0 .7.6 1 .6 1.6M8.3 3.2c0 .7.6 1 .6 1.6"/>' +
      '</svg>';
    toggle.innerHTML = '<span class="sponsor-icon" aria-hidden="true">' + coffeeSvg + '</span>';

    headerRight.appendChild(toggle);

    var dialog = document.createElement('div');
    dialog.id = 'sponsorDialog';
    dialog.className = 'sponsor-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-labelledby', 'sponsorDialogTitle');
    dialog.hidden = true;
    dialog.innerHTML =
      '<div class="sponsor-dialog-backdrop" data-sponsor-close tabindex="-1"></div>' +
      '<div class="sponsor-dialog-card">' +
        '<div class="sponsor-dialog-header">' +
          '<h2 id="sponsorDialogTitle" class="sponsor-dialog-title">赞助我</h2>' +
          '<p class="sponsor-dialog-hint">微信扫一扫，赞助支持作者</p>' +
        '</div>' +
        '<div class="sponsor-dialog-body">' +
          // 二维码 206 KB，弹窗默认隐藏：点开时再赋 src，不随每个页面首屏下载
          '<img class="sponsor-qr" data-src="' + assetBase + 'sponsor/wechat-pay.png" alt="微信收款码" width="260" height="352" />' +
        '</div>' +
        '<div class="sponsor-dialog-actions">' +
          '<button type="button" class="btn-secondary sponsor-dialog-close" data-sponsor-close>关闭</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(dialog);

    toggle.addEventListener('click', function () {
      openDialog(dialog);
    });

    dialog.addEventListener('click', function (event) {
      if (event.target.closest('[data-sponsor-close]')) {
        closeDialog(dialog);
      }
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && !dialog.hidden) {
        closeDialog(dialog);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
