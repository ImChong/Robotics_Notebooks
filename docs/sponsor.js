(function () {
  'use strict';

  var script = document.getElementById('sponsor-js') || document.currentScript;
  if (!script || !script.src) return;
  var assetBase = script.src.replace(/[^/]*$/, '');

  function openDialog(dialog) {
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
    toggle.innerHTML = '<span class="sponsor-icon" aria-hidden="true">❤</span>';

    var themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
      headerRight.insertBefore(toggle, themeToggle);
    } else {
      headerRight.appendChild(toggle);
    }

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
          '<p class="sponsor-dialog-hint">微信扫一扫，赞助支持作者 ❤</p>' +
        '</div>' +
        '<div class="sponsor-dialog-body">' +
          '<img class="sponsor-qr" src="' + assetBase + 'sponsor/wechat-pay.png" alt="微信收款码" width="260" height="352" />' +
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
