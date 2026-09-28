// 首页「具身公司 · 技术路线」：数据来自 company-roadmaps.json（手工维护，节点 id 由
// tests/test_company_roadmaps.py 校验存在于站内）。上排视角筛选，中排公司 tab，
// 下方按时间展示该公司的技术节点；每个节点给「本库笔记」与「原文」两个出口。
(function () {
  'use strict';

  var mount = document.getElementById('companyRoadmapModule');
  if (!mount) return;

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function detailHref(id) {
    return 'detail.html?id=' + encodeURIComponent(id);
  }

  function lensBadges(company, lenses) {
    return (company.lenses || []).map(function (key) {
      return '<span class="co-lens-badge co-lens-' + esc(key) + '">' + esc(lenses[key] || key) + '</span>';
    }).join('');
  }

  function renderNode(node, index) {
    var date = node.date || '时间未注明';
    var wikiLink = node.id
      ? '<a class="co-node-link" href="' + esc(detailHref(node.id)) + '">本库笔记</a>'
      : '';
    var srcLink = node.url
      ? '<a class="co-node-link co-node-link--ext" href="' + esc(node.url) +
        '" target="_blank" rel="noopener noreferrer">原文 ↗</a>'
      : '';
    return '<li class="co-node" style="--co-i:' + index + '">' +
      '<span class="co-node-dot" aria-hidden="true"></span>' +
      '<div class="co-node-card">' +
      '<div class="co-node-meta"><span class="co-node-date' + (node.date ? '' : ' is-empty') + '">' +
      esc(date) + '</span><span class="co-node-track">' + esc(node.track) + '</span></div>' +
      '<h3 class="co-node-title">' + esc(node.title) + '</h3>' +
      '<p class="co-node-desc">' + esc(node.desc) + '</p>' +
      '<div class="co-node-links">' + wikiLink + srcLink + '</div>' +
      '</div></li>';
  }

  function renderPanel(panel, company, lenses) {
    var nodes = company.nodes || [];
    panel.innerHTML =
      '<div class="co-panel-head">' +
      '<div class="co-panel-title"><strong>' + esc(company.name) + '</strong>' + lensBadges(company, lenses) + '</div>' +
      (company.official
        ? '<a class="co-panel-official" href="' + esc(company.official) +
          '" target="_blank" rel="noopener noreferrer">官方技术入口 ↗</a>'
        : '') +
      '</div>' +
      '<p class="co-panel-summary">' + esc(company.summary) + '</p>' +
      '<ol class="co-timeline" aria-label="' + esc(company.name) + ' 技术节点（' + nodes.length + ' 个）">' +
      nodes.map(renderNode).join('') +
      '</ol>';
  }

  function render(data) {
    var companies = Array.isArray(data && data.companies) ? data.companies : [];
    var lenses = (data && data.lenses) || {};
    if (!companies.length) {
      mount.innerHTML = '<p class="data-meta">暂无公司路线数据。</p>';
      mount.classList.remove('data-loading');
      return;
    }

    var lensButtons = ['<button type="button" class="home-hub-tab is-active" data-lens="" aria-pressed="true">全部</button>'];
    Object.keys(lenses).forEach(function (key) {
      lensButtons.push('<button type="button" class="home-hub-tab" data-lens="' + esc(key) +
        '" aria-pressed="false">' + esc(lenses[key]) + '</button>');
    });

    var tabs = companies.map(function (c) {
      return '<button type="button" role="tab" class="co-tab" id="coTab-' + esc(c.key) +
        '" data-key="' + esc(c.key) + '" data-lenses="' + esc((c.lenses || []).join(' ')) +
        '" aria-controls="coPanel" aria-selected="false" tabindex="-1">' +
        esc(c.name) + '<span class="co-tab-count">' + (c.nodes || []).length + '</span></button>';
    });

    mount.innerHTML =
      '<div class="co-lens-filter" role="group" aria-label="按阅读视角筛选公司">' + lensButtons.join('') + '</div>' +
      '<div class="co-tabs" role="tablist" aria-label="选择公司">' + tabs.join('') + '</div>' +
      '<div class="co-panel" id="coPanel" role="tabpanel" aria-live="polite"></div>';
    mount.classList.remove('data-loading');

    var panel = mount.querySelector('#coPanel');
    var tabEls = Array.prototype.slice.call(mount.querySelectorAll('.co-tab'));
    var lensEls = Array.prototype.slice.call(mount.querySelectorAll('[data-lens]'));
    var byKey = {};
    companies.forEach(function (c) { byKey[c.key] = c; });

    function visibleTabs() {
      return tabEls.filter(function (t) { return !t.hidden; });
    }

    function select(tab, focus) {
      tabEls.forEach(function (t) {
        var on = t === tab;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
      });
      panel.setAttribute('aria-labelledby', tab.id);
      renderPanel(panel, byKey[tab.getAttribute('data-key')], lenses);
      var timeline = panel.querySelector('.co-timeline');
      if (timeline) timeline.scrollLeft = 0;
      if (focus) tab.focus();
    }

    tabEls.forEach(function (tab) {
      tab.addEventListener('click', function () { select(tab, false); });
      tab.addEventListener('keydown', function (e) {
        var list = visibleTabs();
        var i = list.indexOf(tab);
        var next = null;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = list[(i + 1) % list.length];
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = list[(i - 1 + list.length) % list.length];
        else if (e.key === 'Home') next = list[0];
        else if (e.key === 'End') next = list[list.length - 1];
        if (!next) return;
        e.preventDefault();
        select(next, true);
      });
    });

    lensEls.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var lens = btn.getAttribute('data-lens');
        lensEls.forEach(function (b) {
          var on = b === btn;
          b.classList.toggle('is-active', on);
          b.setAttribute('aria-pressed', String(on));
        });
        tabEls.forEach(function (t) {
          t.hidden = !!lens && (' ' + t.getAttribute('data-lenses') + ' ').indexOf(' ' + lens + ' ') < 0;
        });
        var current = mount.querySelector('.co-tab.is-active');
        if (!current || current.hidden) select(visibleTabs()[0], false);
      });
    });

    select(tabEls[0], false);
  }

  fetch('company-roadmaps.json')
    .then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(render)
    .catch(function (err) {
      console.warn('[company-roadmap] 加载失败', err);
      mount.innerHTML = '<p class="data-meta">公司路线数据加载失败，可先看 <a href="' +
        detailHref('wiki-comparisons-robot-foundation-model-company-paths-2026') + '">公司技术路线对照</a>。</p>';
      mount.classList.remove('data-loading');
    });
})();
