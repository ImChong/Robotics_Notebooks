// 公司技术路线子页 company.html?id=<key>：数据来自 company-roadmaps.json（手工维护，节点 id 由
// tests/test_company_roadmaps.py 校验存在于站内）。顶部视角筛选 + 公司切换，下方为该公司的
// 纵向时间轴；每个节点给「本库笔记」与「原文」两个出口，页脚提供上一家 / 下一家。
(function () {
  'use strict';

  var mount = document.getElementById('companyRoadmapPage');
  if (!mount) return;

  var COMPARISON_ID = 'wiki-comparisons-robot-foundation-model-company-paths-2026';

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

  function companyHref(key) {
    return 'company.html?id=' + encodeURIComponent(key);
  }

  function extLink(url, text, cls) {
    return '<a class="' + cls + '" href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' + text + '</a>';
  }

  function lensBadges(company, lenses) {
    return (company.lenses || []).map(function (key) {
      return '<span class="co-lens-badge co-lens-' + esc(key) + '">' + esc(lenses[key] || key) + '</span>';
    }).join('');
  }

  function renderSwitcher(companies, lenses, current) {
    var lensButtons = ['<button type="button" class="home-hub-tab is-active" data-lens="" aria-pressed="true">全部</button>'];
    Object.keys(lenses).forEach(function (key) {
      lensButtons.push('<button type="button" class="home-hub-tab" data-lens="' + esc(key) +
        '" aria-pressed="false">' + esc(lenses[key]) + '</button>');
    });
    var chips = companies.map(function (c) {
      var on = current && c.key === current.key;
      return '<a class="co-tab' + (on ? ' is-active' : '') + '" href="' + esc(companyHref(c.key)) +
        '" data-lenses="' + esc((c.lenses || []).join(' ')) + '"' + (on ? ' aria-current="page"' : '') + '>' +
        esc(c.name) + '<span class="co-tab-count">' + (c.nodes || []).length + '</span></a>';
    });
    return '<div class="co-lens-filter" role="group" aria-label="按阅读视角筛选公司">' + lensButtons.join('') + '</div>' +
      '<nav class="co-tabs" aria-label="切换公司">' + chips.join('') + '</nav>';
  }

  function renderNode(node) {
    var links = '';
    if (node.id) links += '<a class="co-node-link" href="' + esc(detailHref(node.id)) + '">本库笔记 →</a>';
    if (node.url) links += extLink(node.url, '原文 ↗', 'co-node-link');
    if (node.date_source) links += extLink(node.date_source, '日期依据 ↗', 'co-node-link');
    return '<li class="co-node">' +
      '<span class="co-node-date' + (node.date ? '' : ' is-empty') + '">' + esc(node.date || '时间未注明') + '</span>' +
      '<span class="co-node-dot" aria-hidden="true"></span>' +
      '<div class="co-node-card">' +
      '<div class="co-node-head"><h3 class="co-node-title">' + esc(node.title) + '</h3>' +
      '<span class="co-node-track">' + esc(node.track) + '</span></div>' +
      '<p class="co-node-desc">' + esc(node.desc) + '</p>' +
      (node.date_note ? '<p class="co-node-desc">日期口径：' + esc(node.date_note) + '</p>' : '') +
      '<div class="co-node-links">' + links + '</div>' +
      '</div></li>';
  }

  // 有日期的节点在跨年处插入年份分隔；未注明时间的节点保持数据里的相对顺序
  function renderTimeline(company) {
    var html = '';
    var lastYear = '';
    (company.nodes || []).forEach(function (node) {
      var year = node.date ? node.date.slice(0, 4) : '';
      if (year && year !== lastYear) {
        html += '<li class="co-year" aria-hidden="true"><span>' + esc(year) + '</span></li>';
        lastYear = year;
      }
      html += renderNode(node);
    });
    return '<ol class="co-timeline" aria-label="' + esc(company.name) + ' 技术节点">' + html + '</ol>';
  }

  function renderCompany(company, lenses) {
    var nodes = company.nodes || [];
    var dates = nodes.map(function (n) { return n.date; }).filter(Boolean);
    var span = dates.length > 1 ? ' · ' + dates[0] + ' → ' + dates[dates.length - 1] : '';
    return '<article class="co-panel" aria-labelledby="coName">' +
      '<div class="co-panel-head">' +
      '<div class="co-panel-title"><h2 id="coName">' + esc(company.name) + '</h2>' + lensBadges(company, lenses) + '</div>' +
      (company.official ? extLink(company.official, '官方技术入口 ↗', 'co-panel-official') : '') +
      '</div>' +
      '<p class="co-panel-summary">' + esc(company.summary) + '</p>' +
      '<p class="co-panel-meta">' + nodes.length + ' 个技术节点' + esc(span) + '</p>' +
      renderTimeline(company) +
      '</article>';
  }

  function renderPager(companies, index) {
    var prev = companies[(index - 1 + companies.length) % companies.length];
    var next = companies[(index + 1) % companies.length];
    return '<nav class="co-pager" aria-label="上一家 / 下一家">' +
      '<a href="' + esc(companyHref(prev.key)) + '">← ' + esc(prev.name) + '</a>' +
      '<a href="' + esc(detailHref(COMPARISON_ID)) + '">三视角对照与阅读顺序</a>' +
      '<a href="' + esc(companyHref(next.key)) + '">' + esc(next.name) + ' →</a>' +
      '</nav>';
  }

  function bindLensFilter() {
    var lensEls = Array.prototype.slice.call(mount.querySelectorAll('[data-lens]'));
    var chips = Array.prototype.slice.call(mount.querySelectorAll('.co-tab'));
    lensEls.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var lens = btn.getAttribute('data-lens');
        lensEls.forEach(function (b) {
          var on = b === btn;
          b.classList.toggle('is-active', on);
          b.setAttribute('aria-pressed', String(on));
        });
        chips.forEach(function (chip) {
          // 当前公司始终保留，避免筛选后找不到自己在哪
          var match = !lens || (' ' + chip.getAttribute('data-lenses') + ' ').indexOf(' ' + lens + ' ') >= 0;
          chip.hidden = !match && !chip.classList.contains('is-active');
        });
      });
    });
  }

  function render(data) {
    var companies = Array.isArray(data && data.companies) ? data.companies : [];
    var lenses = (data && data.lenses) || {};
    mount.classList.remove('data-loading');
    if (!companies.length) {
      mount.innerHTML = '<p class="data-meta">暂无公司路线数据。</p>';
      return;
    }
    var key = new URLSearchParams(window.location.search).get('id') || companies[0].key;
    var index = -1;
    for (var i = 0; i < companies.length; i++) {
      if (companies[i].key === key) index = i;
    }
    var company = index >= 0 ? companies[index] : null;
    var body = company
      ? renderCompany(company, lenses) + renderPager(companies, index)
      : '<p class="data-meta">未找到公司「' + esc(key) + '」，请从上方选择。</p>';
    mount.innerHTML = renderSwitcher(companies, lenses, company) + body;
    if (company) {
      document.title = company.name + ' 技术路线 | Robotics Notebooks';
    }
    bindLensFilter();
    // 窄屏公司条横向滚动时，把当前公司滚进可视区（.co-tabs 为 relative，offsetLeft 相对它）
    var active = mount.querySelector('.co-tab.is-active');
    var bar = mount.querySelector('.co-tabs');
    if (active && bar && bar.scrollWidth > bar.clientWidth) {
      bar.scrollLeft = active.offsetLeft - (bar.clientWidth - active.offsetWidth) / 2;
    }
  }

  fetch('company-roadmaps.json')
    .then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(render)
    .catch(function (err) {
      console.warn('[company-roadmap] 加载失败', err);
      mount.classList.remove('data-loading');
      mount.innerHTML = '<p class="data-meta">公司路线数据加载失败，可先看 <a href="' +
        esc(detailHref(COMPARISON_ID)) + '">公司技术路线对照</a>。</p>';
    });
})();
