/* Robotics Discovery: curated robotics websites ledger list + hover detail popover controller. */
(function (root) {
  'use strict';

  var STORAGE_LANG_KEY = 'robotics-home-language';

  var I18N = {
    zh: {
      page_title: '研究发现 | Robotics Notebooks',
      page_desc:
        'Robotics Discovery（研究发现）：按论文、开源项目、模型与数据集分类的机器人研究网站列表，悬停查看详情并直达原站。',
      skip_link: '跳到正文',
      site_brand: 'Robotics Notebooks | 机器人技术栈地图',
      home_aria: '机器人技术栈地图首页',
      main_nav_aria: '主导航',
      mobile_nav_aria: '移动端导航',
      nav_top: '顶部',
      nav_start: '入口',
      nav_search: '搜索',
      nav_latest: '最新',
      nav_hubs: '榜单',
      nav_graph: '图谱',
      nav_trends: '趋势',
      nav_discovery: '发现',
      ui_classic: '旧版',
      ui_modern: '新版',
      ui_to_classic: '切换到旧版样式',
      ui_to_modern: '切换到新版样式',
      theme_aria: '切换白天黑夜模式',
      menu_aria: '打开或关闭导航菜单',
      back_home: '← 返回首页',
      hero_kicker: '[00] ROBOTICS DISCOVERY · 研究发现与推荐网站',
      hero_title: '研究发现',
      hero_subtitle:
        '面向机器人研究与工程的一手网站列表：按论文、开源项目、模型与数据集分类，鼠标悬停查看站点详情，点击直达原站浏览。',
      telemetry_aria: '研究发现概览',
      stat_sources_sub: '精选网站入口',
      stat_papers_sub: '论文跟踪榜单',
      stat_projects_sub: '开源项目热榜',
      stat_hubs_sub: '模型与数据集',
      sources_kicker: '01 / 推荐网站列表',
      sources_title: '推荐网站列表',
      sources_subtitle: '鼠标悬停在任一条目上即可查看简介、适用场景与更新节奏，点击直达原站。',
      filter_aria: '按类别筛选推荐网站',
      cat_all: '全部',
      cat_papers: '论文',
      cat_projects: '开源项目',
      cat_models: '模型',
      cat_datasets: '数据集',
      loading_sources: '正在加载推荐网站列表…',
      sources_error: '推荐网站配置加载失败，请刷新重试。',
      empty_sources: '该分类下暂无推荐网站。',
      best_for_label: '适合关注',
      cadence_label: '榜单 / 更新',
      popover_hint: '点击直达原站 ↗',
      row_info_btn: '详情',
      row_info_close: '收起',
      row_info_aria: '查看详情',
      footer_author: '刘冲',
      footer_tail: '2026 · 机器人技术栈地图',
      lang_btn_label: 'English',
      lang_btn_aria: 'Switch page to English'
    },
    en: {
      page_title: 'Robotics Discovery | Robotics Notebooks',
      page_desc:
        'Robotics Discovery: curated list of robotics paper, repository, model, and dataset feeds with hover details and direct links.',
      skip_link: 'Skip to content',
      site_brand: 'Robotics Notebooks | Robotics Knowledge Map',
      home_aria: 'Robotics Knowledge Map Home',
      main_nav_aria: 'Main navigation',
      mobile_nav_aria: 'Mobile navigation',
      nav_top: 'Top',
      nav_start: 'Starting points',
      nav_search: 'Search',
      nav_latest: 'Latest',
      nav_hubs: 'Rankings',
      nav_graph: 'Graph',
      nav_trends: 'Trends',
      nav_discovery: 'Discovery',
      ui_classic: 'Classic',
      ui_modern: 'Modern',
      ui_to_classic: 'Switch to classic UI',
      ui_to_modern: 'Switch to modern UI',
      theme_aria: 'Toggle light / dark mode',
      menu_aria: 'Open or close navigation menu',
      back_home: '← Back to Home',
      hero_kicker: '[00] ROBOTICS DISCOVERY · RESOURCE DIRECTORY',
      hero_title: 'Robotics Discovery',
      hero_subtitle:
        'A compact directory for tracking robotics papers, open-source projects, models, and datasets directly on their original feeds. Hover any row for full details or click to open.',
      telemetry_aria: 'Discovery overview',
      stat_sources_sub: 'Curated Feeds',
      stat_papers_sub: 'Paper Boards',
      stat_projects_sub: 'Repo Boards',
      stat_hubs_sub: 'Models & Datasets',
      sources_kicker: '01 / RECOMMENDED WEBSITES',
      sources_title: 'Recommended Websites',
      sources_subtitle:
        'Hover or focus any row to inspect its summary, best-for scope, and update cadence; click to open directly.',
      filter_aria: 'Filter recommended websites by category',
      cat_all: 'All',
      cat_papers: 'Papers',
      cat_projects: 'Open Source',
      cat_models: 'Models',
      cat_datasets: 'Datasets',
      loading_sources: 'Loading recommended websites…',
      sources_error: 'Could not load recommended websites configuration. Please refresh.',
      empty_sources: 'No websites under this category.',
      best_for_label: 'Best for',
      cadence_label: 'Feed / Cadence',
      popover_hint: 'Click row to open original site ↗',
      row_info_btn: 'Info',
      row_info_close: 'Hide',
      row_info_aria: 'Toggle details',
      footer_author: 'Chong Liu',
      footer_tail: '2026 · Robotics Notebooks',
      lang_btn_label: '中文',
      lang_btn_aria: '切换页面为中文'
    }
  };

  function escapeHtml(value) {
    if (value == null) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function safeUrl(raw) {
    if (!raw || typeof raw !== 'string') return '';
    try {
      var parsed = new URL(raw.trim());
      if (parsed.protocol === 'https:' || parsed.protocol === 'http:') return parsed.href;
    } catch (_err) {
      return '';
    }
    return '';
  }

  function iconSvg(name, extraClass) {
    if (root.RNIcons && typeof root.RNIcons.svg === 'function') {
      return root.RNIcons.svg(name, extraClass);
    }
    return '';
  }

  function formatRowIndex(idx) {
    var num = idx + 1;
    return num < 10 ? '0' + num : String(num);
  }

  function initDiscoveryPage(doc, win) {
    doc = doc || (typeof document !== 'undefined' ? document : null);
    win = win || (typeof window !== 'undefined' ? window : null);
    if (!doc || !win) return null;

    var state = {
      lang: 'zh',
      activeCategory: 'all',
      sourcesData: null,
      sourcesError: false
    };

    try {
      if (win.localStorage && win.localStorage.getItem(STORAGE_LANG_KEY) === 'en') {
        state.lang = 'en';
      }
    } catch (_err) { /* ignore storage restriction */ }

    var els = {
      langBtn: doc.getElementById('discoveryLanguageToggle'),
      uiBtn: doc.getElementById('uiStyleToggle'),
      themeBtn: doc.getElementById('themeToggle'),
      menuBtn: doc.getElementById('mobileMenuToggle'),
      telemetry: doc.querySelector('.discovery-telemetry'),
      sourcesMount: doc.getElementById('discoverySourcesMount'),
      categoryTabs: doc.getElementById('discoveryCategoryTabs'),
      statSources: doc.getElementById('discoveryStatSources'),
      statPapers: doc.getElementById('discoveryStatPapers'),
      statProjects: doc.getElementById('discoveryStatProjects'),
      statHubs: doc.getElementById('discoveryStatHubs'),
      popover: doc.getElementById('discoveryDetailPopover')
    };

    var siteLookup = Object.create(null);
    var activeAnchor = null;
    var lastPointerType = '';

    function t(key) {
      return (I18N[state.lang] && I18N[state.lang][key]) || I18N.zh[key] || key;
    }

    function applyStaticI18n() {
      var dict = I18N[state.lang] || I18N.zh;
      var isClassic = doc.documentElement.getAttribute('data-ui') === 'classic';
      doc.documentElement.lang = state.lang === 'en' ? 'en' : 'zh-CN';
      doc.title = dict.page_title;
      var descMeta = doc.querySelector('meta[name="description"]');
      if (descMeta) descMeta.setAttribute('content', dict.page_desc);

      var siteTitle = doc.querySelector('.site-title');
      if (siteTitle) siteTitle.setAttribute('aria-label', dict.home_aria);
      var mainNav = doc.querySelector('.main-nav');
      if (mainNav) mainNav.setAttribute('aria-label', dict.main_nav_aria);
      var mobileNav = doc.getElementById('mobileNavSheet');
      if (mobileNav) mobileNav.setAttribute('aria-label', dict.mobile_nav_aria);
      if (els.telemetry) els.telemetry.setAttribute('aria-label', dict.telemetry_aria);
      if (els.categoryTabs) els.categoryTabs.setAttribute('aria-label', dict.filter_aria);

      if (els.uiBtn) {
        els.uiBtn.textContent = isClassic ? dict.ui_modern : dict.ui_classic;
        var uiLabel = isClassic ? dict.ui_to_modern : dict.ui_to_classic;
        els.uiBtn.setAttribute('aria-label', uiLabel);
        els.uiBtn.setAttribute('title', uiLabel);
      }
      if (els.themeBtn) {
        els.themeBtn.setAttribute('aria-label', dict.theme_aria);
        els.themeBtn.setAttribute('title', dict.theme_aria);
      }
      if (els.menuBtn) {
        els.menuBtn.setAttribute('aria-label', dict.menu_aria);
      }

      if (els.langBtn) {
        els.langBtn.textContent = dict.lang_btn_label;
        els.langBtn.setAttribute('aria-label', dict.lang_btn_aria);
        els.langBtn.setAttribute('lang', state.lang === 'en' ? 'zh-CN' : 'en');
      }

      doc.querySelectorAll('[data-i18n]').forEach(function (el) {
        var key = el.getAttribute('data-i18n');
        if (!key) return;
        if (Object.prototype.hasOwnProperty.call(dict, key)) {
          el.textContent = dict[key];
        }
      });
    }

    function siteFields(site) {
      var isEn = state.lang === 'en';
      return {
        name: isEn ? (site.name_en || site.name_zh) : (site.name_zh || site.name_en),
        summary: isEn ? (site.summary_en || site.summary_zh) : (site.summary_zh || site.summary_en),
        bestFor: isEn ? (site.best_for_en || site.best_for_zh) : (site.best_for_zh || site.best_for_en),
        cadence: isEn ? (site.cadence_en || site.cadence_zh) : (site.cadence_zh || site.cadence_en),
        kindBadge: isEn ? (site.kind_badge_en || site.kind_badge_zh) : (site.kind_badge_zh || site.kind_badge_en),
        href: safeUrl(site.url)
      };
    }

    function renderSiteRow(site, indexInCat) {
      var f = siteFields(site);
      var id = String(site.id || 'site-' + indexInCat);
      var safeId = escapeHtml(id);
      var idxLabel = formatRowIndex(indexInCat);
      var rowContent =
        '<span class="discovery-row-index">' + escapeHtml(idxLabel) + '</span>' +
        '<span class="discovery-row-name">' + escapeHtml(f.name) + '</span>' +
        '<span class="discovery-row-sub">' +
          '<span class="discovery-row-domain">' + escapeHtml(site.domain || '') + '</span>' +
          '<span class="discovery-badge">' + escapeHtml(f.kindBadge) + '</span>' +
        '</span>' +
        '<span class="discovery-row-arrow" aria-hidden="true">↗</span>';

      var anchorHtml = f.href
        ? '<a class="discovery-row-anchor" href="' + escapeHtml(f.href) + '" target="_blank" rel="noopener noreferrer" data-site-id="' + safeId + '">' + rowContent + '</a>'
        : '<span class="discovery-row-anchor discovery-row-anchor--static" tabindex="0" data-site-id="' + safeId + '">' + rowContent + '</span>';

      return (
        '<li class="discovery-row" data-site-id="' + safeId + '">' +
          '<div class="discovery-row-main">' +
            anchorHtml +
            '<button type="button" class="discovery-row-info-btn" data-site-id="' + safeId + '" aria-expanded="false" aria-controls="discovery-inline-' + safeId + '" aria-label="' + escapeHtml(t('row_info_aria') + ': ' + f.name) + '">' + escapeHtml(t('row_info_btn')) + '</button>' +
          '</div>' +
          '<div class="discovery-row-inline-detail" id="discovery-inline-' + safeId + '" hidden>' +
            '<p class="discovery-inline-summary">' + escapeHtml(f.summary) + '</p>' +
            '<dl class="discovery-popover-meta">' +
              '<div class="discovery-popover-meta-row"><dt>' + escapeHtml(t('best_for_label')) + '</dt><dd>' + escapeHtml(f.bestFor) + '</dd></div>' +
              '<div class="discovery-popover-meta-row"><dt>' + escapeHtml(t('cadence_label')) + '</dt><dd>' + escapeHtml(f.cadence) + '</dd></div>' +
            '</dl>' +
          '</div>' +
        '</li>'
      );
    }

    function renderPopoverHtml(entry) {
      var cat = entry.category || {};
      var catLabel = state.lang === 'en' ? (cat.label_en || cat.label_zh || '') : (cat.label_zh || cat.label_en || '');
      var f = siteFields(entry.site);
      return (
        '<div class="discovery-popover-head">' +
          '<span class="discovery-popover-cat">' + escapeHtml(catLabel) + '</span>' +
          '<span class="discovery-badge">' + escapeHtml(f.kindBadge) + '</span>' +
        '</div>' +
        '<p class="discovery-popover-title">' + escapeHtml(f.name) + '</p>' +
        (f.href ? '<p class="discovery-popover-url">' + escapeHtml(f.href) + '</p>' : '') +
        '<p class="discovery-popover-summary">' + escapeHtml(f.summary) + '</p>' +
        '<dl class="discovery-popover-meta">' +
          '<div class="discovery-popover-meta-row"><dt>' + escapeHtml(t('best_for_label')) + '</dt><dd>' + escapeHtml(f.bestFor) + '</dd></div>' +
          '<div class="discovery-popover-meta-row"><dt>' + escapeHtml(t('cadence_label')) + '</dt><dd>' + escapeHtml(f.cadence) + '</dd></div>' +
        '</dl>' +
        (f.href ? '<p class="discovery-popover-hint">' + escapeHtml(t('popover_hint')) + '</p>' : '')
      );
    }

    function positionPopover(anchor, pointerX) {
      if (!els.popover || !anchor) return;
      var rect = anchor.getBoundingClientRect();
      var popRect = els.popover.getBoundingClientRect();
      var popW = popRect.width || 400;
      var popH = popRect.height || 220;
      var vw = win.innerWidth || 1280;
      var vh = win.innerHeight || 800;
      var PAD = 16;
      var GAP = 8;

      var rawLeft = (typeof pointerX === 'number' && pointerX > 0) ? (pointerX - 48) : (rect.left + 48);
      var maxLeft = Math.max(PAD, vw - popW - PAD);
      var left = Math.min(Math.max(PAD, rawLeft), maxLeft);

      var top = rect.bottom + GAP;
      if (top + popH > vh - PAD && rect.top - GAP - popH >= PAD) {
        top = rect.top - GAP - popH;
      } else {
        var maxTop = Math.max(PAD, vh - popH - PAD);
        top = Math.min(Math.max(PAD, top), maxTop);
      }

      els.popover.style.left = Math.round(left) + 'px';
      els.popover.style.top = Math.round(top) + 'px';
    }

    function hidePopover() {
      if (activeAnchor) {
        activeAnchor.removeAttribute('aria-describedby');
        var prevRow = activeAnchor.closest('.discovery-row');
        if (prevRow) prevRow.classList.remove('is-hovered');
        activeAnchor = null;
      }
      if (!els.popover) return;
      els.popover.classList.remove('is-visible');
      els.popover.hidden = true;
      els.popover.setAttribute('aria-hidden', 'true');
    }

    function showPopover(anchor, pointerX) {
      if (!els.popover || !anchor) return;
      var siteId = anchor.getAttribute('data-site-id');
      var entry = siteId ? siteLookup[siteId] : null;
      if (!entry) {
        hidePopover();
        return;
      }

      if (activeAnchor && activeAnchor !== anchor) {
        activeAnchor.removeAttribute('aria-describedby');
        var oldRow = activeAnchor.closest('.discovery-row');
        if (oldRow) oldRow.classList.remove('is-hovered');
      }

      els.popover.innerHTML = renderPopoverHtml(entry);
      els.popover.hidden = false;
      els.popover.setAttribute('aria-hidden', 'false');
      els.popover.classList.add('is-visible');
      positionPopover(anchor, pointerX);

      anchor.setAttribute('aria-describedby', 'discoveryDetailPopover');
      var row = anchor.closest('.discovery-row');
      if (row) row.classList.add('is-hovered');
      activeAnchor = anchor;
    }

    function updateTelemetry(sites) {
      var papersCount = sites.filter(function (s) { return s.category === 'papers'; }).length;
      var projectsCount = sites.filter(function (s) { return s.category === 'projects'; }).length;
      var hubsCount = sites.filter(function (s) {
        return s.category === 'models' || s.category === 'datasets';
      }).length;
      if (els.statSources) els.statSources.textContent = String(sites.length);
      if (els.statPapers) els.statPapers.textContent = String(papersCount);
      if (els.statProjects) els.statProjects.textContent = String(projectsCount);
      if (els.statHubs) els.statHubs.textContent = String(hubsCount);
    }

    function renderSources() {
      hidePopover();
      if (!els.sourcesMount) return;
      els.sourcesMount.classList.remove('data-loading');
      if (state.sourcesError) {
        els.sourcesMount.innerHTML =
          '<div class="discovery-banner discovery-banner--error">' +
            '<p class="discovery-status-note">' + escapeHtml(t('sources_error')) + '</p>' +
          '</div>';
        return;
      }
      if (!state.sourcesData) return;

      var categories = Array.isArray(state.sourcesData.categories) ? state.sourcesData.categories : [];
      var sites = Array.isArray(state.sourcesData.sites) ? state.sourcesData.sites : [];
      updateTelemetry(sites);

      siteLookup = Object.create(null);
      var catById = Object.create(null);
      categories.forEach(function (cat) {
        if (cat && cat.id) catById[cat.id] = cat;
      });
      sites.forEach(function (site) {
        if (site && site.id) {
          siteLookup[site.id] = { site: site, category: catById[site.category] || {} };
        }
      });

      var visibleCats = state.activeCategory === 'all'
        ? categories
        : categories.filter(function (c) { return c.id === state.activeCategory; });

      var html = visibleCats.map(function (cat) {
        var catSites = sites.filter(function (s) { return s.category === cat.id; });
        if (!catSites.length) return '';
        var label = state.lang === 'en' ? (cat.label_en || cat.label_zh) : (cat.label_zh || cat.label_en);
        var desc = state.lang === 'en' ? (cat.desc_en || cat.desc_zh) : (cat.desc_zh || cat.desc_en);
        return (
          '<section class="discovery-cat-group" data-category-group="' + escapeHtml(cat.id) + '">' +
            '<header class="discovery-cat-head">' +
              '<div class="discovery-cat-title-wrap">' +
                iconSvg(cat.icon || 'tag', 'discovery-cat-icon') +
                '<h3 class="discovery-cat-title">' + escapeHtml(label) + '</h3>' +
                '<span class="discovery-cat-count">' + catSites.length + '</span>' +
              '</div>' +
              '<p class="discovery-cat-desc">' + escapeHtml(desc) + '</p>' +
            '</header>' +
            '<ul class="discovery-list" role="list">' +
              catSites.map(renderSiteRow).join('') +
            '</ul>' +
          '</section>'
        );
      }).join('');

      els.sourcesMount.innerHTML = html || '<p class="data-meta">' + escapeHtml(t('empty_sources')) + '</p>';
    }

    function renderAll() {
      applyStaticI18n();
      renderSources();
    }

    if (els.langBtn) {
      els.langBtn.addEventListener('click', function () {
        state.lang = state.lang === 'en' ? 'zh' : 'en';
        try {
          if (win.localStorage) win.localStorage.setItem(STORAGE_LANG_KEY, state.lang);
        } catch (_err) { /* optional */ }
        renderAll();
      });
    }

    win.addEventListener('rn:uichange', function () {
      applyStaticI18n();
    });

    if (els.categoryTabs) {
      els.categoryTabs.addEventListener('click', function (ev) {
        var target = ev.target && typeof ev.target.closest === 'function' ? ev.target : null;
        var btn = target ? target.closest('button[data-category]') : null;
        if (!btn) return;
        state.activeCategory = btn.getAttribute('data-category') || 'all';
        els.categoryTabs.querySelectorAll('button[data-category]').forEach(function (tab) {
          var active = tab === btn;
          tab.classList.toggle('is-active', active);
          tab.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        renderSources();
      });
    }

    if (els.sourcesMount) {
      doc.addEventListener('pointerdown', function (ev) {
        if (ev && ev.pointerType) lastPointerType = ev.pointerType;
      }, { passive: true });

      doc.addEventListener('pointermove', function (ev) {
        if (ev && ev.pointerType) lastPointerType = ev.pointerType;
      }, { passive: true });

      els.sourcesMount.addEventListener('mouseover', function (ev) {
        if (lastPointerType === 'touch') return;
        var target = ev.target && typeof ev.target.closest === 'function' ? ev.target : null;
        var anchor = target ? target.closest('.discovery-row-anchor') : null;
        if (!anchor) {
          hidePopover();
          return;
        }
        if (anchor !== activeAnchor) {
          showPopover(anchor, ev.clientX);
        }
      });

      els.sourcesMount.addEventListener('mouseleave', function () {
        hidePopover();
      });

      els.sourcesMount.addEventListener('focusin', function (ev) {
        var target = ev.target && typeof ev.target.closest === 'function' ? ev.target : null;
        var anchor = target ? target.closest('.discovery-row-anchor') : null;
        if (anchor) showPopover(anchor, null);
      });

      els.sourcesMount.addEventListener('focusout', function (ev) {
        var next = ev.relatedTarget && typeof ev.relatedTarget.closest === 'function' ? ev.relatedTarget : null;
        if (!next || !els.sourcesMount.contains(next) || !next.closest('.discovery-row-anchor')) {
          hidePopover();
        }
      });

      els.sourcesMount.addEventListener('click', function (ev) {
        var target = ev.target && typeof ev.target.closest === 'function' ? ev.target : null;
        var infoBtn = target ? target.closest('.discovery-row-info-btn') : null;
        if (!infoBtn) return;
        var siteId = infoBtn.getAttribute('data-site-id');
        var drawer = siteId ? doc.getElementById('discovery-inline-' + siteId) : null;
        if (!drawer) return;
        var nextOpen = drawer.hidden;
        drawer.hidden = !nextOpen;
        infoBtn.setAttribute('aria-expanded', nextOpen ? 'true' : 'false');
        infoBtn.textContent = nextOpen ? t('row_info_close') : t('row_info_btn');
      });
    }

    doc.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') hidePopover();
    });
    win.addEventListener('scroll', hidePopover, { passive: true });
    win.addEventListener('resize', hidePopover, { passive: true });

    function loadSources() {
      return win.fetch('discovery-sources.json')
        .then(function (resp) {
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          return resp.json();
        })
        .then(function (data) {
          state.sourcesData = data;
          state.sourcesError = false;
          renderSources();
          return data;
        })
        .catch(function () {
          state.sourcesError = true;
          renderSources();
          return null;
        });
    }

    applyStaticI18n();
    var sourcesPromise = loadSources();

    return {
      state: state,
      renderAll: renderAll,
      showPopover: showPopover,
      hidePopover: hidePopover,
      ready: sourcesPromise
    };
  }

  var api = {
    I18N: I18N,
    escapeHtml: escapeHtml,
    safeUrl: safeUrl,
    formatRowIndex: formatRowIndex,
    initDiscoveryPage: initDiscoveryPage
  };

  root.RNDiscovery = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else if (typeof document !== 'undefined' && !root.__rnDiscoveryManualInit) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () { initDiscoveryPage(document, window); });
    } else {
      initDiscoveryPage(document, window);
    }
  }
})(typeof window !== 'undefined' ? window : globalThis);
