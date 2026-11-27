/* Robotics Discovery: static recommended robotics websites directory controller. */
(function (root) {
  'use strict';

  var STORAGE_LANG_KEY = 'robotics-home-language';

  var I18N = {
    zh: {
      page_title: 'Robotics Discovery · 研究发现 | Robotics Notebooks',
      page_desc:
        'Robotics Discovery（机器人研究发现）：按论文、开源项目、模型与数据集分类的机器人研究网站导航，直达原站榜单。',
      skip_link: '跳到正文',
      site_brand: 'Robotics Notebooks | 机器人技术栈地图',
      home_aria: '机器人技术栈地图首页',
      main_nav_aria: '主导航',
      mobile_nav_aria: '移动端导航',
      nav_home: '首页',
      nav_sources: '推荐网站',
      nav_graph: '完整图谱',
      nav_trends: '增长趋势',
      nav_discovery: '研究发现',
      mobile_ui_classic: '旧版样式',
      mobile_ui_modern: '新版样式',
      back_home: '← 返回首页',
      hero_kicker: '[00] ROBOTICS DISCOVERY · 研究发现与资源导航',
      hero_title: 'Robotics Discovery · 机器人研究发现',
      hero_subtitle:
        '面向机器人研究与工程读者的资源导航：帮你快速定位追踪最新论文、开源项目、模型与数据集的权威入口，直接跳转原站浏览，不做全网热度排名或第三方榜单镜像。',
      jump_sources: '浏览推荐网站',
      stat_sources_sub: '精选网站入口',
      stat_papers_sub: '论文跟踪榜单',
      stat_projects_sub: '开源项目热榜',
      stat_hubs_sub: '模型与数据集',
      sources_kicker: '01 / 推荐网站导航',
      sources_title: '推荐网站导航',
      sources_subtitle:
        '按论文、开源项目、模型与数据集分类，无需登录或等待采集任务即可直接跳转原站浏览榜单；外部网站临时不可用时不影响本页正常访问。',
      cat_all: '全部',
      cat_papers: '论文',
      cat_projects: '开源项目',
      cat_models: '模型',
      cat_datasets: '数据集',
      loading_sources: '正在加载推荐网站导航…',
      sources_error: '推荐网站配置加载失败，请刷新重试。',
      empty_sources: '该分类下暂无推荐网站。',
      best_for_label: '适合关注',
      cadence_label: '榜单 / 更新',
      visit_site: '访问原站',
      footer_author: '刘冲',
      footer_tail: '2026 · 机器人技术栈地图',
      lang_btn_label: 'English',
      lang_btn_aria: 'Switch page to English'
    },
    en: {
      page_title: 'Robotics Discovery | Robotics Notebooks',
      page_desc:
        'Robotics Discovery: curated directory of robotics paper, repository, model, and dataset feeds with direct links to original boards.',
      skip_link: 'Skip to content',
      site_brand: 'Robotics Notebooks | Robotics Knowledge Map',
      home_aria: 'Robotics Knowledge Map Home',
      main_nav_aria: 'Main navigation',
      mobile_nav_aria: 'Mobile navigation',
      nav_home: 'Home',
      nav_sources: 'Websites',
      nav_graph: 'Full Graph',
      nav_trends: 'Trends',
      nav_discovery: 'Discovery',
      mobile_ui_classic: 'Classic UI',
      mobile_ui_modern: 'Modern UI',
      back_home: '← Back to Home',
      hero_kicker: '[00] ROBOTICS DISCOVERY · RESOURCE DIRECTORY',
      hero_title: 'Robotics Discovery',
      hero_subtitle:
        'A reader-first directory for tracking robotics papers, open-source projects, models, and datasets directly on their original feeds — without building a mirrored global leaderboard.',
      jump_sources: 'Browse Websites',
      stat_sources_sub: 'Curated Feeds',
      stat_papers_sub: 'Paper Boards',
      stat_projects_sub: 'Repo Boards',
      stat_hubs_sub: 'Models & Datasets',
      sources_kicker: '01 / RECOMMENDED WEBSITES',
      sources_title: 'Recommended Websites',
      sources_subtitle:
        'Organized by papers, open-source projects, models, and datasets. Jump straight to original boards with no sign-in required; external site downtime never blocks this directory.',
      cat_all: 'All',
      cat_papers: 'Papers',
      cat_projects: 'Open Source',
      cat_models: 'Models',
      cat_datasets: 'Datasets',
      loading_sources: 'Loading recommended websites…',
      sources_error: 'Could not load website directory configuration. Please refresh.',
      empty_sources: 'No websites under this category.',
      best_for_label: 'Best for',
      cadence_label: 'Feed / Cadence',
      visit_site: 'Visit Site',
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

  function gridClassForCount(count) {
    if (count <= 1) return 'discovery-site-grid discovery-site-grid--single';
    if (count === 2 || count === 4) return 'discovery-site-grid discovery-site-grid--2col';
    return 'discovery-site-grid discovery-site-grid--3col';
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
      sourcesMount: doc.getElementById('discoverySourcesMount'),
      categoryTabs: doc.getElementById('discoveryCategoryTabs'),
      statSources: doc.getElementById('discoveryStatSources'),
      statPapers: doc.getElementById('discoveryStatPapers'),
      statProjects: doc.getElementById('discoveryStatProjects'),
      statHubs: doc.getElementById('discoveryStatHubs')
    };

    function t(key) {
      return (I18N[state.lang] && I18N[state.lang][key]) || I18N.zh[key] || key;
    }

    function applyStaticI18n() {
      var dict = I18N[state.lang] || I18N.zh;
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

      if (els.langBtn) {
        els.langBtn.textContent = dict.lang_btn_label;
        els.langBtn.setAttribute('aria-label', dict.lang_btn_aria);
        els.langBtn.setAttribute('lang', state.lang === 'en' ? 'zh-CN' : 'en');
      }

      doc.querySelectorAll('[data-i18n]').forEach(function (el) {
        var key = el.getAttribute('data-i18n');
        if (!key) return;
        if (key === 'mobile_ui_classic') {
          var isClassic = doc.documentElement.getAttribute('data-ui') === 'classic';
          el.textContent = isClassic ? dict.mobile_ui_modern : dict.mobile_ui_classic;
          return;
        }
        if (Object.prototype.hasOwnProperty.call(dict, key)) {
          el.textContent = dict[key];
        }
      });
    }

    function renderSiteCard(site) {
      var isEn = state.lang === 'en';
      var name = isEn ? (site.name_en || site.name_zh) : (site.name_zh || site.name_en);
      var summary = isEn ? (site.summary_en || site.summary_zh) : (site.summary_zh || site.summary_en);
      var bestFor = isEn ? (site.best_for_en || site.best_for_zh) : (site.best_for_zh || site.best_for_en);
      var cadence = isEn ? (site.cadence_en || site.cadence_zh) : (site.cadence_zh || site.cadence_en);
      var kindBadge = isEn ? (site.kind_badge_en || site.kind_badge_zh) : (site.kind_badge_zh || site.kind_badge_en);
      var href = safeUrl(site.url);

      return (
        '<article class="discovery-site-card">' +
          '<div class="discovery-site-card-top">' +
            '<span class="discovery-site-domain">' + escapeHtml(site.domain || '') + '</span>' +
            '<span class="discovery-badge">' + escapeHtml(kindBadge) + '</span>' +
          '</div>' +
          '<h4 class="discovery-site-name">' +
            (href
              ? '<a href="' + escapeHtml(href) + '" target="_blank" rel="noopener noreferrer">' +
                  '<span>' + escapeHtml(name) + '</span>' +
                  iconSvg('external-link', 'discovery-ext-icon') +
                '</a>'
              : escapeHtml(name)) +
          '</h4>' +
          '<p class="discovery-site-summary">' + escapeHtml(summary) + '</p>' +
          '<dl class="discovery-site-meta">' +
            '<div class="discovery-site-meta-row"><dt>' + escapeHtml(t('best_for_label')) + '</dt><dd>' + escapeHtml(bestFor) + '</dd></div>' +
            '<div class="discovery-site-meta-row"><dt>' + escapeHtml(t('cadence_label')) + '</dt><dd>' + escapeHtml(cadence) + '</dd></div>' +
          '</dl>' +
          (href
            ? '<div class="discovery-site-actions">' +
                '<a class="discovery-site-cta" href="' + escapeHtml(href) + '" target="_blank" rel="noopener noreferrer">' +
                  '<span>' + escapeHtml(t('visit_site')) + '</span>' +
                  '<span aria-hidden="true">↗</span>' +
                '</a>' +
              '</div>'
            : '') +
        '</article>'
      );
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
            '<div class="' + gridClassForCount(catSites.length) + '">' +
              catSites.map(renderSiteCard).join('') +
            '</div>' +
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
        var btn = ev.target.closest('button[data-category]');
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
      ready: sourcesPromise
    };
  }

  var api = {
    I18N: I18N,
    escapeHtml: escapeHtml,
    safeUrl: safeUrl,
    gridClassForCount: gridClassForCount,
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
