/* Robotics Discovery: decoupled recommended website navigation + curated picks controller. */
(function (root) {
  'use strict';

  var STORAGE_LANG_KEY = 'robotics-home-language';
  var BASE_SITE_URL = 'https://imchong.github.io/Robotics_Notebooks/';

  var I18N = {
    zh: {
      page_title: 'Robotics Discovery · 研究发现 | Robotics Notebooks',
      page_desc:
        'Robotics Discovery（机器人研究发现）：机器人论文、开源项目、模型与数据集推荐网站导航，以及每日核验去重的最新研究精选。',
      skip_link: '跳到正文',
      site_brand: 'Robotics Notebooks | 机器人技术栈地图',
      home_aria: '机器人技术栈地图首页',
      main_nav_aria: '主导航',
      mobile_nav_aria: '移动端导航',
      nav_home: '首页',
      nav_sources: '推荐网站',
      nav_picks: '最新精选',
      nav_graph: '完整图谱',
      nav_trends: '增长趋势',
      nav_discovery: '研究发现',
      mobile_ui_classic: '旧版样式',
      mobile_ui_modern: '新版样式',
      back_home: '← 返回首页',
      hero_kicker: '[00] ROBOTICS DISCOVERY · 研究发现与资源导航',
      hero_title: 'Robotics Discovery · 机器人研究发现',
      hero_subtitle:
        '面向机器人研究与工程读者的资源导航与近期精选：优先引导直达原站榜单追踪论文、开源项目、模型与数据集，不做全网热度排名或第三方榜单镜像。',
      jump_sources: '推荐网站导航',
      jump_picks: '最新论文与项目精选',
      stat_sources_sub: '精选网站入口',
      stat_picks_sub: '近期精选条目',
      stat_window_sub: '精选时间范围',
      stat_updated_sub: '最近核验时间',
      sources_kicker: '01 / 推荐网站导航',
      sources_title: '推荐网站导航',
      sources_subtitle:
        '按论文、开源项目、模型、数据集与追踪工具分类。无需登录或等待采集任务即可直接访问原站；即使精选任务或某个外部站点暂不可用，此处导航始终独立可用。',
      cat_all: '全部',
      cat_papers: '论文',
      cat_projects: '开源项目',
      cat_models: '模型',
      cat_datasets: '数据集',
      cat_tools: '补充工具',
      picks_kicker: '02 / 最新论文与项目精选',
      picks_title: '最新论文与项目精选',
      picks_subtitle:
        '优先精选最近 24 小时机器人论文与项目（必要时扩展至近 7 天并标注），优先覆盖具身智能、人形机器人、VLA、WAM、RL、IL、WBC、灵巧操作、仿真、遥操作与 SLAM；同源研究已去重聚合，命中知识库节点时直达详情页。',
      kind_all: '全部',
      kind_paper: '论文',
      kind_project: '项目',
      kind_model: '模型',
      kind_dataset: '数据集',
      copy_list: '复制纯文本清单',
      copy_item: '复制',
      copied_item: '已复制',
      topic_all: '全部方向',
      loading_sources: '正在加载推荐网站导航…',
      loading_picks: '正在加载最新精选…',
      sources_error: '推荐网站配置加载失败，请刷新重试。',
      picks_error_title: '最新精选暂时无法加载',
      picks_error_note: '精选数据请求失败，但上方「推荐网站导航」完全独立且仍可正常跳转使用。',
      empty_sources: '该分类下暂无推荐网站。',
      empty_picks: '当前筛选条件下暂无匹配精选条目。',
      best_for_label: '适合关注',
      cadence_label: '榜单 / 更新',
      visit_site: '访问原站',
      tool_badge: '补充工具 · 非榜单',
      date_published: '发布',
      date_updated: '更新',
      window_24h: '24h',
      window_7d: '7d',
      link_paper: '论文',
      link_project: '项目主页',
      link_code: '源码',
      link_model: '模型',
      link_dataset: '数据集',
      kb_detail: '知识库详情',
      status_updated: '实际更新',
      status_attempt: '最近尝试',
      status_stale_badge: '保留上次有效内容',
      status_partial_badge: '部分源暂不可用',
      status_ok_badge: '已核验同步',
      copied_count_prefix: '已复制 ',
      copied_count_suffix: ' 项纯文本清单（含一手链接）',
      copied_single_prefix: '已复制：',
      copy_empty: '当前无可复制的精选条目。',
      copy_failed: '浏览器未授予剪贴板权限，请手动选择文本复制。',
      footer_author: '刘冲',
      footer_tail: '2026 · 机器人技术栈地图',
      lang_btn_label: 'English',
      lang_btn_aria: 'Switch page to English',
      telemetry_window_24h: '近 24h',
      telemetry_window_7d: '24h + 7d'
    },
    en: {
      page_title: 'Robotics Discovery | Robotics Notebooks',
      page_desc:
        'Robotics Discovery: curated directory of robotics paper, repository, model, and dataset feeds plus daily deduplicated research picks.',
      skip_link: 'Skip to content',
      site_brand: 'Robotics Notebooks | Robotics Knowledge Map',
      home_aria: 'Robotics Knowledge Map Home',
      main_nav_aria: 'Main navigation',
      mobile_nav_aria: 'Mobile navigation',
      nav_home: 'Home',
      nav_sources: 'Websites',
      nav_picks: 'Curated Picks',
      nav_graph: 'Full Graph',
      nav_trends: 'Trends',
      nav_discovery: 'Discovery',
      mobile_ui_classic: 'Classic UI',
      mobile_ui_modern: 'Modern UI',
      back_home: '← Back to Home',
      hero_kicker: '[00] ROBOTICS DISCOVERY · RESOURCE DIRECTORY & DAILY PICKS',
      hero_title: 'Robotics Discovery',
      hero_subtitle:
        'A reader-first directory for tracking robotics papers, open-source projects, models, and datasets directly on their original feeds — without building a mirrored global leaderboard.',
      jump_sources: 'Recommended Websites',
      jump_picks: 'Recent Picks',
      stat_sources_sub: 'Curated Feeds',
      stat_picks_sub: 'Curated Items',
      stat_window_sub: 'Time Window',
      stat_updated_sub: 'Last Verified',
      sources_kicker: '01 / RECOMMENDED WEBSITES',
      sources_title: 'Recommended Websites',
      sources_subtitle:
        'Organized by papers, open-source projects, models, datasets, and tracking utilities. Jump straight to original boards with no sign-in required; navigation remains live regardless of curation or third-party status.',
      cat_all: 'All',
      cat_papers: 'Papers',
      cat_projects: 'Open Source',
      cat_models: 'Models',
      cat_datasets: 'Datasets',
      cat_tools: 'Tools',
      picks_kicker: '02 / RECENT PAPERS & PROJECTS',
      picks_title: 'Recent Papers & Projects',
      picks_subtitle:
        'Default ~10 curated picks prioritizing the past 24 hours (explicitly extended to 7 days when needed) across Embodied AI, Humanoid, VLA, WAM, RL, IL, WBC, Dexterous Manipulation, Sim, Teleop, and SLAM. Deduplicated with direct KB links.',
      kind_all: 'All',
      kind_paper: 'Papers',
      kind_project: 'Projects',
      kind_model: 'Models',
      kind_dataset: 'Datasets',
      copy_list: 'Copy Plain-Text List',
      copy_item: 'Copy',
      copied_item: 'Copied',
      topic_all: 'All Topics',
      loading_sources: 'Loading recommended websites…',
      loading_picks: 'Loading curated picks…',
      sources_error: 'Could not load website directory configuration. Please refresh.',
      picks_error_title: 'Curated picks temporarily unavailable',
      picks_error_note:
        'Failed to load curated picks, while the recommended websites directory above remains fully operational.',
      empty_sources: 'No websites under this category.',
      empty_picks: 'No curated picks match the selected filters.',
      best_for_label: 'Best for',
      cadence_label: 'Feed / Cadence',
      visit_site: 'Visit Site',
      tool_badge: 'Tracking Tool · Not a Ranking',
      date_published: 'Published',
      date_updated: 'Updated',
      window_24h: '24h',
      window_7d: '7d',
      link_paper: 'Paper',
      link_project: 'Project',
      link_code: 'Code',
      link_model: 'Model',
      link_dataset: 'Dataset',
      kb_detail: 'KB Detail',
      status_updated: 'Updated',
      status_attempt: 'Last attempt',
      status_stale_badge: 'Showing last valid snapshot',
      status_partial_badge: 'Partial source outage',
      status_ok_badge: 'Verified & synced',
      copied_count_prefix: 'Copied ',
      copied_count_suffix: ' items with primary links as plain text',
      copied_single_prefix: 'Copied: ',
      copy_empty: 'No visible picks to copy.',
      copy_failed: 'Clipboard access blocked; please copy manually.',
      footer_author: 'Chong Liu',
      footer_tail: '2026 · Robotics Notebooks',
      lang_btn_label: '中文',
      lang_btn_aria: '切换页面为中文',
      telemetry_window_24h: '24h',
      telemetry_window_7d: '24h + 7d'
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
    var trimmed = raw.trim();
    if (/^detail\.html\?id=[A-Za-z0-9_.:-]+$/.test(trimmed)) return trimmed;
    try {
      var parsed = new URL(trimmed);
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

  function formatTimestamp(isoString, fallbackDate) {
    if (!isoString && !fallbackDate) return '—';
    var raw = String(isoString || fallbackDate || '').trim();
    var m = raw.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::\d{2})?([+-]\d{2}:\d{2}|Z)?$/);
    if (m) {
      var tz = m[3] === '+08:00' ? 'UTC+8' : (m[3] || '');
      return m[1] + ' ' + m[2] + (tz ? ' ' + tz : '');
    }
    return raw;
  }

  function gridClassForCount(count) {
    if (count <= 1) return 'discovery-site-grid discovery-site-grid--single';
    if (count === 2 || count === 4) return 'discovery-site-grid discovery-site-grid--2col';
    return 'discovery-site-grid discovery-site-grid--3col';
  }

  function formatItemLine(item, index, lang) {
    var isEn = lang === 'en';
    var kindMap = {
      paper: isEn ? 'Paper' : '论文',
      project: isEn ? 'Project' : '项目',
      model: isEn ? 'Model' : '模型',
      dataset: isEn ? 'Dataset' : '数据集'
    };
    var kindLabel = kindMap[item.kind] || item.kind || '';
    var titlePrimary = isEn ? (item.title_en || item.title_zh || '') : (item.title_zh || item.title_en || '');
    var titleSecondary = isEn ? (item.title_zh || '') : (item.title_en || '');
    var titleText = titleSecondary && titleSecondary !== titlePrimary
      ? titlePrimary + ' | ' + titleSecondary
      : titlePrimary;
    var dateWord = item.date_type === 'updated'
      ? (isEn ? 'Updated' : '更新')
      : (isEn ? 'Published' : '发布');
    var header = (index != null ? (index + '. ') : '') + '[' + kindLabel + '] ' + titleText +
      ' (' + dateWord + ': ' + (item.date || '') + (item.team ? ', ' + item.team : '') + ')';

    var lines = [header];
    var linkOrder = [
      ['paper', isEn ? 'Paper' : '论文'],
      ['project', isEn ? 'Project' : '主页'],
      ['code', isEn ? 'Code' : '源码'],
      ['model', isEn ? 'Model' : '模型'],
      ['dataset', isEn ? 'Dataset' : '数据集']
    ];
    var links = item.links || {};
    linkOrder.forEach(function (pair) {
      var url = safeUrl(links[pair[0]]);
      if (url) lines.push('   - ' + pair[1] + ': ' + url);
    });
    if (item.kb_node && item.kb_node.url) {
      var kbRel = safeUrl(item.kb_node.url);
      if (kbRel) lines.push('   - KB: ' + BASE_SITE_URL + kbRel);
    }
    return lines.join('\n');
  }

  function formatPlainTextChecklist(items, picksData, lang) {
    var isEn = lang === 'en';
    var ts = formatTimestamp(picksData && picksData.updated_at, picksData && picksData.updated_date_bjt);
    var header = isEn
      ? 'Robotics Discovery Curated Picks (Updated: ' + ts + ', ' + items.length + ' items)'
      : 'Robotics Discovery 最新精选清单（更新：' + ts + '，共 ' + items.length + ' 项）';
    var blocks = items.map(function (item, idx) {
      return formatItemLine(item, idx + 1, lang);
    });
    return header + '\n\n' + blocks.join('\n\n');
  }

  function initDiscoveryPage(doc, win) {
    doc = doc || (typeof document !== 'undefined' ? document : null);
    win = win || (typeof window !== 'undefined' ? window : null);
    if (!doc || !win) return null;

    var state = {
      lang: 'zh',
      activeCategory: 'all',
      activeKind: 'all',
      activeTopic: 'all',
      sourcesData: null,
      sourcesError: false,
      picksData: null,
      picksError: false,
      lastCopyMsgKey: null
    };

    try {
      if (win.localStorage && win.localStorage.getItem(STORAGE_LANG_KEY) === 'en') {
        state.lang = 'en';
      }
    } catch (_err) { /* ignore storage restriction */ }

    var els = {
      langBtn: doc.getElementById('discoveryLanguageToggle'),
      sourcesMount: doc.getElementById('discoverySourcesMount'),
      picksMount: doc.getElementById('discoveryPicksMount'),
      statusBar: doc.getElementById('discoveryStatusBar'),
      topicBar: doc.getElementById('discoveryTopicBar'),
      categoryTabs: doc.getElementById('discoveryCategoryTabs'),
      kindTabs: doc.getElementById('discoveryKindTabs'),
      copyBtn: doc.getElementById('copyDiscoveryListBtn'),
      copyStatus: doc.getElementById('copyDiscoveryStatus'),
      statSources: doc.getElementById('discoveryStatSources'),
      statPicks: doc.getElementById('discoveryStatPicks'),
      statWindow: doc.getElementById('discoveryStatWindow'),
      statUpdated: doc.getElementById('discoveryStatUpdated')
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

      if (els.copyStatus && !els.copyStatus.hidden && state.lastCopyMsgKey) {
        if (state.lastCopyMsgKey.type === 'all') {
          els.copyStatus.textContent = t('copied_count_prefix') + state.lastCopyMsgKey.count + t('copied_count_suffix');
        } else if (state.lastCopyMsgKey.type === 'single') {
          els.copyStatus.textContent = t('copied_single_prefix') + state.lastCopyMsgKey.title;
        } else if (state.lastCopyMsgKey.type === 'empty') {
          els.copyStatus.textContent = t('copy_empty');
        } else if (state.lastCopyMsgKey.type === 'failed') {
          els.copyStatus.textContent = t('copy_failed');
        }
      }
    }

    function renderSiteCard(site) {
      var isEn = state.lang === 'en';
      var name = isEn ? (site.name_en || site.name_zh) : (site.name_zh || site.name_en);
      var summary = isEn ? (site.summary_en || site.summary_zh) : (site.summary_zh || site.summary_en);
      var bestFor = isEn ? (site.best_for_en || site.best_for_zh) : (site.best_for_zh || site.best_for_en);
      var cadence = isEn ? (site.cadence_en || site.cadence_zh) : (site.cadence_zh || site.cadence_en);
      var kindBadge = isEn ? (site.kind_badge_en || site.kind_badge_zh) : (site.kind_badge_zh || site.kind_badge_en);
      var href = safeUrl(site.url);
      var toolTag = site.is_supplement_tool
        ? '<span class="discovery-badge discovery-badge--warn">' + escapeHtml(t('tool_badge')) + '</span>'
        : '<span class="discovery-badge">' + escapeHtml(kindBadge) + '</span>';

      return (
        '<article class="discovery-site-card' + (site.is_supplement_tool ? ' is-tool-card' : '') + '">' +
          '<div class="discovery-site-card-top">' +
            '<span class="discovery-site-domain">' + escapeHtml(site.domain || '') + '</span>' +
            toolTag +
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

    function renderSources() {
      if (!els.sourcesMount) return;
      els.sourcesMount.classList.remove('data-loading');
      if (state.sourcesError) {
        els.sourcesMount.innerHTML = '<div class="discovery-banner discovery-banner--error">' + escapeHtml(t('sources_error')) + '</div>';
        return;
      }
      if (!state.sourcesData) return;

      var categories = Array.isArray(state.sourcesData.categories) ? state.sourcesData.categories : [];
      var sites = Array.isArray(state.sourcesData.sites) ? state.sourcesData.sites : [];
      if (els.statSources) els.statSources.textContent = String(sites.length);

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

    function filteredPicks() {
      var items = (state.picksData && Array.isArray(state.picksData.items)) ? state.picksData.items : [];
      return items.filter(function (item) {
        if (state.activeKind !== 'all' && item.kind !== state.activeKind) return false;
        if (state.activeTopic !== 'all') {
          var topics = Array.isArray(item.topics) ? item.topics : [];
          if (topics.indexOf(state.activeTopic) === -1) return false;
        }
        return true;
      });
    }

    function renderTelemetryPicks() {
      if (state.picksError || !state.picksData) {
        if (els.statPicks) els.statPicks.textContent = '0';
        if (els.statWindow) els.statWindow.textContent = '—';
        if (els.statUpdated) els.statUpdated.textContent = '—';
        return;
      }
      var items = Array.isArray(state.picksData.items) ? state.picksData.items : [];
      var winMeta = state.picksData.window || {};
      if (els.statPicks) els.statPicks.textContent = String(items.length);
      if (els.statWindow) {
        els.statWindow.textContent = winMeta.expanded ? t('telemetry_window_7d') : t('telemetry_window_24h');
      }
      if (els.statUpdated) {
        els.statUpdated.textContent = state.picksData.updated_date_bjt || formatTimestamp(state.picksData.updated_at, '');
      }
    }

    function renderStatusBar() {
      if (!els.statusBar) return;
      if (state.picksError) {
        els.statusBar.innerHTML =
          '<div class="discovery-banner discovery-banner--error">' +
            '<span class="discovery-badge discovery-badge--warn">' + escapeHtml(t('picks_error_title')) + '</span>' +
            '<span>' + escapeHtml(t('picks_error_note')) + '</span>' +
          '</div>';
        return;
      }
      if (!state.picksData) {
        els.statusBar.innerHTML = '';
        return;
      }
      var status = state.picksData.status || {};
      var winMeta = state.picksData.window || {};
      var isEn = state.lang === 'en';
      var stState = status.state || 'ok';
      var badgeCls = stState === 'ok' ? 'discovery-badge discovery-badge--accent' : 'discovery-badge discovery-badge--warn';
      var badgeText = stState === 'stale'
        ? t('status_stale_badge')
        : (stState === 'partial' ? t('status_partial_badge') : t('status_ok_badge'));
      var msg = isEn ? (status.message_en || status.message_zh || '') : (status.message_zh || status.message_en || '');
      var winNote = isEn ? (winMeta.note_en || winMeta.note_zh || '') : (winMeta.note_zh || winMeta.note_en || '');
      var updatedLabel = formatTimestamp(state.picksData.updated_at, state.picksData.updated_date_bjt);
      var attemptLabel = formatTimestamp(status.last_attempt_at, '');

      var bannerCls = stState === 'stale'
        ? 'discovery-banner discovery-banner--warn'
        : (stState === 'partial' ? 'discovery-banner discovery-banner--partial' : 'discovery-banner');

      els.statusBar.innerHTML =
        '<div class="' + bannerCls + '">' +
          '<div class="discovery-status-row">' +
            '<span class="' + badgeCls + '">' + escapeHtml(badgeText) + '</span>' +
            '<span class="discovery-status-time">' + escapeHtml(t('status_updated')) + ': <strong>' + escapeHtml(updatedLabel) + '</strong></span>' +
            (stState === 'stale' && attemptLabel && attemptLabel !== updatedLabel
              ? '<span class="discovery-status-time">' + escapeHtml(t('status_attempt')) + ': ' + escapeHtml(attemptLabel) + '</span>'
              : '') +
          '</div>' +
          (winNote ? '<p class="discovery-status-note">' + escapeHtml(winNote) + '</p>' : '') +
          (msg ? '<p class="discovery-status-note">' + escapeHtml(msg) + '</p>' : '') +
        '</div>';
    }

    function renderTopicBar() {
      if (!els.topicBar) return;
      if (state.picksError || !state.picksData || !Array.isArray(state.picksData.items) || !state.picksData.items.length) {
        els.topicBar.innerHTML = '';
        els.topicBar.hidden = true;
        return;
      }
      var seen = Object.create(null);
      var topics = [];
      state.picksData.items.forEach(function (item) {
        (item.topics || []).forEach(function (tp) {
          if (!seen[tp]) {
            seen[tp] = true;
            topics.push(tp);
          }
        });
      });
      if (!topics.length) {
        els.topicBar.innerHTML = '';
        els.topicBar.hidden = true;
        return;
      }
      els.topicBar.hidden = false;
      var html = '<button type="button" class="tag-chip' + (state.activeTopic === 'all' ? ' is-active' : '') +
        '" data-topic="all" aria-pressed="' + (state.activeTopic === 'all' ? 'true' : 'false') + '">' +
        escapeHtml(t('topic_all')) + '</button>';
      html += topics.map(function (tp) {
        var active = state.activeTopic === tp;
        return '<button type="button" class="tag-chip' + (active ? ' is-active' : '') +
          '" data-topic="' + escapeHtml(tp) + '" aria-pressed="' + (active ? 'true' : 'false') + '">' +
          escapeHtml(tp) + '</button>';
      }).join('');
      els.topicBar.innerHTML = html;
    }

    function renderPickCard(item, idx) {
      var isEn = state.lang === 'en';
      var kindMap = {
        paper: t('kind_paper'),
        project: t('kind_project'),
        model: t('kind_model'),
        dataset: t('kind_dataset')
      };
      var kindLabel = kindMap[item.kind] || item.kind || '';
      var primaryTitle = isEn ? (item.title_en || item.title_zh || '') : (item.title_zh || item.title_en || '');
      var secondaryTitle = isEn ? (item.title_zh || '') : (item.title_en || '');
      var showSecondary = Boolean(secondaryTitle && secondaryTitle !== primaryTitle);
      var contrib = isEn ? (item.contribution_en || item.contribution_zh || '') : (item.contribution_zh || item.contribution_en || '');
      var dateWord = item.date_type === 'updated' ? t('date_updated') : t('date_published');
      var winLabel = item.window_bucket === '24h' ? t('window_24h') : t('window_7d');
      var num = String(idx + 1).padStart(2, '0');

      var topicsHtml = (item.topics || []).map(function (tp) {
        return '<span class="discovery-topic-pill">' + escapeHtml(tp) + '</span>';
      }).join('');

      var signalsHtml = (item.signals || []).map(function (sig) {
        var label = isEn ? (sig.label_en || sig.label_zh || '') : (sig.label_zh || sig.label_en || '');
        if (!label) return '';
        var obs = sig.observed_at ? ' (' + sig.observed_at + ')' : '';
        return '<span class="discovery-signal-pill">' + escapeHtml(label + obs) + '</span>';
      }).join('');

      var links = item.links || {};
      var linkOrder = [
        ['paper', t('link_paper')],
        ['project', t('link_project')],
        ['code', t('link_code')],
        ['model', t('link_model')],
        ['dataset', t('link_dataset')]
      ];
      var linkBtns = linkOrder.map(function (pair) {
        var href = safeUrl(links[pair[0]]);
        if (!href) return '';
        return '<a class="discovery-pick-link" href="' + escapeHtml(href) + '" target="_blank" rel="noopener noreferrer">' +
          '<span>' + escapeHtml(pair[1]) + '</span><span aria-hidden="true">↗</span></a>';
      }).join('');

      var kbHtml = '';
      if (item.kb_node && item.kb_node.url) {
        var kbHref = safeUrl(item.kb_node.url);
        if (kbHref) {
          kbHtml = '<a class="discovery-pick-kb" href="' + escapeHtml(kbHref) + '">' +
            '<span>' + escapeHtml(t('kb_detail')) + '</span><span aria-hidden="true">→</span></a>';
        }
      }

      return (
        '<article class="discovery-pick-card" data-pick-id="' + escapeHtml(item.id || '') + '">' +
          '<div class="discovery-pick-head">' +
            '<div class="discovery-pick-badges">' +
              '<span class="discovery-pick-index">' + num + '</span>' +
              '<span class="discovery-badge">' + escapeHtml(kindLabel) + '</span>' +
              '<span class="discovery-badge ' + (item.window_bucket === '24h' ? 'discovery-badge--accent' : '') + '">' + escapeHtml(winLabel) + '</span>' +
            '</div>' +
            '<div class="discovery-pick-date-team">' +
              '<time datetime="' + escapeHtml(item.date || '') + '">' + escapeHtml(dateWord + ' ' + (item.date || '')) + '</time>' +
              (item.team ? '<span class="discovery-pick-team">' + escapeHtml(item.team) + '</span>' : '') +
            '</div>' +
          '</div>' +
          '<h3 class="discovery-pick-title">' + escapeHtml(primaryTitle) + '</h3>' +
          (showSecondary ? '<p class="discovery-pick-subtitle">' + escapeHtml(secondaryTitle) + '</p>' : '') +
          '<p class="discovery-pick-contrib">' + escapeHtml(contrib) + '</p>' +
          (topicsHtml || signalsHtml
            ? '<div class="discovery-pick-tags">' + topicsHtml + signalsHtml + '</div>'
            : '') +
          '<div class="discovery-pick-foot">' +
            '<div class="discovery-pick-links">' + linkBtns + kbHtml + '</div>' +
            '<button type="button" class="discovery-item-copy" data-copy-id="' + escapeHtml(item.id || '') + '">' +
              iconSvg('copy') +
              '<span>' + escapeHtml(t('copy_item')) + '</span>' +
            '</button>' +
          '</div>' +
        '</article>'
      );
    }

    function renderPicks() {
      renderTelemetryPicks();
      renderStatusBar();
      renderTopicBar();
      if (!els.picksMount) return;
      els.picksMount.classList.remove('data-loading');
      if (state.picksError) {
        els.picksMount.innerHTML = '';
        return;
      }
      if (!state.picksData) return;
      var items = filteredPicks();
      if (!items.length) {
        els.picksMount.innerHTML = '<p class="data-meta">' + escapeHtml(t('empty_picks')) + '</p>';
        return;
      }
      els.picksMount.innerHTML = '<div class="discovery-picks-list">' + items.map(renderPickCard).join('') + '</div>';
    }

    function renderAll() {
      applyStaticI18n();
      renderSources();
      renderPicks();
    }

    function execCommandCopy(text) {
      return new Promise(function (resolve, reject) {
        try {
          var ta = doc.createElement('textarea');
          ta.value = text;
          ta.setAttribute('readonly', '');
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          doc.body.appendChild(ta);
          ta.select();
          var ok = typeof doc.execCommand === 'function' && doc.execCommand('copy');
          doc.body.removeChild(ta);
          if (ok) resolve();
          else reject(new Error('execCommand failed'));
        } catch (err) {
          reject(err);
        }
      });
    }

    function copyTextToClipboard(text) {
      if (win.navigator && win.navigator.clipboard && typeof win.navigator.clipboard.writeText === 'function') {
        return win.navigator.clipboard.writeText(text).catch(function () {
          return execCommandCopy(text);
        });
      }
      return execCommandCopy(text);
    }

    function showCopyMessage(msg, meta) {
      state.lastCopyMsgKey = meta || null;
      if (!els.copyStatus) return;
      els.copyStatus.textContent = msg;
      els.copyStatus.hidden = false;
    }

    // Event bindings
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

    if (els.kindTabs) {
      els.kindTabs.addEventListener('click', function (ev) {
        var btn = ev.target.closest('button[data-kind]');
        if (!btn) return;
        state.activeKind = btn.getAttribute('data-kind') || 'all';
        els.kindTabs.querySelectorAll('button[data-kind]').forEach(function (tab) {
          var active = tab === btn;
          tab.classList.toggle('is-active', active);
          tab.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
        renderPicks();
      });
    }

    if (els.topicBar) {
      els.topicBar.addEventListener('click', function (ev) {
        var btn = ev.target.closest('button[data-topic]');
        if (!btn) return;
        state.activeTopic = btn.getAttribute('data-topic') || 'all';
        renderPicks();
      });
    }

    if (els.copyBtn) {
      els.copyBtn.addEventListener('click', function () {
        var items = filteredPicks();
        if (!items.length) {
          showCopyMessage(t('copy_empty'), { type: 'empty' });
          return;
        }
        var text = formatPlainTextChecklist(items, state.picksData, state.lang);
        copyTextToClipboard(text)
          .then(function () {
            showCopyMessage(
              t('copied_count_prefix') + items.length + t('copied_count_suffix'),
              { type: 'all', count: items.length }
            );
          })
          .catch(function () {
            showCopyMessage(t('copy_failed'), { type: 'failed' });
          });
      });
    }

    if (els.picksMount) {
      els.picksMount.addEventListener('click', function (ev) {
        var btn = ev.target.closest('button[data-copy-id]');
        if (!btn || !state.picksData || !Array.isArray(state.picksData.items)) return;
        var id = btn.getAttribute('data-copy-id');
        var item = state.picksData.items.find(function (x) { return x.id === id; });
        if (!item) return;
        var text = formatItemLine(item, null, state.lang);
        copyTextToClipboard(text)
          .then(function () {
            var span = btn.querySelector('span');
            if (span) {
              span.textContent = t('copied_item');
              win.setTimeout(function () { span.textContent = t('copy_item'); }, 1800);
            }
            var title = state.lang === 'en' ? (item.title_en || item.title_zh) : (item.title_zh || item.title_en);
            showCopyMessage(t('copied_single_prefix') + title, { type: 'single', title: title });
          })
          .catch(function () {
            showCopyMessage(t('copy_failed'), { type: 'failed' });
          });
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

    function loadPicks() {
      return win.fetch('discovery-picks.json')
        .then(function (resp) {
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          return resp.json();
        })
        .then(function (data) {
          state.picksData = data;
          state.picksError = false;
          renderPicks();
          return data;
        })
        .catch(function () {
          state.picksError = true;
          renderPicks();
          return null;
        });
    }

    applyStaticI18n();
    var sourcesPromise = loadSources();
    var picksPromise = loadPicks();

    return {
      state: state,
      renderAll: renderAll,
      ready: Promise.all([sourcesPromise, picksPromise])
    };
  }

  var api = {
    I18N: I18N,
    escapeHtml: escapeHtml,
    safeUrl: safeUrl,
    formatTimestamp: formatTimestamp,
    gridClassForCount: gridClassForCount,
    formatItemLine: formatItemLine,
    formatPlainTextChecklist: formatPlainTextChecklist,
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
