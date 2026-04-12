(function () {
  const root = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');
  const key = 'robotics-notebooks-theme';
  const saved = localStorage.getItem(key);
  const preferDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = saved ? saved === 'dark' : preferDark;
  root.setAttribute('data-theme', dark ? 'dark' : 'light');

  function updateThemeToggle() {
    if (!themeToggle) return;
    const isDark = root.getAttribute('data-theme') === 'dark';
    themeToggle.setAttribute('title', isDark ? '切换到白天模式' : '切换到黑夜模式');
    themeToggle.setAttribute('aria-label', isDark ? '切换到白天模式' : '切换到黑夜模式');
  }

  updateThemeToggle();

  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      const isDark = root.getAttribute('data-theme') === 'dark';
      root.setAttribute('data-theme', isDark ? 'light' : 'dark');
      localStorage.setItem(key, isDark ? 'light' : 'dark');
      updateThemeToggle();
    });
  }

  const links = document.querySelectorAll('.page-subnav a, .main-nav a');
  const sections = Array.from(links)
    .map(link => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  function updateActive() {
    const scrollPos = window.scrollY + 100;
    let currentId = sections.length ? '#' + sections[0].id : '';
    sections.forEach(section => {
      if (section.offsetTop <= scrollPos) currentId = '#' + section.id;
    });
    links.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === currentId);
    });
  }

  if (sections.length) {
    window.addEventListener('scroll', updateActive);
    updateActive();
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function removeLoadingState(element) {
    if (element) element.classList.remove('data-loading');
  }

  function renderChipList(container, items, options) {
    if (!container) return;
    const renderItem = (options && options.renderItem) || function (item) {
      return '<span class="data-chip">' + escapeHtml(item) + '</span>';
    };
    if (!Array.isArray(items) || !items.length) {
      container.innerHTML = '<p class="data-meta">暂无数据</p>';
      removeLoadingState(container);
      return;
    }
    container.innerHTML = items.map(renderItem).join('');
    removeLoadingState(container);
  }

  function renderPreviewPage(siteData) {
    if (!siteData || !siteData.pages) return;

    const pages = siteData.pages;
    const homePage = pages.home_page || {};
    const modulePages = pages.module_pages || {};
    const roadmapPages = pages.roadmap_pages || {};
    const techMapPage = pages.tech_map_page || {};
    const detailPages = pages.detail_pages || {};

    const summary = document.getElementById('previewSummary');
    if (summary) {
      const moduleCount = Object.keys(modulePages).length;
      const roadmapCount = Object.keys(roadmapPages).length;
      const detailCount = Object.keys(detailPages).length;
      const nodeCount = Array.isArray(techMapPage.nodes) ? techMapPage.nodes.length : 0;
      summary.innerHTML = [
        '<article class="card kpi-card">',
        '  <div class="kpi-value">' + moduleCount + '</div>',
        '  <div class="kpi-label">模块页数量</div>',
        '  <p class="kpi-note">当前聚合模块：' + escapeHtml(Object.keys(modulePages).join(' / ')) + '</p>',
        '</article>',
        '<article class="card kpi-card">',
        '  <div class="kpi-value">' + roadmapCount + '</div>',
        '  <div class="kpi-label">路线页数量</div>',
        '  <p class="kpi-note">已覆盖 route 与 learning path，可直接生成路线入口。</p>',
        '</article>',
        '<article class="card kpi-card">',
        '  <div class="kpi-value">' + detailCount + '</div>',
        '  <div class="kpi-label">详情页对象数量</div>',
        '  <p class="kpi-note">detail_pages 已可支持第一阶段通用详情页渲染。</p>',
        '</article>',
        '<article class="card kpi-card">',
        '  <div class="kpi-value">' + nodeCount + '</div>',
        '  <div class="kpi-label">tech-map 节点数量</div>',
        '  <p class="kpi-note">说明 tech_map_page 已经不只是说明文档，而是可消费的数据页。</p>',
        '</article>'
      ].join('');
      removeLoadingState(summary);
    }

    const hero = document.getElementById('homeHeroPreview');
    if (hero) {
      const title = homePage.hero && homePage.hero.title ? homePage.hero.title : '未提供标题';
      const subtitle = homePage.hero && homePage.hero.subtitle ? homePage.hero.subtitle : '未提供副标题';
      hero.innerHTML = [
        '<h4>' + escapeHtml(title) + '</h4>',
        '<p class="data-meta">' + escapeHtml(subtitle) + '</p>',
        '<p class="data-submeta">当前首页 CTA、quick entries、featured chain、featured modules 都能直接从聚合导出中拿到。</p>'
      ].join('');
      removeLoadingState(hero);
    }

    const quickEntries = document.getElementById('quickEntriesPreview');
    if (quickEntries) {
      const entries = Array.isArray(homePage.quick_entries) ? homePage.quick_entries : [];
      quickEntries.innerHTML = entries.length
        ? entries.map(function (item) {
            const page = roadmapPages[item] || detailPages[item] || {};
            return '<li><strong>' + escapeHtml(page.title || item) + '</strong><br /><small>' + escapeHtml(item) + '</small></li>';
          }).join('')
        : '<li>暂无快速入口数据</li>';
      removeLoadingState(quickEntries);
    }

    renderChipList(document.getElementById('featuredChainPreview'), homePage.featured_chain, {
      renderItem: function (item) {
        const page = detailPages[item] || {};
        return '<span class="data-chip" title="' + escapeHtml(item) + '">' + escapeHtml(page.title || item) + '</span>';
      }
    });

    renderChipList(document.getElementById('featuredModulesPreview'), homePage.featured_modules, {
      renderItem: function (item) {
        const page = modulePages[item] || {};
        return '<span class="data-chip" title="' + escapeHtml(item) + '">' + escapeHtml(page.title || item) + '</span>';
      }
    });

    const moduleGrid = document.getElementById('modulePreviewGrid');
    if (moduleGrid) {
      const moduleCards = Object.values(modulePages).map(function (modulePage) {
        const references = Array.isArray(modulePage.references) ? modulePage.references.length : 0;
        const roadmaps = Array.isArray(modulePage.roadmaps) ? modulePage.roadmaps.length : 0;
        const entries = Array.isArray(modulePage.entry_items) ? modulePage.entry_items.slice(0, 4) : [];
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3>' + escapeHtml(modulePage.title || modulePage.module_id || '未命名模块') + '</h3>',
          '    <p class="card-meta">tag: ' + escapeHtml(modulePage.tag || '-') + '</p>',
          '    <p>' + escapeHtml(modulePage.summary || '暂无模块摘要') + '</p>',
          '  </div>',
          '  <div class="chip-list">',
          '    <span class="data-chip">入口 ' + escapeHtml((modulePage.entry_items || []).length) + '</span>',
          '    <span class="data-chip">参考 ' + escapeHtml(references) + '</span>',
          '    <span class="data-chip">路线 ' + escapeHtml(roadmaps) + '</span>',
          '  </div>',
          '  <ul>',
               entries.map(function (item) { return '    <li><code>' + escapeHtml(item) + '</code></li>'; }).join(''),
          '  </ul>',
          '</article>'
        ].join('');
      });
      moduleGrid.innerHTML = moduleCards.length ? moduleCards.join('') : '<article class="card"><p>暂无模块页数据</p></article>';
      removeLoadingState(moduleGrid);
    }

    const roadmapGrid = document.getElementById('roadmapPreviewGrid');
    if (roadmapGrid) {
      const roadmapCards = Object.entries(roadmapPages).map(function (entry) {
        const roadmapId = entry[0];
        const roadmapPage = entry[1] || {};
        const stages = Array.isArray(roadmapPage.stages) ? roadmapPage.stages : [];
        const related = Array.isArray(roadmapPage.related_items) ? roadmapPage.related_items : [];
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3>' + escapeHtml(roadmapPage.title || roadmapId) + '</h3>',
          '    <p class="card-meta">' + escapeHtml(roadmapId) + '</p>',
          '    <p>' + escapeHtml(roadmapPage.summary || '暂无路线摘要') + '</p>',
          '  </div>',
          '  <div class="chip-list">',
          '    <span class="data-chip">阶段 ' + escapeHtml(stages.length) + '</span>',
          '    <span class="data-chip">关联项 ' + escapeHtml(related.length) + '</span>',
          '  </div>',
          '  <ul>',
               stages.slice(0, 4).map(function (stage) { return '    <li>' + escapeHtml(stage.title || stage.id || '未命名阶段') + '</li>'; }).join(''),
          '  </ul>',
          '</article>'
        ].join('');
      });
      roadmapGrid.innerHTML = roadmapCards.length ? roadmapCards.join('') : '<article class="card"><p>暂无路线页数据</p></article>';
      removeLoadingState(roadmapGrid);
    }
  }

  const previewRoot = document.getElementById('previewSummary');
  if (previewRoot) {
    fetch('../exports/site-data-v1.json')
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        return response.json();
      })
      .then(renderPreviewPage)
      .catch(function (error) {
        ['previewSummary', 'homeHeroPreview', 'quickEntriesPreview', 'featuredChainPreview', 'featuredModulesPreview', 'modulePreviewGrid', 'roadmapPreviewGrid']
          .map(function (id) { return document.getElementById(id); })
          .filter(Boolean)
          .forEach(function (element) {
            element.innerHTML = '<p class="data-meta">读取 <code>../exports/site-data-v1.json</code> 失败：' + escapeHtml(error.message) + '</p>';
            removeLoadingState(element);
          });
      });
  }
})();
