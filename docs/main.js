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
    .map(function (link) { return document.querySelector(link.getAttribute('href')); })
    .filter(Boolean);

  function updateActive() {
    const scrollPos = window.scrollY + 100;
    let currentId = sections.length ? '#' + sections[0].id : '';
    sections.forEach(function (section) {
      if (section.offsetTop <= scrollPos) currentId = '#' + section.id;
    });
    links.forEach(function (link) {
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

  function detailHref(id) {
    return 'detail.html?id=' + encodeURIComponent(id);
  }

  function moduleHref(id) {
    return 'module.html?id=' + encodeURIComponent(id);
  }

  function roadmapHref(id) {
    return 'roadmap.html?id=' + encodeURIComponent(id);
  }

  function renderInlineMarkdown(text) {
    return escapeHtml(text || '')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, function (_, label, url) {
        return '<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
      })
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  }

  function renderMathBlocks(text) {
    return String(text || '')
      .replace(/\\\((.+?)\\\)/g, function (_, expr) {
        return '<span class="math-inline">\\(' + expr + '\\)</span>';
      })
      .replace(/\$\$([\s\S]+?)\$\$/g, function (_, expr) {
        return '<div class="math-block">$$' + expr.trim() + '$$</div>';
      });
  }

  function slugifyHeading(text) {
    const normalized = String(text || '')
      .toLowerCase()
      .replace(/<[^>]+>/g, '')
      .replace(/[^\p{Letter}\p{Number}\s-]+/gu, ' ')
      .trim()
      .replace(/\s+/g, '-');
    return normalized || 'section';
  }

  function collectMarkdownHeadings(markdown) {
    const source = String(markdown || '').replace(/\r\n/g, '\n').trim();
    if (!source) return [];
    const counts = {};
    return source.split('\n').map(function (line) {
      const match = line.trim().match(/^(#{2,4})\s+(.*)$/);
      if (!match) return null;
      const text = match[2].trim();
      const baseSlug = slugifyHeading(text);
      counts[baseSlug] = (counts[baseSlug] || 0) + 1;
      return {
        level: Math.min(match[1].length, 4),
        text: text,
        slug: counts[baseSlug] === 1 ? baseSlug : baseSlug + '-' + counts[baseSlug]
      };
    }).filter(Boolean);
  }

  function renderDetailToc(container, headings) {
    if (!container) return;
    if (!Array.isArray(headings) || !headings.length) {
      container.innerHTML = '<p class="data-meta">当前正文较短，暂不生成目录。</p>';
      removeLoadingState(container);
      return;
    }

    container.innerHTML = '<ol>' + headings.map(function (heading) {
      return '<li class="toc-level-' + escapeHtml(heading.level) + '"><a href="#' + escapeHtml(heading.slug) + '">' + escapeHtml(heading.text) + '</a></li>';
    }).join('') + '</ol>';
    removeLoadingState(container);
  }

  function renderMarkdownContent(markdown, headings) {
    const source = String(markdown || '').replace(/\r\n/g, '\n').trim();
    if (!source) {
      return '<p>当前 detail page 暂无可同步正文。</p>';
    }

    const lines = source.split('\n');
    const blocks = [];
    const headingQueue = Array.isArray(headings) ? headings.slice() : collectMarkdownHeadings(source);
    let paragraphLines = [];
    let listItems = [];
    let listTag = '';
    let quoteLines = [];
    let codeLines = [];
    let inCodeBlock = false;

    function flushParagraph() {
      if (!paragraphLines.length) return;
      blocks.push('<p>' + renderMathBlocks(renderInlineMarkdown(paragraphLines.join(' '))) + '</p>');
      paragraphLines = [];
    }

    function flushList() {
      if (!listItems.length) return;
      const openTag = listTag === 'ol' ? 'ol' : 'ul';
      blocks.push((function () {
        if (openTag === 'ul') return '<ul>';
        if (openTag === 'ol') return '<ol>';
        return '<ul>';
      })() + listItems.map(function (item) {
        return '<li>' + renderMathBlocks(renderInlineMarkdown(item)) + '</li>';
      }).join('') + '</' + openTag + '>');
      listItems = [];
      listTag = '';
    }

    function flushQuote() {
      if (!quoteLines.length) return;
      blocks.push((function () {
        return '<blockquote>';
      })() + quoteLines.map(function (line) {
        return '<p>' + renderMathBlocks(renderInlineMarkdown(line)) + '</p>';
      }).join('') + '</blockquote>');
      quoteLines = [];
    }

    function flushCodeBlock() {
      if (!codeLines.length) return;
      blocks.push((function () {
        return '<pre><code>';
      })() + escapeHtml(codeLines.join('\n')) + '</code></pre>');
      codeLines = [];
    }

    lines.forEach(function (line) {
      const trimmed = line.trim();

      if (trimmed.startsWith('```')) {
        if (inCodeBlock) {
          flushCodeBlock();
          inCodeBlock = false;
        } else {
          flushParagraph();
          flushList();
          flushQuote();
          inCodeBlock = true;
        }
        return;
      }

      if (inCodeBlock) {
        codeLines.push(line);
        return;
      }

      if (!trimmed) {
        flushParagraph();
        flushList();
        flushQuote();
        return;
      }

      const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (headingMatch) {
        flushParagraph();
        flushList();
        flushQuote();
        const level = Math.min(headingMatch[1].length, 6);
        const text = headingMatch[2].trim();
        const headingMeta = level >= 2 && headingQueue.length ? headingQueue.shift() : null;
        const headingId = headingMeta ? headingMeta.slug : slugifyHeading(text);
        blocks.push('<h' + level + ' id="' + escapeHtml(headingId) + '">' + renderMathBlocks(renderInlineMarkdown(text)) + '</h' + level + '>');
        return;
      }

      const quoteMatch = trimmed.match(/^>\s?(.*)$/);
      if (quoteMatch) {
        flushParagraph();
        flushList();
        quoteLines.push(quoteMatch[1]);
        return;
      }

      const unorderedMatch = trimmed.match(/^[-*]\s+(.*)$/);
      if (unorderedMatch) {
        flushParagraph();
        flushQuote();
        if (listTag && listTag !== 'ul') flushList();
        listTag = 'ul';
        listItems.push(unorderedMatch[1]);
        return;
      }

      const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (orderedMatch) {
        flushParagraph();
        flushQuote();
        if (listTag && listTag !== 'ol') flushList();
        listTag = 'ol';
        listItems.push(orderedMatch[1]);
        return;
      }

      flushList();
      flushQuote();
      paragraphLines.push(trimmed);
    });

    if (inCodeBlock) flushCodeBlock();
    flushParagraph();
    flushList();
    flushQuote();

    return blocks.join('');
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

  function renderInternalLinks(container, ids, detailPages, options) {
    if (!container) return;
    const emptyText = (options && options.emptyText) || '暂无内部关联项';
    if (!Array.isArray(ids) || !ids.length) {
      container.innerHTML = '<article class="card"><p>' + escapeHtml(emptyText) + '</p></article>';
      removeLoadingState(container);
      return;
    }

    container.innerHTML = ids.map(function (id) {
      const page = detailPages[id] || {};
      return [
        '<article class="card data-card">',
        '  <div>',
        '    <h3><a href="' + escapeHtml(detailHref(id)) + '">' + escapeHtml(page.title || id) + '</a></h3>',
        '    <p class="card-meta">' + escapeHtml(page.type || 'detail_page') + '</p>',
        '    <p>' + escapeHtml(page.summary || '当前关联项暂无摘要') + '</p>',
        '  </div>',
        '  <div class="chip-list">',
        '    <span class="data-chip"><code>' + escapeHtml(id) + '</code></span>',
        '    <a class="btn-secondary btn-inline" href="' + escapeHtml(detailHref(id)) + '">打开详情页</a>',
        '  </div>',
        '</article>'
      ].join('');
    }).join('');
    removeLoadingState(container);
  }

  function renderSourceCards(container, links, emptyText) {
    if (!container) return;
    if (!Array.isArray(links) || !links.length) {
      container.innerHTML = '<article class="card"><p>' + escapeHtml(emptyText || '暂无来源链接') + '</p></article>';
      removeLoadingState(container);
      return;
    }

    container.innerHTML = links.map(function (url) {
      return [
        '<article class="card data-card">',
        '  <div>',
        '    <h3><a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">打开来源</a></h3>',
        '    <p class="data-submeta"><code>' + escapeHtml(url) + '</code></p>',
        '  </div>',
        '</article>'
      ].join('');
    }).join('');
    removeLoadingState(container);
  }

  function renderDetailPage(siteData) {
    if (!siteData || !siteData.pages) return;

    const pages = siteData.pages;
    const detailPages = pages.detail_pages || {};
    const params = new URLSearchParams(window.location.search);
    const detailId = params.get('id') || '';
    const detailPage = detailId ? detailPages[detailId] : null;

    const titleEl = document.getElementById('detailTitle');
    const summaryEl = document.getElementById('detailSummary');
    const metaEl = document.getElementById('detailMeta');
    const tocSectionEl = document.getElementById('detailTocSection');
    const tocEl = document.getElementById('detailTocList');
    const contentSectionEl = document.getElementById('detailContentSection');
    const contentEl = document.getElementById('detailContent');
    const tagEl = document.getElementById('detailTagList');
    const relatedEl = document.getElementById('detailRelatedList');
    const sourceEl = document.getElementById('detailSourceList');
    const emptyState = document.getElementById('detailEmptyState');
    const breadcrumb = document.getElementById('detailBreadcrumb');

    if (!detailPage) {
      if (emptyState) emptyState.hidden = false;
      if (titleEl) titleEl.textContent = '未找到对应 detail page';
      if (summaryEl) {
        summaryEl.innerHTML = '请在 URL 里传入合法的 <code>?id=...</code>，例如 <code>detail.html?id=wiki-concepts-centroidal-dynamics</code>。';
        removeLoadingState(summaryEl);
      }
      if (metaEl) {
        metaEl.innerHTML = '<p class="data-meta">当前没有匹配到 detail_pages 项。</p>';
        removeLoadingState(metaEl);
      }
      if (tocSectionEl) tocSectionEl.hidden = true;
      if (tocEl) {
        tocEl.innerHTML = '';
        removeLoadingState(tocEl);
      }
      if (contentSectionEl) contentSectionEl.hidden = true;
      if (contentEl) {
        contentEl.textContent = '';
        removeLoadingState(contentEl);
      }
      renderChipList(tagEl, [], {});
      renderInternalLinks(relatedEl, [], detailPages, { emptyText: '当前无可展示的关联项。' });
      renderSourceCards(sourceEl, [], '当前无可展示的来源链接。');
      if (breadcrumb) removeLoadingState(breadcrumb);
      return;
    }

    if (emptyState) emptyState.hidden = true;
    document.title = (detailPage.title || detailId) + ' | Robotics Notebooks';

    if (titleEl) titleEl.textContent = detailPage.title || detailId;
    if (summaryEl) {
      summaryEl.innerHTML = escapeHtml(detailPage.summary || '当前页面暂无摘要，可先通过 tags / related / source links 继续导航。');
      removeLoadingState(summaryEl);
    }
    if (metaEl) {
      metaEl.innerHTML = [
        '<p><strong>id：</strong><code>' + escapeHtml(detailPage.id || detailId) + '</code></p>',
        '<p><strong>type：</strong>' + escapeHtml(detailPage.type || '-') + '</p>',
        '<p><strong>status：</strong>' + escapeHtml(detailPage.status || 'active') + '</p>',
        '<p><strong>path：</strong><code>' + escapeHtml(detailPage.path || '-') + '</code></p>'
      ].join('');
      removeLoadingState(metaEl);
    }
    if (breadcrumb) {
      breadcrumb.innerHTML = [
        '<a href="index.html">首页</a>',
        '<span>/</span>',
        '<a href="site-data-preview.html">页面级导出预览</a>',
        '<span>/</span>',
        '<span>' + escapeHtml(detailPage.title || detailId) + '</span>'
      ].join('');
      removeLoadingState(breadcrumb);
    }

    const contentMarkdown = detailPage.content_markdown || '';
    const detailHeadings = collectMarkdownHeadings(contentMarkdown);
    if (tocSectionEl) {
      tocSectionEl.hidden = !detailHeadings.length;
    }
    if (tocEl) {
      renderDetailToc(tocEl, collectMarkdownHeadings(contentMarkdown));
    }
    if (contentSectionEl) {
      contentSectionEl.hidden = !contentMarkdown;
    }
    if (contentEl) {
      contentEl.innerHTML = contentMarkdown ? renderMarkdownContent(contentMarkdown, detailHeadings) : '<p>当前 detail page 暂无可同步正文。</p>';
      removeLoadingState(contentEl);
    }

    renderChipList(tagEl, detailPage.tags, {
      renderItem: function (tag) {
        return '<span class="data-chip">' + escapeHtml(tag) + '</span>';
      }
    });
    renderInternalLinks(relatedEl, detailPage.related, detailPages, { emptyText: '当前 detail page 暂无 related。' });
    renderSourceCards(sourceEl, detailPage.source_links, '当前 detail page 暂无来源链接。');
  }

  function renderModulePage(siteData) {
    if (!siteData || !siteData.pages) return;

    const pages = siteData.pages;
    const modulePages = pages.module_pages || {};
    const detailPages = pages.detail_pages || {};
    const params = new URLSearchParams(window.location.search);
    const moduleId = params.get('id') || '';
    const modulePage = moduleId ? modulePages[moduleId] : null;

    const titleEl = document.getElementById('moduleTitle');
    const summaryEl = document.getElementById('moduleSummary');
    const metaEl = document.getElementById('moduleMeta');
    const entryEl = document.getElementById('moduleEntryList');
    const referenceEl = document.getElementById('moduleReferenceList');
    const roadmapEl = document.getElementById('moduleRoadmapList');
    const relatedModuleEl = document.getElementById('moduleRelatedModules');
    const emptyState = document.getElementById('moduleEmptyState');
    const breadcrumb = document.getElementById('moduleBreadcrumb');

    if (!modulePage) {
      if (emptyState) emptyState.hidden = false;
      if (titleEl) titleEl.textContent = '未找到对应 module page';
      if (summaryEl) {
        summaryEl.innerHTML = '请在 URL 里传入合法的 <code>?id=...</code>，例如 <code>module.html?id=control</code>。';
        removeLoadingState(summaryEl);
      }
      if (metaEl) {
        metaEl.innerHTML = '<p class="data-meta">当前没有匹配到 module_pages 项。</p>';
        removeLoadingState(metaEl);
      }
      renderInternalLinks(entryEl, [], detailPages, { emptyText: '当前无可展示的模块入口项。' });
      renderInternalLinks(referenceEl, [], detailPages, { emptyText: '当前无可展示的 references。' });
      renderInternalLinks(roadmapEl, [], detailPages, { emptyText: '当前无可展示的 roadmap 入口。' });
      renderChipList(relatedModuleEl, [], {});
      if (breadcrumb) removeLoadingState(breadcrumb);
      return;
    }

    if (emptyState) emptyState.hidden = true;
    document.title = (modulePage.title || moduleId) + ' | Robotics Notebooks';

    if (titleEl) titleEl.textContent = modulePage.title || moduleId;
    if (summaryEl) {
      summaryEl.innerHTML = escapeHtml(modulePage.summary || '当前模块暂无摘要。');
      removeLoadingState(summaryEl);
    }
    if (metaEl) {
      metaEl.innerHTML = [
        '<p><strong>module_id：</strong><code>' + escapeHtml(modulePage.module_id || moduleId) + '</code></p>',
        '<p><strong>tag：</strong>' + escapeHtml(modulePage.tag || '-') + '</p>',
        '<p><strong>入口项：</strong>' + escapeHtml((modulePage.entry_items || []).length) + '</p>',
        '<p><strong>深挖入口：</strong>' + escapeHtml((modulePage.references || []).length) + '</p>'
      ].join('');
      removeLoadingState(metaEl);
    }
    if (breadcrumb) {
      breadcrumb.innerHTML = [
        '<a href="index.html">首页</a>',
        '<span>/</span>',
        '<span>' + escapeHtml(modulePage.title || moduleId) + '</span>'
      ].join('');
      removeLoadingState(breadcrumb);
    }

    renderInternalLinks(entryEl, modulePage.entry_items, detailPages, { emptyText: '当前模块暂无入口项。' });
    renderInternalLinks(referenceEl, modulePage.references, detailPages, { emptyText: '当前模块暂无 references。' });
    if (roadmapEl) {
      const roadmapPages = pages.roadmap_pages || {};
      roadmapEl.innerHTML = Array.isArray(modulePage.roadmaps) && modulePage.roadmaps.length ? modulePage.roadmaps.map(function (id) {
        const page = roadmapPages[id] || {};
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3><a href="' + escapeHtml(roadmapHref(id)) + '">' + escapeHtml(page.title || id) + '</a></h3>',
          '    <p class="card-meta">roadmap_page</p>',
          '    <p>' + escapeHtml(page.summary || '当前路线暂无摘要') + '</p>',
          '  </div>',
          '  <div class="chip-list">',
          '    <span class="data-chip"><code>' + escapeHtml(id) + '</code></span>',
          '    <a class="btn-secondary btn-inline" href="' + escapeHtml(roadmapHref(id)) + '">打开路线页</a>',
          '  </div>',
          '</article>'
        ].join('');
      }).join('') : '<article class="card"><p>当前模块暂无 roadmap 入口。</p></article>';
      removeLoadingState(roadmapEl);
    }
    renderChipList(relatedModuleEl, modulePage.related_modules, {
      renderItem: function (id) {
        const relatedModule = modulePages[id] || {};
        return '<a class="data-chip" href="' + escapeHtml(moduleHref(id)) + '">' + escapeHtml(relatedModule.title || id) + '</a>';
      }
    });
  }

  function renderRoadmapPage(siteData) {
    if (!siteData || !siteData.pages) return;

    const pages = siteData.pages;
    const roadmapPages = pages.roadmap_pages || {};
    const detailPages = pages.detail_pages || {};
    const params = new URLSearchParams(window.location.search);
    const roadmapId = params.get('id') || '';
    const roadmapPage = roadmapId ? roadmapPages[roadmapId] : null;

    const titleEl = document.getElementById('roadmapTitle');
    const summaryEl = document.getElementById('roadmapSummary');
    const metaEl = document.getElementById('roadmapMeta');
    const stageEl = document.getElementById('roadmapStageList');
    const relatedEl = document.getElementById('roadmapRelatedList');
    const sourceEl = document.getElementById('roadmapSourceList');
    const emptyState = document.getElementById('roadmapEmptyState');
    const breadcrumb = document.getElementById('roadmapBreadcrumb');

    if (!roadmapPage) {
      if (emptyState) emptyState.hidden = false;
      if (titleEl) titleEl.textContent = '未找到对应 roadmap page';
      if (summaryEl) {
        summaryEl.innerHTML = '请在 URL 里传入合法的 <code>?id=...</code>，例如 <code>roadmap.html?id=roadmap-route-a-motion-control</code>。';
        removeLoadingState(summaryEl);
      }
      if (metaEl) {
        metaEl.innerHTML = '<p class="data-meta">当前没有匹配到 roadmap_pages 项。</p>';
        removeLoadingState(metaEl);
      }
      if (stageEl) {
        stageEl.innerHTML = '<article class="card"><p>当前无可展示的阶段。</p></article>';
        removeLoadingState(stageEl);
      }
      renderInternalLinks(relatedEl, [], detailPages, { emptyText: '当前无可展示的相关项。' });
      renderSourceCards(sourceEl, [], '当前无可展示的来源链接。');
      if (breadcrumb) removeLoadingState(breadcrumb);
      return;
    }

    if (emptyState) emptyState.hidden = true;
    document.title = (roadmapPage.title || roadmapId) + ' | Robotics Notebooks';

    if (titleEl) titleEl.textContent = roadmapPage.title || roadmapId;
    if (summaryEl) {
      summaryEl.innerHTML = escapeHtml(roadmapPage.summary || '当前路线暂无摘要。');
      removeLoadingState(summaryEl);
    }
    if (metaEl) {
      metaEl.innerHTML = [
        '<p><strong>id：</strong><code>' + escapeHtml(roadmapPage.id || roadmapId) + '</code></p>',
        '<p><strong>阶段数：</strong>' + escapeHtml((roadmapPage.stages || []).length) + '</p>',
        '<p><strong>相关项：</strong>' + escapeHtml((roadmapPage.related_items || []).length) + '</p>',
        '<p><strong>来源链接：</strong>' + escapeHtml((roadmapPage.source_links || []).length) + '</p>'
      ].join('');
      removeLoadingState(metaEl);
    }
    if (breadcrumb) {
      breadcrumb.innerHTML = [
        '<a href="index.html">首页</a>',
        '<span>/</span>',
        '<span>' + escapeHtml(roadmapPage.title || roadmapId) + '</span>'
      ].join('');
      removeLoadingState(breadcrumb);
    }
    if (stageEl) {
      const stages = Array.isArray(roadmapPage.stages) ? roadmapPage.stages : [];
      stageEl.innerHTML = stages.length ? stages.map(function (stage) {
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3>' + escapeHtml(stage.title || stage.id || '未命名阶段') + '</h3>',
          '    <p class="card-meta">阶段 ID：' + escapeHtml(stage.id || '-') + '</p>',
          '  </div>',
          '</article>'
        ].join('');
      }).join('') : '<article class="card"><p>当前路线暂无阶段定义。</p></article>';
      removeLoadingState(stageEl);
    }

    renderInternalLinks(relatedEl, roadmapPage.related_items, detailPages, { emptyText: '当前路线暂无相关项。' });
    renderSourceCards(sourceEl, roadmapPage.source_links, '当前路线暂无来源链接。');
  }

  function renderTechMapNodeCard(node, detailPages) {
    const related = Array.isArray(node.related) ? node.related.slice(0, 3) : [];
    const detailSummary = detailPages[node.id] && detailPages[node.id].summary ? detailPages[node.id].summary : node.summary;
    return [
      '<article class="card data-card" data-layer="' + escapeHtml(node.layer || 'meta') + '">',
      '  <div>',
      '    <h3><a href="' + escapeHtml(detailHref(node.id)) + '">' + escapeHtml(node.title || node.id) + '</a></h3>',
      '    <p class="card-meta">layer: ' + escapeHtml(node.layer || 'meta') + ' · kind: ' + escapeHtml(node.node_kind || '-') + '</p>',
      '    <p>' + escapeHtml(detailSummary || '暂无节点摘要') + '</p>',
      '  </div>',
      '  <div class="chip-list">',
      '    <span class="data-chip"><code>' + escapeHtml(node.id || '-') + '</code></span>',
      '    <a class="btn-secondary btn-inline" href="' + escapeHtml(detailHref(node.id)) + '">打开详情页</a>',
      '  </div>',
      '  <ul>' + (related.length ? related.map(function (item) { return '<li><a href="' + escapeHtml(detailHref(item)) + '"><code>' + escapeHtml(item) + '</code></a></li>'; }).join('') : '<li>当前节点暂无 related</li>') + '</ul>',
      '</article>'
    ].join('');
  }

  function renderTechMapGroupedNodes(nodes, detailPages) {
    const grouped = nodes.reduce(function (acc, node) {
      const layer = node.layer || 'meta';
      if (!acc[layer]) acc[layer] = [];
      acc[layer].push(node);
      return acc;
    }, {});
    return Object.keys(grouped).map(function (layer) {
      const layerNodes = grouped[layer];
      return [
        '<details class="tech-map-group" open>',
        '  <summary class="tech-map-group-summary">' + escapeHtml(layer) + ' · ' + escapeHtml(layerNodes.length) + '</summary>',
        '  <div class="card-grid data-grid tech-map-group-grid">',
             layerNodes.map(function (node) { return renderTechMapNodeCard(node, detailPages); }).join(''),
        '  </div>',
        '</details>'
      ].join('');
    }).join('');
  }

  function renderTechMapNodes(nodes, detailPages, activeLayer) {
    const nodeGrid = document.getElementById('techMapNodeGrid');
    if (!nodeGrid) return;

    const visibleNodes = activeLayer === 'all'
      ? nodes
      : nodes.filter(function (node) { return (node.layer || 'meta') === activeLayer; });

    nodeGrid.innerHTML = visibleNodes.length
      ? renderTechMapGroupedNodes(visibleNodes, detailPages)
      : '<article class="card"><p>当前筛选条件下暂无 tech-map 节点。</p></article>';
    removeLoadingState(nodeGrid);
  }

  function renderTechMapFilters(layerCounts, activeLayer, onSelect) {
    const filterList = document.getElementById('techMapFilterList');
    const filterState = document.getElementById('techMapFilterState');
    if (!filterList || !filterState) return;

    const layers = ['all'].concat(Object.keys(layerCounts));
    filterState.innerHTML = activeLayer === 'all'
      ? '当前展示 <strong>全部 layer</strong>。点击下方 chip 可只看单个 layer；回到 all 时会自动清掉 URL 里的筛选参数。'
      : '当前仅展示 <strong>' + escapeHtml(activeLayer) + '</strong> layer。这个状态会同步到 URL，刷新或分享链接后仍可保留。';
    removeLoadingState(filterState);

    filterList.innerHTML = layers.map(function (layer) {
      const count = layer === 'all'
        ? Object.keys(layerCounts).reduce(function (sum, key) { return sum + layerCounts[key]; }, 0)
        : layerCounts[layer];
      const activeClass = layer === activeLayer ? ' data-chip-active' : '';
      return '<button type="button" class="data-chip data-chip-button' + activeClass + '" data-layer="' + escapeHtml(layer) + '">' + escapeHtml(layer) + ' · ' + escapeHtml(count) + '</button>';
    }).join('');
    removeLoadingState(filterList);

    Array.from(filterList.querySelectorAll('[data-layer]')).forEach(function (button) {
      button.addEventListener('click', function () {
        onSelect(button.getAttribute('data-layer'));
      });
    });
  }

  function renderTechMapPage(siteData) {
    if (!siteData || !siteData.pages) return;

    const techMapPage = siteData.pages.tech_map_page || {};
    const detailPages = siteData.pages.detail_pages || {};
    const nodes = Array.isArray(techMapPage.nodes) ? techMapPage.nodes : [];
    const heroSummary = document.getElementById('techMapHeroSummary');
    const graphMeta = document.getElementById('techMapGraphMeta');
    const layerList = document.getElementById('techMapLayerList');
    const params = new URLSearchParams(window.location.search);

    const layerCounts = nodes.reduce(function (acc, node) {
      const layer = node.layer || 'meta';
      acc[layer] = (acc[layer] || 0) + 1;
      return acc;
    }, {});

    if (heroSummary) {
      const layerCount = Object.keys(layerCounts).length;
      heroSummary.innerHTML = '当前 tech-map 共收录 <strong>' + escapeHtml(nodes.length) + '</strong> 个节点，覆盖 <strong>' + escapeHtml(layerCount) + '</strong> 个 layer。第一阶段先用 layer 分布 + 节点卡片验证页面消费模型，不急着上复杂可视化。';
      removeLoadingState(heroSummary);
    }

    if (graphMeta) {
      graphMeta.innerHTML = [
        '<p><strong>overview：</strong><a href="' + escapeHtml(detailHref((techMapPage.graph_meta || {}).overview_id || '')) + '"><code>' + escapeHtml((techMapPage.graph_meta || {}).overview_id || '-') + '</code></a></p>',
        '<p><strong>dependency_graph：</strong><a href="' + escapeHtml(detailHref((techMapPage.graph_meta || {}).dependency_graph_id || '')) + '"><code>' + escapeHtml((techMapPage.graph_meta || {}).dependency_graph_id || '-') + '</code></a></p>',
        '<p class="data-meta">当前页面直接消费 <code>tech_map_page</code>，节点统一回流到 detail page。</p>'
      ].join('');
      removeLoadingState(graphMeta);
    }

    renderChipList(layerList, Object.keys(layerCounts), {
      renderItem: function (layer) {
        return '<span class="data-chip">' + escapeHtml(layer) + ' · ' + escapeHtml(layerCounts[layer]) + '</span>';
      }
    });

    const allowedLayers = ['all'].concat(Object.keys(layerCounts));
    const requestedLayer = params.get('layer') || 'all';
    const initialLayer = allowedLayers.indexOf(requestedLayer) >= 0 ? requestedLayer : 'all';

    function syncTechMapLayerInUrl(layer) {
      const url = new URL(window.location.href);
      if (layer === 'all') {
        url.searchParams.delete('layer');
      } else {
        url.searchParams.set('layer', layer);
      }
      history.replaceState({}, '', url.toString());
    }

    var currentLayer = initialLayer;
    function updateTechMapLayer(nextLayer) {
      currentLayer = allowedLayers.indexOf(nextLayer) >= 0 ? nextLayer : 'all';
      syncTechMapLayerInUrl(currentLayer);
      renderTechMapFilters(layerCounts, currentLayer, updateTechMapLayer);
      renderTechMapNodes(nodes, detailPages, currentLayer);
    }

    updateTechMapLayer(currentLayer);
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
            return '<li><a href="' + escapeHtml(roadmapHref(item)) + '"><strong>' + escapeHtml(page.title || item) + '</strong></a><br /><small>' + escapeHtml(item) + '</small></li>';
          }).join('')
        : '<li>暂无快速入口数据</li>';
      removeLoadingState(quickEntries);
    }

    renderChipList(document.getElementById('featuredChainPreview'), homePage.featured_chain, {
      renderItem: function (item) {
        const page = detailPages[item] || {};
        return '<a class="data-chip" href="' + escapeHtml(detailHref(item)) + '" title="' + escapeHtml(item) + '">' + escapeHtml(page.title || item) + '</a>';
      }
    });

    renderChipList(document.getElementById('featuredModulesPreview'), homePage.featured_modules, {
      renderItem: function (item) {
        const page = modulePages[item] || {};
        return '<a class="data-chip" href="' + escapeHtml(moduleHref(item)) + '" title="' + escapeHtml(item) + '">' + escapeHtml(page.title || item) + '</a>';
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
               entries.map(function (item) { return '    <li><a href="' + escapeHtml(detailHref(item)) + '"><code>' + escapeHtml(item) + '</code></a></li>'; }).join(''),
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
        const related = Array.isArray(roadmapPage.related_items) ? roadmapPage.related_items.slice(0, 4) : [];
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3><a href="' + escapeHtml(roadmapHref(roadmapId)) + '">' + escapeHtml(roadmapPage.title || roadmapId) + '</a></h3>',
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
          '  <div class="chip-list">',
               related.map(function (item) { return '<a class="data-chip" href="' + escapeHtml(detailHref(item)) + '">' + escapeHtml(item) + '</a>'; }).join(''),
          '  </div>',
          '</article>'
        ].join('');
      });
      roadmapGrid.innerHTML = roadmapCards.length ? roadmapCards.join('') : '<article class="card"><p>暂无路线页数据</p></article>';
      removeLoadingState(roadmapGrid);
    }

    const detailGrid = document.getElementById('detailPreviewGrid');
    if (detailGrid) {
      const preferredDetails = [
        'wiki-concepts-centroidal-dynamics',
        'wiki-methods-model-predictive-control',
        'entity-isaac-gym-isaac-lab',
        'tech-node-control-mpc'
      ];
      const detailCards = preferredDetails
        .map(function (id) { return detailPages[id]; })
        .filter(Boolean)
        .map(function (detailPage) {
          const tags = Array.isArray(detailPage.tags) ? detailPage.tags.slice(0, 5) : [];
          const related = Array.isArray(detailPage.related) ? detailPage.related.slice(0, 4) : [];
          const sources = Array.isArray(detailPage.source_links) ? detailPage.source_links.slice(0, 2) : [];
          return [
            '<article class="card data-card">',
            '  <div>',
            '    <h3><a href="' + escapeHtml(detailHref(detailPage.id)) + '">' + escapeHtml(detailPage.title || detailPage.id) + '</a></h3>',
            '    <p class="card-meta">' + escapeHtml(detailPage.type || 'detail_page') + '</p>',
            '    <p>' + escapeHtml(detailPage.summary || '暂无摘要') + '</p>',
            '    <p class="data-submeta"><code>' + escapeHtml(detailPage.path || detailPage.id || '') + '</code></p>',
            '  </div>',
            '  <div>',
            '    <h4>标签</h4>',
            '    <div class="chip-list">' + (tags.length ? tags.map(function (tag) { return '<span class="data-chip">' + escapeHtml(tag) + '</span>'; }).join('') : '<span class="data-meta">暂无标签</span>') + '</div>',
            '  </div>',
            '  <div>',
            '    <h4>关联项</h4>',
            '    <ul>' + (related.length ? related.map(function (item) { return '<li><a href="' + escapeHtml(detailHref(item)) + '"><code>' + escapeHtml(item) + '</code></a></li>'; }).join('') : '<li>暂无关联项</li>') + '</ul>',
            '  </div>',
            '  <div>',
            '    <h4>来源链接</h4>',
            '    <ul>' + (sources.length ? sources.map(function (url) { return '<li><a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(url) + '</a></li>'; }).join('') : '<li>暂无来源链接</li>') + '</ul>',
            '  </div>',
            '  <div class="chip-list">',
            '    <a class="btn-secondary btn-inline" href="' + escapeHtml(detailHref(detailPage.id)) + '">打开详情页</a>',
            '  </div>',
            '</article>'
          ].join('');
        });
      detailGrid.innerHTML = detailCards.length ? detailCards.join('') : '<article class="card"><p>暂无详情页数据</p></article>';
      removeLoadingState(detailGrid);
    }

    const techMapSummary = document.getElementById('techMapSummary');
    const techMapNodes = Array.isArray(techMapPage.nodes) ? techMapPage.nodes : [];
    if (techMapSummary) {
      const uniqueLayers = Array.from(new Set(techMapNodes.map(function (node) { return node.layer; }).filter(Boolean)));
      techMapSummary.innerHTML = [
        '<h4>graph_meta</h4>',
        '<p class="data-meta">overview_id: <code>' + escapeHtml((techMapPage.graph_meta || {}).overview_id || '-') + '</code></p>',
        '<p class="data-meta">dependency_graph_id: <code>' + escapeHtml((techMapPage.graph_meta || {}).dependency_graph_id || '-') + '</code></p>',
        '<p class="data-submeta">当前 tech-map 共 ' + escapeHtml(techMapNodes.length) + ' 个节点，覆盖 ' + escapeHtml(uniqueLayers.length) + ' 个 layer，可直接生成第一版分层节点视图。</p>'
      ].join('');
      removeLoadingState(techMapSummary);
    }

    const layerCounts = techMapNodes.reduce(function (acc, node) {
      const layer = node.layer || 'unknown';
      acc[layer] = (acc[layer] || 0) + 1;
      return acc;
    }, {});
    renderChipList(document.getElementById('techMapLayers'), Object.keys(layerCounts), {
      renderItem: function (layer) {
        return '<span class="data-chip">' + escapeHtml(layer) + ' · ' + escapeHtml(layerCounts[layer]) + '</span>';
      }
    });

    const techMapNodeGrid = document.getElementById('techMapNodeGrid');
    if (techMapNodeGrid) {
      const nodeCards = techMapNodes.slice(0, 6).map(function (node) {
        const related = Array.isArray(node.related) ? node.related.slice(0, 3) : [];
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3><a href="' + escapeHtml(detailHref(node.id)) + '">' + escapeHtml(node.title || node.id) + '</a></h3>',
          '    <p class="card-meta">layer: ' + escapeHtml(node.layer || '-') + ' · kind: ' + escapeHtml(node.node_kind || '-') + '</p>',
          '    <p>' + escapeHtml(node.summary || '暂无节点摘要') + '</p>',
          '  </div>',
          '  <div class="chip-list">',
          '    <span class="data-chip"><code>' + escapeHtml(node.id || '-') + '</code></span>',
          '    <a class="btn-secondary btn-inline" href="' + escapeHtml(detailHref(node.id)) + '">打开详情页</a>',
          '  </div>',
          '  <ul>' + (related.length ? related.map(function (item) { return '<li><a href="' + escapeHtml(detailHref(item)) + '"><code>' + escapeHtml(item) + '</code></a></li>'; }).join('') : '<li>当前节点暂无 related</li>') + '</ul>',
          '</article>'
        ].join('');
      });
      techMapNodeGrid.innerHTML = nodeCards.length ? nodeCards.join('') : '<article class="card"><p>暂无 tech-map 节点数据</p></article>';
      removeLoadingState(techMapNodeGrid);
    }
  }

  function handlePageDataError(error, ids) {
    ids
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean)
      .forEach(function (element) {
        element.innerHTML = '<p class="data-meta">读取 <code>exports/site-data-v1.json</code> 失败：' + escapeHtml(error.message) + '</p>';
        removeLoadingState(element);
      });
  }

  const previewRoot = document.getElementById('previewSummary');
  const detailRoot = document.getElementById('detailTitle');
  const techMapRoot = document.getElementById('techMapNodeGrid');
  const moduleRoot = document.getElementById('moduleEntryList');
  const roadmapRoot = document.getElementById('roadmapStageList');

  if (previewRoot || detailRoot || techMapRoot || moduleRoot || roadmapRoot) {
    fetch('exports/site-data-v1.json')
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        return response.json();
      })
      .then(function (siteData) {
        if (previewRoot) renderPreviewPage(siteData);
        if (detailRoot) renderDetailPage(siteData);
        if (techMapRoot) renderTechMapPage(siteData);
        if (moduleRoot) renderModulePage(siteData);
        if (roadmapRoot) renderRoadmapPage(siteData);
      })
      .catch(function (error) {
        if (previewRoot) {
          handlePageDataError(error, [
            'previewSummary',
            'homeHeroPreview',
            'quickEntriesPreview',
            'featuredChainPreview',
            'featuredModulesPreview',
            'modulePreviewGrid',
            'roadmapPreviewGrid',
            'detailPreviewGrid',
            'techMapSummary',
            'techMapLayers',
            'techMapNodeGrid'
          ]);
        }
        if (detailRoot) {
          handlePageDataError(error, [
            'detailBreadcrumb',
            'detailSummary',
            'detailMeta',
            'detailTocList',
            'detailContent',
            'detailTagList',
            'detailRelatedList',
            'detailSourceList'
          ]);
        }
        if (techMapRoot) {
          handlePageDataError(error, [
            'techMapHeroSummary',
            'techMapGraphMeta',
            'techMapLayerList',
            'techMapNodeGrid'
          ]);
        }
        if (moduleRoot) {
          handlePageDataError(error, [
            'moduleBreadcrumb',
            'moduleSummary',
            'moduleMeta',
            'moduleEntryList',
            'moduleReferenceList',
            'moduleRoadmapList',
            'moduleRelatedModules'
          ]);
        }
        if (roadmapRoot) {
          handlePageDataError(error, [
            'roadmapBreadcrumb',
            'roadmapSummary',
            'roadmapMeta',
            'roadmapStageList',
            'roadmapRelatedList',
            'roadmapSourceList'
          ]);
        }
      });
  }
})();
