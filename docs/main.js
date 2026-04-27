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

  function stripYamlFrontmatter(markdown) {
    const source = String(markdown || '').replace(/\r\n/g, '\n').trim();
    if (!source.startsWith('---\n')) return source;

    const endMatch = source.match(/\n---\s*(?:\n|$)/);
    if (!endMatch || typeof endMatch.index !== 'number') return source;
    return source.slice(endMatch.index + endMatch[0].length).trim();
  }

  function renderHomeStats(graphStats, coverageText) {
    var heroNodeCount = document.getElementById('heroNodeCount');
    var heroEdgeCount = document.getElementById('heroEdgeCount');
    var heroCoverageCount = document.getElementById('heroCoverageCount');
    var heroNotesCount = document.getElementById('heroNotesCount');
    var wikiSearchSubtitle = document.getElementById('wikiSearchSubtitle');
    if (!heroNodeCount && !heroEdgeCount && !heroCoverageCount && !heroNotesCount && !wikiSearchSubtitle) return;

    var nodeCount = graphStats && typeof graphStats.node_count === 'number' ? graphStats.node_count : null;
    var edgeCount = graphStats && typeof graphStats.edge_count === 'number' ? graphStats.edge_count : null;
    var coverageCount = String(coverageText || '').trim();
    var notesCount = graphStats && graphStats.paper_notes && typeof graphStats.paper_notes.count === 'number'
      ? graphStats.paper_notes.count
      : null;

    if (heroNodeCount && nodeCount !== null) heroNodeCount.textContent = String(nodeCount);
    if (heroEdgeCount && edgeCount !== null) heroEdgeCount.textContent = String(edgeCount);
    if (heroCoverageCount && coverageCount) heroCoverageCount.textContent = coverageCount;
    if (heroNotesCount && notesCount !== null) heroNotesCount.textContent = String(notesCount);
    if (wikiSearchSubtitle && nodeCount !== null) {
      wikiSearchSubtitle.textContent = '在 ' + nodeCount + ' 个知识节点中快速定位概念、方法或任务。↑↓ 键导航，Enter 打开，Esc 清空。';
    }
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

  function buildMarkdownRouteIndex(siteData) {
    const pages = siteData && siteData.pages ? siteData.pages : {};
    const detailPages = pages.detail_pages || {};
    const roadmapPages = pages.roadmap_pages || {};
    const routeIndex = {};

    Object.keys(detailPages).forEach(function (id) {
      const page = detailPages[id] || {};
      if (page.path) routeIndex[page.path] = detailHref(id);
    });
    Object.keys(roadmapPages).forEach(function (id) {
      const page = roadmapPages[id] || {};
      if (page.path) routeIndex[page.path] = roadmapHref(id);
    });

    return routeIndex;
  }

  function normalizeInternalMarkdownTarget(target, currentPath) {
    const raw = String(target || '').trim();
    if (!raw || /^(https?:)?\/\//i.test(raw) || /^[a-z][a-z0-9+.-]*:/i.test(raw)) return '';
    if (raw.startsWith('#')) return (currentPath || '') + raw;

    const parts = raw.split('#');
    const pathPart = parts[0] || '';
    const hash = parts[1] ? '#' + parts[1] : '';
    if (!pathPart) return (currentPath || '') + hash;

    const baseSegments = String(currentPath || '').split('/').filter(Boolean);
    if (baseSegments.length) baseSegments.pop();
    const targetSegments = pathPart.startsWith('/')
      ? pathPart.split('/').filter(Boolean)
      : baseSegments.concat(pathPart.split('/').filter(Boolean));
    const resolvedSegments = [];
    targetSegments.forEach(function (segment) {
      if (!segment || segment === '.') return;
      if (segment === '..') {
        if (resolvedSegments.length) resolvedSegments.pop();
        return;
      }
      resolvedSegments.push(segment);
    });

    return resolvedSegments.join('/') + hash;
  }

  function resolveInternalMarkdownHref(target, currentPath, routeIndex) {
    const normalizedTarget = normalizeInternalMarkdownTarget(target, currentPath);
    if (!normalizedTarget) return '';

    const hashIndex = normalizedTarget.indexOf('#');
    const normalizedPath = hashIndex >= 0 ? normalizedTarget.slice(0, hashIndex) : normalizedTarget;
    const hash = hashIndex >= 0 ? normalizedTarget.slice(hashIndex) : '';
    if (!normalizedPath && hash) return hash;

    return routeIndex && routeIndex[normalizedPath] ? routeIndex[normalizedPath] + hash : '';
  }

  function renderInlineMarkdown(text, markdownContext) {
    markdownContext = markdownContext || {};
    const source = String(text || '');

    // 1. Math protection: Extract all math formulas before they get mangled by Markdown parsing
    const mathTokens = [];
    const mathPrefix = '@@MDMATHTOKEN';
    const withMathTokens = source
      .replace(/\$\$([\s\S]+?)\$\$/g, function (match, expr) {
        const token = mathPrefix + mathTokens.length + '@@';
        mathTokens.push({ token: token, html: '$$' + expr + '$$' });
        return token;
      })
      .replace(/\\\(([\s\S]+?)\\\)/g, function (match, expr) {
        const token = mathPrefix + mathTokens.length + '@@';
        mathTokens.push({ token: token, html: '\\(' + expr + '\\)' });
        return token;
      })
      .replace(/\$([^\$\s](?:[^\$]*[^\$\s])?)\$/g, function (match, expr) {
        const token = mathPrefix + mathTokens.length + '@@';
        // Normalize $...$ to \(...\) so downstream renderMathBlocks can catch it
        mathTokens.push({ token: token, html: '\\(' + expr + '\\)' });
        return token;
      });

    // 2. Link protection (existing logic)
    const linkTokens = [];
    const linkPrefix = '@@MDLINKTOKEN';
    const withLinkTokens = withMathTokens.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (match, label, target) {
      let html = '';
      if (/^https?:\/\//i.test(target)) {
        html = '<a href="' + escapeHtml(target) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(label) + '</a>';
      } else {
        const internalHref = resolveInternalMarkdownHref(target, markdownContext.currentPath, markdownContext.routeIndex);
        if (internalHref) {
          html = '<a href="' + escapeHtml(internalHref) + '">' + escapeHtml(label) + '</a>';
        }
      }
      if (!html) return match;
      const token = linkPrefix + linkTokens.length + '@@';
      linkTokens.push({ token: token, html: html });
      return token;
    });

    // 3. Apply standard escapes and basic Markdown styles
    let rendered = escapeHtml(withLinkTokens)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // 4. Restore Links
    linkTokens.forEach(function (entry) {
      rendered = rendered.replace(entry.token, entry.html);
    });

    // 5. Restore Protected Math (safely escaped)
    mathTokens.forEach(function (entry) {
      // The math content must be escaped because it will be part of innerHTML
      // but it should NOT be processed by other markdown rules (already protected).
      rendered = rendered.replace(entry.token, escapeHtml(entry.html));
    });

    return rendered;
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

  function normalizeCodeLang(lang) {
    const value = String(lang || '').trim().toLowerCase();
    if (!value) return 'text';
    if (['py', 'python3'].includes(value)) return 'python';
    if (['sh', 'shell', 'zsh'].includes(value)) return 'bash';
    if (['yml'].includes(value)) return 'yaml';
    if (['js'].includes(value)) return 'javascript';
    if (['txt', 'plain', 'plaintext'].includes(value)) return 'text';
    return value.replace(/[^\w-]/g, '') || 'text';
  }

  function highlightGenericLine(line) {
    return escapeHtml(line);
  }

  function highlightPythonLine(line) {
    const keywords = new Set([
      'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
      'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from',
      'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not',
      'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
    ]);
    const builtins = new Set(['False', 'None', 'True', 'self', 'super', 'len', 'range', 'dict', 'list', 'set', 'tuple', 'str', 'int', 'float', 'print']);
    const tokenRe = /(#.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b[A-Za-z_]\w*\b|\b\d+(?:\.\d+)?\b|[=+\-*\/<>!%]+|[()[\]{}.,:])/g;
    let out = '';
    let lastIndex = 0;
    let afterKeyword = '';
    line.replace(tokenRe, function (token, _whole, offset) {
      out += escapeHtml(line.slice(lastIndex, offset));
      if (token.startsWith('#')) {
        out += '<span class="tok-comment">' + escapeHtml(token) + '</span>';
      } else if (/^['"]/.test(token)) {
        out += '<span class="tok-string">' + escapeHtml(token) + '</span>';
      } else if (/^\d/.test(token)) {
        out += '<span class="tok-number">' + escapeHtml(token) + '</span>';
      } else if (/^[=+\-*\/<>!%]+$/.test(token)) {
        out += '<span class="tok-operator">' + escapeHtml(token) + '</span>';
      } else if (/^[()[\]{}.,:]$/.test(token)) {
        out += '<span class="tok-punctuation">' + escapeHtml(token) + '</span>';
      } else if (afterKeyword === 'class') {
        out += '<span class="tok-class">' + escapeHtml(token) + '</span>';
        afterKeyword = '';
      } else if (afterKeyword === 'def') {
        out += '<span class="tok-function">' + escapeHtml(token) + '</span>';
        afterKeyword = '';
      } else if (keywords.has(token)) {
        out += '<span class="tok-keyword">' + escapeHtml(token) + '</span>';
        afterKeyword = token === 'class' || token === 'def' ? token : '';
      } else if (builtins.has(token)) {
        out += '<span class="tok-builtin">' + escapeHtml(token) + '</span>';
      } else {
        out += '<span class="tok-name">' + escapeHtml(token) + '</span>';
      }
      lastIndex = offset + token.length;
      return token;
    });
    out += escapeHtml(line.slice(lastIndex));
    return out;
  }

  function highlightBashLine(line) {
    const tokenRe = /(#.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:cd|cp|echo|export|git|make|mkdir|mv|pip|python|python3|rm|uv|source|test|then|fi|do|done|for|if|in)\b|\b\d+(?:\.\d+)?\b|[=|&;<>]+)/g;
    let out = '';
    let lastIndex = 0;
    line.replace(tokenRe, function (token, _whole, offset) {
      out += escapeHtml(line.slice(lastIndex, offset));
      if (token.startsWith('#')) out += '<span class="tok-comment">' + escapeHtml(token) + '</span>';
      else if (/^['"]/.test(token)) out += '<span class="tok-string">' + escapeHtml(token) + '</span>';
      else if (/^\d/.test(token)) out += '<span class="tok-number">' + escapeHtml(token) + '</span>';
      else if (/^[=|&;<>]+$/.test(token)) out += '<span class="tok-operator">' + escapeHtml(token) + '</span>';
      else out += '<span class="tok-keyword">' + escapeHtml(token) + '</span>';
      lastIndex = offset + token.length;
      return token;
    });
    out += escapeHtml(line.slice(lastIndex));
    return out;
  }

  function highlightYamlLine(line) {
    const commentIndex = line.indexOf('#');
    const codePart = commentIndex >= 0 ? line.slice(0, commentIndex) : line;
    const commentPart = commentIndex >= 0 ? line.slice(commentIndex) : '';
    const renderedCode = escapeHtml(codePart).replace(/^(\s*)([A-Za-z0-9_.-]+)(\s*:)/, function (_, lead, key, sep) {
      return lead + '<span class="tok-attr">' + key + '</span>' + sep;
    }).replace(/(:\s*)([-+]?\d+(?:\.\d+)?|true|false|null)\b/gi, function (_, sep, value) {
      return sep + '<span class="tok-number">' + value + '</span>';
    });
    return renderedCode + (commentPart ? '<span class="tok-comment">' + escapeHtml(commentPart) + '</span>' : '');
  }

  function highlightCodeLine(line, lang) {
    if (lang === 'python') return highlightPythonLine(line);
    if (lang === 'bash') return highlightBashLine(line);
    if (lang === 'yaml') return highlightYamlLine(line);
    return highlightGenericLine(line);
  }

  function renderCodeBlock(code, lang) {
    const normalizedLang = normalizeCodeLang(lang);
    const rawCode = String(code || '').endsWith('\n') ? String(code || '').slice(0, -1) : String(code || '');
    const lines = rawCode.split('\n');
    const rows = lines.map(function (line, index) {
      return '<div class="code-row">'
        + '<span class="code-ln">' + (index + 1) + '</span>'
        + '<span class="code-cell">' + highlightCodeLine(line, normalizedLang) + '</span>'
        + '</div>';
    }).join('');
    return '<div class="detail-code-block highlight language-' + escapeHtml(normalizedLang) + '">'
      + rows
      + '</div>';
  }

  function renderDetailMath(container) {
    if (!container || typeof window.renderMathInElement !== 'function') return;
    window.renderMathInElement(container, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false
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

  function enhanceDetailHeadings(container) {
    if (!container) return;
    Array.from(container.querySelectorAll('h2[id], h3[id], h4[id]')).forEach(function (heading) {
      if (heading.querySelector('.heading-anchor-link')) return;
      heading.classList.add('detail-heading');
      const anchorLink = document.createElement('button');
      anchorLink.type = 'button';
      anchorLink.className = 'heading-anchor-link';
      anchorLink.setAttribute('class', 'heading-anchor-link');
      anchorLink.setAttribute('aria-label', '复制当前标题链接');
      anchorLink.setAttribute('title', '复制当前标题链接');
      anchorLink.innerHTML = '#';
      anchorLink.addEventListener('click', function () {
        const headingUrl = window.location.origin + window.location.pathname + window.location.search + '#' + heading.id;
        if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
          navigator.clipboard.writeText(headingUrl).catch(function () {});
        }
        history.replaceState({}, '', '#' + heading.id);
        anchorLink.classList.add('copied');
        anchorLink.textContent = '已复制';
        window.setTimeout(function () {
          anchorLink.classList.remove('copied');
          anchorLink.textContent = '#';
        }, 1200);
      });
      heading.appendChild(anchorLink);
    });
  }

  function bindDetailTocSpy(container, tocContainer) {
    if (!container || !tocContainer) return;
    const headings = Array.from(container.querySelectorAll('h2[id], h3[id], h4[id]'));
    const links = Array.from(tocContainer.querySelectorAll('a[href^="#"]'));
    if (!headings.length || !links.length) return;

    function updateActiveTocLink() {
      let activeId = headings[0].id;
      headings.forEach(function (heading) {
        if (heading.getBoundingClientRect().top <= 140) activeId = heading.id;
      });
      links.forEach(function (link) {
        link.classList.toggle('active', link.getAttribute('href') === '#' + activeId);
      });
    }

    window.addEventListener('scroll', updateActiveTocLink, { passive: true });
    window.addEventListener('hashchange', updateActiveTocLink);
    updateActiveTocLink();
  }

  function scrollToDetailHashTarget(container) {
    if (!container) return;
    const rawHash = window.location.hash.replace(/^#/, '');
    if (!rawHash) return;

    let decodedHash = rawHash;
    try {
      decodedHash = decodeURIComponent(rawHash);
    } catch (_) {}

    const safeHash = typeof window.CSS !== 'undefined' && typeof window.CSS.escape === 'function'
      ? window.CSS.escape(decodedHash)
      : decodedHash.replace(/[^\w-]/g, '\\$&');
    const target = container.querySelector('#' + safeHash);
    if (!target) return;

    Array.from(container.querySelectorAll('.detail-hash-target')).forEach(function (node) {
      node.classList.remove('detail-hash-target');
    });
    target.classList.add('detail-hash-target');
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    window.setTimeout(function () {
      target.classList.remove('detail-hash-target');
    }, 1800);
  }

  function renderMarkdownContent(markdown, headings, markdownContext) {
    let source = stripYamlFrontmatter(markdown);
    if (!source) {
      return '<p>当前 detail page 暂无可同步正文。</p>';
    }

    const context = markdownContext || {};
    const lines = source.split('\n');
    const blocks = [];
    const headingQueue = Array.isArray(headings) ? headings.slice() : collectMarkdownHeadings(source);
    let paragraphLines = [];
    let listItems = [];
    let listTag = '';
    let quoteLines = [];
    let codeLines = [];
    let codeLang = '';
    let tableLines = [];
    let inCodeBlock = false;

    function flushParagraph() {
      if (!paragraphLines.length) return;
      blocks.push('<p>' + renderMathBlocks(renderInlineMarkdown(paragraphLines.join(' '), context)) + '</p>');
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
        return '<li>' + renderMathBlocks(renderInlineMarkdown(item, context)) + '</li>';
      }).join('') + '</' + openTag + '>');
      listItems = [];
      listTag = '';
    }

    function flushQuote() {
      if (!quoteLines.length) return;
      blocks.push((function () {
        return '<blockquote>';
      })() + quoteLines.map(function (line) {
        return '<p>' + renderMathBlocks(renderInlineMarkdown(line, context)) + '</p>';
      }).join('') + '</blockquote>');
      quoteLines = [];
    }

    function flushCodeBlock() {
      if (!codeLines.length) return;
      blocks.push(renderCodeBlock(codeLines.join('\n'), codeLang));
      codeLines = [];
      codeLang = '';
    }

    function flushTable() {
      if (!tableLines.length) return;
      const htmlRows = tableLines.map(function (row, i) {
        const isHeader = i === 0;
        const isSeparator = row.replace(/\|/g, '').replace(/-/g, '').replace(/:/g, '').trim().length === 0;
        if (isSeparator) return '';
        let cells = row.split('|').map(function (c) { return c.trim(); });
        if (cells.length > 0 && cells[0] === '') cells.shift();
        if (cells.length > 0 && cells[cells.length - 1] === '') cells.pop();
        const tag = isHeader ? 'th' : 'td';
        return '<tr>' + cells.map(function (c) { return '<' + tag + '>' + renderMathBlocks(renderInlineMarkdown(c, context)) + '</' + tag + '>'; }).join('') + '</tr>';
      }).join('');
      blocks.push('<div class="table-wrapper"><table>' + htmlRows + '</table></div>');
      tableLines = [];
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
          flushTable();
          inCodeBlock = true;
          codeLang = normalizeCodeLang(trimmed.replace(/^```+/, '').trim().split(/\s+/)[0] || '');
        }
        return;
      }

      if (inCodeBlock) {
        codeLines.push(line);
        return;
      }

      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        flushParagraph();
        flushList();
        flushQuote();
        tableLines.push(trimmed);
        return;
      }

      if (tableLines.length) flushTable();

      if (!trimmed) {
        flushParagraph();
        flushList();
        flushQuote();
        return;
      }

      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        flushParagraph();
        flushList();
        flushQuote();
        flushTable();
        blocks.push('<hr>');
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
        blocks.push('<h' + level + ' id="' + escapeHtml(headingId) + '">' + renderMathBlocks(renderInlineMarkdown(text, context)) + '</h' + level + '>');
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
    flushTable();

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
      const href = page.type === 'roadmap_page' ? roadmapHref(id) : detailHref(id);
      const buttonText = page.type === 'roadmap_page' ? '打开路线页' : '打开详情页';
      return [
        '<article class="card data-card">',
        '  <div>',
        '    <h3><a href="' + escapeHtml(href) + '">' + escapeHtml(page.title || id) + '</a></h3>',
        '    <p class="card-meta">' + escapeHtml(page.type || 'detail_page') + '</p>',
        '    <p>' + escapeHtml(page.summary || '当前关联项暂无摘要') + '</p>',
        '  </div>',
        '  <div class="chip-list">',
        '    <span class="data-chip"><code>' + escapeHtml(id) + '</code></span>',
        '    <a class="btn-secondary btn-inline" href="' + escapeHtml(href) + '">' + buttonText + '</a>',
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

  function findRelatedByTags(currentId, currentTags, detailPages, maxResults) {
    if (!Array.isArray(currentTags) || !currentTags.length) return [];
    maxResults = typeof maxResults === 'number' ? maxResults : 5;
    var tagSet = {};
    currentTags.forEach(function (tag) { tagSet[tag] = true; });

    var scored = [];
    Object.keys(detailPages).forEach(function (id) {
      if (id === currentId) return;
      var page = detailPages[id] || {};
      var pageTags = Array.isArray(page.tags) ? page.tags : [];
      var matchCount = 0;
      pageTags.forEach(function (tag) {
        if (tagSet[tag]) matchCount++;
      });
      if (matchCount > 0) {
        scored.push({ id: id, page: page, score: matchCount, topTag: pageTags[0] || '' });
      }
    });

    scored.sort(function (a, b) {
      if (b.score !== a.score) return b.score - a.score;
      return (b.page.title || '').localeCompare(a.page.title || '');
    });

    return scored.slice(0, maxResults).map(function (item) { return item.id; });
  }

  function renderDetailPage(siteData) {
    if (!siteData || !siteData.pages) return;

    const pages = siteData.pages;
    const detailPages = pages.detail_pages || {};
    const markdownRouteIndex = buildMarkdownRouteIndex(siteData);
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
    const recommendedEl = document.getElementById('detailRecommendedList');
    const sourceEl = document.getElementById('detailSourceList');
    const emptyState = document.getElementById('detailEmptyState');
    const emptySection = document.getElementById('detail-empty-section');
    const breadcrumb = document.getElementById('detailBreadcrumb');

    if (!detailPage) {
      if (emptySection) emptySection.hidden = false;
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
      if (recommendedEl) {
        recommendedEl.innerHTML = '<article class="card"><p>当前无可展示的相关推荐。</p></article>';
        removeLoadingState(recommendedEl);
      }
      renderSourceCards(sourceEl, [], '当前无可展示的来源链接。');
      if (breadcrumb) removeLoadingState(breadcrumb);
      return;
    }

    if (emptySection) emptySection.hidden = true;
    if (emptyState) emptyState.hidden = true;
    document.title = (detailPage.title || detailId) + ' | Robotics Notebooks';

    const graphLink = document.getElementById('detailGraphLink');
    if (graphLink) {
      graphLink.href = 'graph.html?focus=' + encodeURIComponent(detailPage.id || detailId);
    }
    var ogTitle = document.getElementById('ogTitleMeta');
    var ogDesc = document.getElementById('ogDescMeta');
    var pageDesc = detailPage.summary || '当前页面暂无摘要，可先通过 tags / related / source links 继续导航。';
    if (ogTitle) ogTitle.setAttribute('content', (detailPage.title || detailId) + ' | Robotics Notebooks');
    if (ogDesc) ogDesc.setAttribute('content', pageDesc);
    var metaDesc = document.getElementById('metaDescription');
    if (metaDesc && detailPage.summary) {
      metaDesc.setAttribute('content', detailPage.summary.slice(0, 160));
    }

    function isMetadataOnlySummary(summary) {
      return /^type:\s*[\w-]+[。.]?$/i.test(String(summary || '').trim());
    }

    if (titleEl) titleEl.textContent = detailPage.title || detailId;
    if (summaryEl) {
      const summaryText = detailPage.summary || '';
      if (summaryText && !isMetadataOnlySummary(summaryText)) {
        summaryEl.hidden = false;
        summaryEl.innerHTML = escapeHtml(summaryText);
      } else {
        summaryEl.hidden = true;
        summaryEl.textContent = '';
      }
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
      contentEl.innerHTML = contentMarkdown ? renderMarkdownContent(contentMarkdown, detailHeadings, {
        currentPath: detailPage.path || '',
        routeIndex: markdownRouteIndex
      }) : '<p>当前 detail page 暂无可同步正文。</p>';
      renderDetailMath(contentEl);
      enhanceDetailHeadings(contentEl);
      bindDetailTocSpy(contentEl, tocEl);
      window.addEventListener('hashchange', function () { scrollToDetailHashTarget(contentEl); });
      scrollToDetailHashTarget(contentEl);
      removeLoadingState(contentEl);
    }

    renderChipList(tagEl, detailPage.tags, {
      renderItem: function (tag) {
        return '<span class="data-chip">' + escapeHtml(tag) + '</span>';
      }
    });
    renderInternalLinks(relatedEl, detailPage.related, detailPages, { emptyText: '当前 detail page 暂无 related。' });

    // V17: 记录并渲染阅读足迹
    updateRecentVisits(detailPage);

    if (recommendedEl) {
      var recommendedIds = findRelatedByTags(detailId, detailPage.tags, detailPages, 5);
      if (recommendedIds.length) {
        recommendedEl.innerHTML = recommendedIds.map(function (id) {
          var page = detailPages[id] || {};
          var pageTags = Array.isArray(page.tags) ? page.tags : [];
          return [
            '<article class="card data-card">',
            '  <div>',
            '    <h3><a href="' + escapeHtml(detailHref(id)) + '">' + escapeHtml(page.title || id) + '</a></h3>',
            '    <p class="card-meta">tag 匹配推荐</p>',
            '    <p>' + escapeHtml(page.summary || '暂无摘要') + '</p>',
            '  </div>',
            '  <div class="chip-list">',
            pageTags.slice(0, 3).map(function (tag) {
              return '<span class="data-chip">' + escapeHtml(tag) + '</span>';
            }).join(''),
            '    <a class="btn-secondary btn-inline" href="' + escapeHtml(detailHref(id)) + '">打开详情页</a>',
            '  </div>',
            '</article>'
          ].join('');
        }).join('');
      } else {
        recommendedEl.innerHTML = '<article class="card"><p>暂无 tag 匹配的相关推荐。</p></article>';
      }
      removeLoadingState(recommendedEl);
    }

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
    const legacyRoadmapIds = {
      'roadmap-route-a-motion-control': 'roadmap-motion-control'
    };
    const requestedRoadmapId = params.get('id') || '';
    const roadmapId = legacyRoadmapIds[requestedRoadmapId] || requestedRoadmapId;
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
        summaryEl.innerHTML = '请在 URL 里传入合法的 <code>?id=...</code>，例如 <code>roadmap.html?id=roadmap-motion-control</code>。';
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
        const related = Array.isArray(stage.related_items) ? stage.related_items.slice(0, 8) : [];
        const linksHtml = related.length ? [
          '    <div class="chip-list stage-link-list">',
               related.map(function (id) {
                 const page = detailPages[id] || {};
                 const href = page.type === 'roadmap_page' ? roadmapHref(id) : detailHref(id);
                 return '<a class="data-chip" href="' + escapeHtml(href) + '" title="' + escapeHtml(id) + '">' + escapeHtml(page.title || id) + '</a>';
               }).join(''),
          '    </div>'
        ].join('') : '    <p class="data-meta">当前阶段暂无内部链接。</p>';
        return [
          '<article class="card data-card">',
          '  <div>',
          '    <h3>' + escapeHtml(stage.title || stage.id || '未命名阶段') + '</h3>',
          '    <p class="card-meta">阶段 ID：' + escapeHtml(stage.id || '-') + '</p>',
          '    <p class="card-meta">关联入口：' + escapeHtml(related.length) + '</p>',
          '  </div>',
          '  <div>',
          linksHtml,
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
    const detail = detailPages[node.id] || {};
    const detailSummary = detail.summary || node.summary;
    const hasIngest = detail.has_ingest;
    const ingestBadge = hasIngest
      ? '<span class="ingest-badge" title="已有 sources/ ingest 来源：' + escapeHtml(detail.ingest_source || '') + '">📄 ingest</span>'
      : '<span class="ingest-badge ingest-missing" title="暂无 sources/papers/ 对应条目">— no ingest</span>';
    return [
      '<article class="card data-card" data-layer="' + escapeHtml(node.layer || 'meta') + '">',
      '  <div>',
      '    <h3><a href="' + escapeHtml(detailHref(node.id)) + '">' + escapeHtml(node.title || node.id) + '</a></h3>',
      '    <p class="card-meta">layer: ' + escapeHtml(node.layer || 'meta') + ' · kind: ' + escapeHtml(node.node_kind || '-') + ' · ' + ingestBadge + '</p>',
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
    const chipList = document.getElementById('filter-chip-list');
    const stateText = document.getElementById('filter-state-text');
    const toggleText = document.getElementById('filter-toggle-text');
    const badge = document.getElementById('filter-badge');
    if (!chipList) return;

    const layers = ['all'].concat(Object.keys(layerCounts));

    // 更新浮窗内的状态文字
    if (stateText) {
      stateText.textContent = activeLayer === 'all'
        ? '当前展示全部 layer'
        : '当前展示 ' + activeLayer + ' layer';
    }

    // 更新按钮文字 + badge
    if (toggleText) {
      toggleText.textContent = activeLayer === 'all' ? '筛选' : activeLayer;
    }
    if (badge) {
      if (activeLayer === 'all') {
        badge.style.display = 'none';
        badge.textContent = '';
      } else {
        badge.style.display = 'inline';
        badge.textContent = '●';
      }
    }

    // 渲染 layer chips 到浮窗
    chipList.innerHTML = layers.map(function (layer) {
      const count = layer === 'all'
        ? Object.keys(layerCounts).reduce(function (sum, key) { return sum + layerCounts[key]; }, 0)
        : layerCounts[layer];
      const activeClass = layer === activeLayer ? ' data-chip-active' : '';
      return '<button type="button" class="data-chip data-chip-button' + activeClass + '" data-layer="' + escapeHtml(layer) + '">' + escapeHtml(layer) + ' · ' + escapeHtml(count) + '</button>';
    }).join('');

    Array.from(chipList.querySelectorAll('[data-layer]')).forEach(function (button) {
      button.addEventListener('click', function () {
        onSelect(button.getAttribute('data-layer'));
        // 选完后关闭浮窗
        var panel = document.getElementById('filter-panel');
        if (panel) panel.hidden = true;
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

    /* ── 筛选浮窗交互（参照 physics-panel 模式）── */
    var filterToggle = document.getElementById('filter-toggle');
    var filterPanel = document.getElementById('filter-panel');
    var filterClose = document.getElementById('filter-close');

    if (filterToggle && filterPanel) {
      filterToggle.addEventListener('click', function () {
        filterPanel.hidden = !filterPanel.hidden;
      });
    }
    if (filterClose) {
      filterClose.addEventListener('click', function () {
        filterPanel.hidden = true;
      });
    }
    document.addEventListener('click', function (ev) {
      if (!filterPanel || filterPanel.hidden) return;
      var onToggle = ev.target.closest && ev.target.closest('#filter-toggle');
      var onPanel = ev.target.closest && ev.target.closest('#filter-panel');
      if (!onToggle && !onPanel) {
        filterPanel.hidden = true;
      }
    });
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && filterPanel && !filterPanel.hidden) {
        filterPanel.hidden = true;
      }
    });
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
  const homeStatsRoot = document.getElementById('heroNodeCount') || document.getElementById('wikiSearchSubtitle');

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

  if (homeStatsRoot) {
    fetch('exports/home-stats.json')
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (stats) {
        renderHomeStats(stats, stats.coverage ? (stats.coverage.covered + '/' + stats.coverage.total) : '');
      })
      .catch(function (error) {
        console.warn('Home stats sync failed:', error);
      });
  }

  // ── Wiki 全文搜索（index.html 搜索框） ────────────────────────────────────
  var searchInput = document.getElementById('wikiSearchInput');
  var searchResults = document.getElementById('wikiSearchResults');
  var typeFilter = document.getElementById('wikiTypeFilter');
  var tagCloudEl = document.getElementById('wikiTagCloud');
  if (searchInput && searchResults) {
    var _indexData = null;
    var _selectedIndex = -1;  // 键盘导航当前选中项

    var _searchIndex = null;
    var _searchIndexPromise = null;
    var _searchIndexFailed = false;

    fetch('exports/index-v1.json')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        _indexData = data.items || [];
        renderTagCloud(_indexData);
      })
      .catch(function() {});

    function ensureSearchIndex() {
      if (_searchIndex) return Promise.resolve(_searchIndex);
      if (_searchIndexFailed) return Promise.reject(new Error('search-index.json unavailable'));
      if (_searchIndexPromise) return _searchIndexPromise;
      _searchIndexPromise = fetch('search-index.json')
        .then(function(r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(function(data) {
          _searchIndex = data;
          return data;
        })
        .catch(function(error) {
          _searchIndexFailed = true;
          throw error;
        });
      return _searchIndexPromise;
    }

    function tokenizeQuery(text) {
      var matches = String(text || '').toLowerCase().match(/[a-z0-9_+\-.]+|[\u4e00-\u9fff]+/g) || [];
      var out = [];
      matches.forEach(function(token) {
        out.push(token);
        if (/^[\u4e00-\u9fff]+$/.test(token) && token.length > 1) {
          for (var i = 0; i < token.length - 1; i += 1) out.push(token.slice(i, i + 2));
          token.split('').forEach(function(char) { out.push(char); });
        }
      });
      return out;
    }

    // ── 标签云 ──────────────────────────────────────────────────────────────
    function renderTagCloud(items) {
      if (!tagCloudEl) return;
      var freq = {};
      items.forEach(function(item) {
        (item.tags || []).forEach(function(tag) {
          if (tag) freq[tag] = (freq[tag] || 0) + 1;
        });
      });
      var sorted = Object.entries(freq).sort(function(a,b){ return b[1]-a[1]; }).slice(0,20);
      tagCloudEl.innerHTML = sorted.map(function(entry) {
        var tag = entry[0], count = entry[1];
        return '<button class="tag-chip" data-wiki-tag="' + escapeHtml(tag) + '" title="' + count + ' 页面">'
          + escapeHtml(tag) + '</button>';
      }).join('');
    }

    function getResultCards() {
      return Array.from(searchResults.querySelectorAll('article.card[data-result-url]'));
    }

    function setSelectedIndex(idx) {
      var cards = getResultCards();
      if (!cards.length) return;
      if (idx < -1) idx = cards.length - 1;
      if (idx >= cards.length) idx = -1;
      _selectedIndex = idx;
      cards.forEach(function(card, i) {
        card.classList.toggle('search-result-selected', i === idx);
        if (i === idx) card.scrollIntoView({ block: 'nearest' });
      });
    }

    var HOT_QUERIES = ['强化学习', 'WBC 全身控制', 'Sim2Real', '模仿学习', '运动控制', 'MPC'];

    function renderEmptyState() {
      var hotHtml = HOT_QUERIES.map(function(q) {
        return '<button class="tag-chip" onclick="wikiSearchInput.value=\'' + escapeHtml(q)
          + '\';triggerSearch()" style="cursor:pointer">' + escapeHtml(q) + '</button>';
      }).join('');
      searchResults.innerHTML = '<div style="grid-column:1/-1;color:var(--text-muted);font-size:.85rem">'
        + '<p style="margin-bottom:.5rem">热门查询：</p>'
        + '<div class="chip-list">' + hotHtml + '</div>'
        + '</div>';
    }

    function renderNoResults(q) {
      searchResults.innerHTML = '<div style="grid-column:1/-1;color:var(--text-muted)">'
        + '<p>未找到 <strong>' + escapeHtml(q) + '</strong> 的匹配结果。</p>'
        + '<ul style="margin:.5rem 0;padding-left:1.2rem;font-size:.85rem">'
        + '<li>尝试更短的关键词，或英文原文</li>'
        + '<li>命令行搜索：<code>python3 scripts/search_wiki.py "' + escapeHtml(q) + '"</code></li>'
        + '<li>在 <a href="graph.html">知识图谱</a> 中浏览相关节点</li>'
        + '</ul></div>';
    }

    function matchExplanation(item, queryTokens) {
      if (!queryTokens || !queryTokens.length) return '';
      var title = (item.title || '').toLowerCase();
      var summary = (item.summary || '').toLowerCase();
      var tags = (item.tags || []).map(function(t) { return t.toLowerCase(); });

      // 检查标签命中 (V20 增强)
      for (var k = 0; k < tags.length; k++) {
        for (var l = 0; l < queryTokens.length; l++) {
          if (tags[k].indexOf(queryTokens[l]) >= 0) {
            return '核心标签命中: ' + tags[k];
          }
        }
      }

      for (var i = 0; i < queryTokens.length; i++) {
        var t = queryTokens[i];
        if (title.indexOf(t) >= 0) return '标题命中: ' + t;
      }
      for (var j = 0; j < queryTokens.length; j++) {
        if (summary.indexOf(queryTokens[j]) >= 0) return '摘要命中';
      }
      return '正文匹配';
    }


    function renderCards(matched, queryTokens) {
      if (!matched.length) return;
      searchResults.innerHTML = matched.map(function(item) {
        var detailUrl = 'detail.html?id=' + encodeURIComponent(item.id);
        var graphUrl = 'graph.html?focus=' + encodeURIComponent(item.id);
        var typeLabel = item.page_type || (item.path ? item.path.split('/').slice(1, 3).join(' / ') : '');
        var tagLine = (item.tags || []).slice(0, 4).map(function(tag) {
          return '<span class="data-chip">' + escapeHtml(tag) + '</span>';
        }).join('');
        var explain = queryTokens && queryTokens.length
          ? '<span style="font-size:.72rem;color:var(--text-muted);margin-left:6px">'
            + matchExplanation(item, queryTokens) + '</span>'
          : '';
        var graphBtn = '<a href="' + graphUrl + '" onclick="event.stopPropagation()" '
          + 'style="font-size:.75rem;opacity:.6;margin-left:8px;text-decoration:none" '
          + 'title="查看图谱邻居" tabindex="-1">🔗图谱</a>';
        return '<article class="card" data-result-url="' + detailUrl + '">'
          + '<p class="card-meta" style="font-size:.75rem;margin-bottom:.25rem">' + escapeHtml(typeLabel) + explain + '</p>'
          + '<h3><a href="' + detailUrl + '">' + escapeHtml(item.title || item.id) + '</a>' + graphBtn + '</h3>'
          + '<p>' + escapeHtml((item.summary || '').slice(0, 120)) + '</p>'
          + (tagLine ? '<div class="chip-list">' + tagLine + '</div>' : '')
          + '</article>';
      }).join('');
    }

    function bm25Score(doc, queryTokens, indexData) {
      var meta = (indexData && indexData.meta) || {};
      var avgdl = meta.avgdl || 1;
      var k1 = meta.k1 || 1.5;
      var b = meta.b || 0.75;
      var score = 0;
      queryTokens.forEach(function(token) {
        var tf = ((doc.tokens || {})[token]) || 0;
        if (!tf) return;
        var idf = ((indexData.idf || {})[token]) || 0;
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * ((doc.dl || 1) / avgdl)));
      });
      return score;
    }

    function renderSearchResults(query) {
      _selectedIndex = -1;
      var q = query.trim();
      var typeVal = typeFilter ? typeFilter.value : '';
      if (!q && !typeVal) { renderEmptyState(); return; }
      searchResults.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1">加载离线搜索索引中…</p>';
      ensureSearchIndex()
        .then(function(indexData) {
          var docs = (indexData && indexData.docs) || [];
          var queryTokens = tokenizeQuery(q);
          var matched = docs.filter(function(doc) {
            if (typeVal && doc.page_type !== typeVal) return false;
            if (!queryTokens.length) return true;
            return queryTokens.some(function(token) { return ((doc.tokens || {})[token]) > 0; });
          }).map(function(doc) {
            return {
              id: doc.id,
              path: doc.path,
              title: doc.title,
              summary: doc.summary,
              page_type: doc.page_type,
              tags: doc.tags || [],
              _score: queryTokens.length ? bm25Score(doc, queryTokens, indexData) : 0
            };
          }).sort(function(a, b) {
            if (queryTokens.length && b._score !== a._score) return b._score - a._score;
            return String(a.title || '').localeCompare(String(b.title || ''));
          }).slice(0, 10);
          if (!matched.length) {
            renderNoResults(q);
          } else {
            renderCards(matched, queryTokens);
          }
        })
        .catch(function() {
          searchResults.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1">离线搜索索引加载失败，请使用命令行搜索：<code>python3 scripts/search_wiki.py "关键词"</code></p>';
        });
    }

    function escapeHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function triggerSearch() {
      renderSearchResults(searchInput.value);
    }

    // ── 键盘导航（↑↓ 选中 / Enter 打开 / Esc 清空）────────────────────────
    searchInput.addEventListener('keydown', function(ev) {
      var cards = getResultCards();
      if (ev.key === 'ArrowDown') {
        ev.preventDefault();
        setSelectedIndex(_selectedIndex + 1);
      } else if (ev.key === 'ArrowUp') {
        ev.preventDefault();
        setSelectedIndex(_selectedIndex - 1);
      } else if (ev.key === 'Enter') {
        ev.preventDefault();
        var target;
        if (_selectedIndex >= 0 && cards[_selectedIndex]) {
          target = cards[_selectedIndex].getAttribute('data-result-url');
        } else if (cards.length > 0) {
          target = cards[0].getAttribute('data-result-url');
        }
        if (target) window.location.href = target;
      } else if (ev.key === 'Escape') {
        searchInput.value = '';
        searchResults.innerHTML = '';
        _selectedIndex = -1;
        if (typeFilter) typeFilter.value = '';
      }
    });

    searchInput.addEventListener('focus', function() {
      if (_searchIndex || _searchIndexFailed || _searchIndexPromise) return;
      searchResults.innerHTML = '<p style="color:var(--text-muted);grid-column:1/-1">加载中…</p>';
      ensureSearchIndex().then(function() {
        if (searchInput.value.trim()) {
          triggerSearch();
        } else {
          searchResults.innerHTML = '';
        }
      }).catch(function() {
        searchResults.innerHTML = '';
      });
    });

    var _searchTimer;
    searchInput.addEventListener('input', function() {
      clearTimeout(_searchTimer);
      _searchTimer = setTimeout(triggerSearch, 120);
    });
    if (typeFilter) {
      typeFilter.addEventListener('change', triggerSearch);
    }

    document.addEventListener('click', function(e) {
      var tag = e.target.closest('[data-wiki-tag]');
      if (!tag) return;
      var term = tag.getAttribute('data-wiki-tag');
      if (term && searchInput) {
        searchInput.value = term;
        if (typeFilter) typeFilter.value = '';
        triggerSearch();
        searchInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  }

  function updateRecentVisits(page) {
    if (!page || !page.id) return;
    const container = document.getElementById('recentVisitList');
    if (!container) return;

    let recent = [];
    try {
      recent = JSON.parse(sessionStorage.getItem('recent_visits') || '[]');
    } catch (e) {
      recent = [];
    }

    // 移除已存在的当前页，并推入头部
    recent = recent.filter(item => item.id !== page.id);
    recent.unshift({ id: page.id, title: page.title });
    
    // 仅保留最近 10 个
    recent = recent.slice(0, 10);
    sessionStorage.setItem('recent_visits', JSON.stringify(recent));

    // 渲染，排除当前页
    const others = recent.filter(item => item.id !== page.id);
    if (!others.length) {
      container.innerHTML = '<p class="data-meta">暂无更多最近访问记录</p>';
      return;
    }

    container.innerHTML = others.map(item => {
      return '<a href="detail.html?id=' + encodeURIComponent(item.id) + '" class="data-chip">' + escapeHtml(item.title || item.id) + '</a>';
    }).join('');
  }
})();
