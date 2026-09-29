/* Homepage interface copy only. Never translate route/company/node names or exported content. */
(function () {
  'use strict';
  var button = document.getElementById('homeLanguageToggle');
  if (!button) return;
  var storageKey = 'robotics-home-language', language = 'zh';
  try { if (localStorage.getItem(storageKey) === 'en') language = 'en'; } catch (_) { /* optional */ }
  var copy = {
    'Robotics Notebooks | 机器人技术栈地图': 'Robotics Notebooks | Robotics Knowledge Map',
    '切换白天黑夜模式': 'Toggle light / dark mode',
    '持续更新的机器人技术栈地图': 'An evolving robotics knowledge map',
    '面向人形机器人': 'For humanoid robots', '运动控制的知识入口': 'humanoid motion control',
    '从运动控制到': 'From motion control to ', '物理智能': 'Physical AI',
    '先选一个入口，沿': 'Choose a starting point and explore',
    '路线、图谱、模块、论文': 'roadmaps, graphs, modules and papers', '逐步深入。': 'step by step.',
    '知识库当前规模': 'Knowledge base size', '知识节点': 'Knowledge nodes',
    '互链关系': 'Connections', '主路线': 'Main roadmap', '纵深路线': 'Deep-dive roadmaps',
    '公司路线': 'Company roadmaps', '查看知识图谱（知识节点）': 'View knowledge graph nodes',
    '查看知识图谱（互链关系）': 'View knowledge graph connections',
    '定位到从零开始入口卡': 'Jump to the getting-started card',
    '定位到更多路线入口卡': 'Jump to more roadmaps',
    '定位到公司路线入口卡': 'Jump to company roadmaps',
    '入口': 'Starting points', '从零开始': 'Getting started',
    '建立运动控制全局认识': 'Build an overview of motion control',
    '进入主路线 →': 'Follow the main roadmap →', '项目查询': 'Find a project',
    '查找框架、论文与机器人平台': 'Find frameworks, papers and robot platforms',
    '搜索知识库 →': 'Search the knowledge base →', '知识图谱': 'Knowledge graph',
    '俯瞰节点与连接全貌': 'Explore the nodes and their connections',
    '查看图谱预览 →': 'Preview the graph →', '更多路线': 'More roadmaps',
    '按研究方向进入纵深路线': 'Explore roadmaps by research area',
    '按公司查看具身基础模型与人形技术演进': 'Explore embodied models and humanoid advances by company',
    '收起纵深路线 ↑': 'Collapse roadmaps ↑', '收起公司列表 ↑': 'Collapse companies ↑',
    '搜索知识库': 'Search the knowledge base',
    '搜索概念、方法或任务：MPC、PPO、Diffusion Policy…': 'Search concepts, methods or tasks: MPC, PPO, Diffusion Policy…',
    '搜索知识页面': 'Search knowledge pages', '按社区过滤': 'Filter by community',
    '热门主题': 'Popular topics', '全部社区': 'All communities',
    '最新知识节点': 'Latest knowledge nodes', '加载中…': 'Loading…',
    '知识图谱预览': 'Knowledge graph preview',
    '预览 2D / 3D 切换': 'Switch preview between 2D and 3D',
    '打开完整图谱 →': 'Open the full graph →',
    '互链枢纽 · Top 10': 'Most connected · Top 10',
    '切换全站 / 论文互链枢纽': 'Switch between all pages and papers',
    '全站': 'All pages', '论文': 'Papers',
    '按站内互链总数（无向度数）排序，从与其他条目连接最多的核心页开始探索。': 'Ranked by total connections. Start with the pages that connect to the most other entries.',
    '正在加载互链统计…': 'Loading connection statistics…',
    '查看完整榜单': 'View full rankings',
    '暂无互链统计数据。': 'No connection statistics available.',
    '暂无论文互链统计数据。': 'No paper connection statistics available.',
    '统计加载失败，请稍后刷新。': 'Statistics could not load. Please refresh later.',
    '暂无「最近新增」数据。': 'No recently added nodes available.',
    '查看全部更新 →': 'View all updates →',
    '无向边总数（入链+出链）': 'Total connections (incoming + outgoing)',
    '加载离线搜索索引中…': 'Loading the offline search index…',
    '未找到': 'No results for', '的匹配结果。': '.',
    '尝试更短的关键词，或英文原文': 'Try a shorter keyword or the original English term',
    '命令行搜索：': 'Search from the command line: ', '在': 'Browse the',
    '中浏览相关节点': 'for related nodes',
    '当前筛选条件下暂无索引条目，或数据仍在加载。': 'No entries match these filters, or data is still loading.',
    '搜索暂时无法加载，请检查网络后重试。': 'Search could not load. Check your connection and try again.',
    '重试搜索': 'Retry search', '浏览知识图谱': 'Browse knowledge graph',
    '预览全文': 'Show full summary', '收起': 'Collapse', '🔗图谱': '🔗Graph',
    '查看图谱邻居': 'View graph neighbors', '精确匹配': 'Exact matches',
    '潜在关联': 'Related results', '摘要命中': 'Summary match', '正文匹配': 'Body match',
    '打开路线页 →': 'Open roadmap →', '打开详情页 →': 'Open details →',
    '正在加载 3D 组件…': 'Loading 3D viewer…',
    '3D 组件加载失败，请刷新重试': '3D viewer could not load. Please refresh to retry.',
    '赞助我': 'Support my work', '微信扫一扫，赞助支持作者 ☕': 'Scan with WeChat to support the author ☕',
    '微信收款码': 'WeChat payment QR code', '关闭': 'Close',
    '新增': 'Added', '维护': 'Updated', '回到顶部': 'Back to top',
    '章节导航': 'Section navigation', '打开或关闭章节导航': 'Toggle section navigation',
    '刘冲': 'Chong Liu', '2026 · 机器人技术栈地图': '2026 · Robotics Notebooks'
  };
  var description = '面向人形机器人与 Physical AI 的技术栈导航，系统梳理运动控制、强化学习、模仿学习、Sim2Real、VLA、世界模型与真机部署。';
  copy[description] = 'A knowledge map for humanoid robotics and Physical AI: motion control, reinforcement learning, imitation learning, Sim2Real, VLA, world models and deployment.';
  copy['Robotics Notebooks - ' + description] = 'Robotics Notebooks - ' + copy[description];
  copy['Robotics Notebooks, 机器人技术栈, Humanoid Robot, Motion Control, Reinforcement Learning, Imitation Learning, Whole-Body Control'] =
    'Robotics Notebooks, Robotics Knowledge Map, Humanoid Robot, Motion Control, Reinforcement Learning, Imitation Learning, Whole-Body Control';
  var types = { '概念': 'Concept', '方法': 'Method', '任务': 'Task', '对比': 'Comparison',
    '实体': 'Entity', '查询': 'Query', '形式化': 'Formalization', '总览': 'Overview',
    '参考': 'Reference', '路线': 'Roadmap', '知识页': 'Wiki', '技术节点': 'Tech Node', '详情页': 'Detail Page' };
  Object.keys(types).forEach(function (zh) { copy[zh] = types[zh]; copy[zh + ' (' + types[zh] + ')'] = types[zh]; });

  // This allowlist excludes route links, company links, hot-topic labels, titles, summaries and graph nodes.
  var selector = [
    '[data-home-copy]', '#homeNavToggle', '#homeNavMenu', '#backToTop',
    '#homeRouteToggle', '#homeCompanyToggle', '#wikiCommunityFilter option[value=""]',
    '#homeLatestWikiModule > .data-meta', '#homeLatestWikiModule .home-latest-more a',
    '#homeLatestWikiModule .updates-badge', '.home-latest-row-type',
    '#homeHubPanelAll > .data-meta', '#homeHubPanelPaper > .data-meta',
    '.home-hub-row-degree', '.home-hub-row-type', '#wikiSearchResults > p',
    '#wikiSearchResults > div > p', '#wikiSearchResults > div > ul > li',
    '#wikiSearchResults > div a', '#wikiSearchResults .search-retry',
    '#wikiSearchResults .result-preview-toggle', '#wikiSearchResults .js-graph-btn',
    '#wikiSearchResults .search-tier-heading', '#wikiSearchResults .search-tier-heading .data-meta',
    '#wikiSearchResults .card-meta', '#wikiSearchResults .card-meta > span',
    '#mini-graph-stats', '#mini-graph-tooltip .tt-link', '#mini-graph-tooltip .tt-type',
    '#mini-graph-3d .mini-graph-3d-hint', '#sponsorToggle', '#sponsorDialogTitle',
    '#sponsorDialog .sponsor-dialog-hint', '#sponsorDialog .sponsor-dialog-close',
    '#sponsorDialog .sponsor-qr'
  ].join(',');
  var patterns = [
    [/^展开全部 (\d+) 条纵深路线 ↓$/, 'Show all $1 roadmaps ↓'],
    [/^展开全部 (\d+) 家公司 ↓$/, 'Show all $1 companies ↓'],
    [/^互链 (\d+)$/, 'Connections $1'], [/^· (\d+) 项$/, '· $1 results'],
    [/^预览：全站连接度 Top-(\d+) 枢纽节点$/, 'Preview: top $1 nodes by connections'],
    [/^别名命中: /, 'Alias match: '], [/^核心标签命中: /, 'Tag match: '],
    [/^标题命中: /, 'Title match: ']
  ];
  var originals = new WeakMap();
  function translate(value) {
    var trimmed = value.trim(), translated = copy[trimmed];
    if (translated === undefined) {
      for (var i = 0; i < patterns.length; i++) {
        if (patterns[i][0].test(trimmed)) { translated = trimmed.replace(patterns[i][0], patterns[i][1]); break; }
      }
    }
    return translated === undefined ? value : value.replace(trimmed, function () { return translated; });
  }
  function update(owner, key, current, write) {
    var record = originals.get(owner) || {};
    if (!record[key] || record[key].last !== current) record[key] = { source: current };
    var next = language === 'en' ? translate(record[key].source) : record[key].source;
    record[key].last = next; originals.set(owner, record);
    if (next !== current) write(next);
  }
  function applyElement(el) {
    // Direct text nodes preserve child elements, event listeners and dynamic knowledge content.
    el.childNodes.forEach(function (node) {
      if (node.nodeType === 3) update(node, 'text', node.nodeValue, function (v) { node.nodeValue = v; });
    });
    ['aria-label', 'title', 'placeholder', 'content', 'alt'].forEach(function (attr) {
      if (el.hasAttribute(attr)) update(el, attr, el.getAttribute(attr), function (v) { el.setAttribute(attr, v); });
    });
  }
  function apply(root) {
    if (root.nodeType === 1 && root.matches(selector)) applyElement(root);
    if (root.querySelectorAll) root.querySelectorAll(selector).forEach(applyElement);
  }
  function syncNavigation() {
    document.querySelectorAll('#homeNavMenu a').forEach(function (link) {
      var section = document.getElementById(link.getAttribute('href').slice(1));
      var title = section && section.querySelector('.section-title');
      link.textContent = title ? title.textContent : (language === 'en' ? 'Top' : '顶部');
    });
  }
  function applyLanguage() {
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    button.textContent = language === 'en' ? '中文' : 'English';
    button.setAttribute('aria-label', language === 'en' ? '切换首页为中文' : 'Switch homepage to English');
    button.setAttribute('lang', language === 'en' ? 'zh-CN' : 'en');
    apply(document); syncNavigation();
  }
  button.hidden = false;
  button.addEventListener('click', function () {
    language = language === 'en' ? 'zh' : 'en';
    try { localStorage.setItem(storageKey, language); } catch (_) { /* keep the control usable */ }
    applyLanguage();
  });
  applyLanguage();
  // Observe asynchronous UI mounts only, never the graph's SVG animation attributes or the whole page.
  var observer = new MutationObserver(function (records) {
    var roots = new Set();
    records.forEach(function (record) {
      if (record.type === 'characterData') roots.add(record.target.parentElement);
      else roots.add(record.target);
    });
    roots.forEach(function (root) { if (root) apply(root); });
  });
  function observeMounts() {
    ['homeRouteToggle', 'homeCompanyToggle', 'wikiCommunityFilter', 'wikiSearchResults',
      'homeLatestWikiModule', 'homeHubPanelAll', 'homeHubPanelPaper',
      'mini-graph-stats', 'mini-graph-tooltip', 'mini-graph-3d', 'sponsorDialog'
    ].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) observer.observe(el, { childList: true, subtree: true, characterData: true });
    });
    apply(document); syncNavigation();
  }
  observeMounts();
  document.addEventListener('DOMContentLoaded', function () { setTimeout(observeMounts, 0); }, { once: true });
})();
