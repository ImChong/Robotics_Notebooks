(function () {
  'use strict';

  var DEFAULT_CONFIG = {
    // 部署 Workers Free 网关与 Turnstile 后填写公开配置（严禁在此填写 Secret Key）
    endpoint: 'https://robotics-notebooks-kb-chat.chongliu.workers.dev/chat',
    turnstileSiteKey: '0x4AAAAAAGIWdfoeyhjsNN-h',
    maxQuestionLength: 300,
    requestTimeoutMs: 22000,
  };

  var CONTENT_URL_RE = /^exports\/page-content\/([a-f0-9]{64})\.json$/;
  var SAFE_NODE_URL_RE =
    /^(?:detail|roadmap)\.html\?id=[A-Za-z0-9%._~!*()-]+(?:&(?:cite|cite_h)=[A-Za-z0-9%._~!*()'-]+)*(?:#[A-Za-z0-9%._~!*()'-]+)?$/;
  var WORD_OR_CJK_RE = /[A-Za-z0-9_+\-.\u0370-\u03ff]+|[\u4e00-\u9fff]+/g;
  var CODE_FENCE_RE = /```[\s\S]*?```/g;
  var HTML_TAG_RE = /<[^>]+>/g;
  var RAW_URL_RE = /\b(?:https?:\/\/|javascript:|data:|vbscript:)[^\s)\]<>"']+/gi;
  var TURNSTILE_SCRIPT_URL = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
  var KATEX_CDN_BASE = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/';
  var MERMAID_CDN_URL = 'https://cdn.jsdelivr.net/npm/mermaid@12.3.0/dist/mermaid.min.js';
  var CHAT_HISTORY_KEY = 'rn_kb_chat_history_v1';
  var MAX_STORED_MESSAGES = 40;
  var MAX_STORED_BYTES = 120000;
  var _katexLoadPromise = null;
  var _mermaidLoadPromise = null;

  var SYNONYMS = {
    wbc: ['全身控制', 'whole-body-control'],
    '全身控制': ['wbc', 'whole-body-control', 'whole', 'body', 'control'],
    mpc: ['模型预测控制', 'model-predictive-control'],
    '模型预测控制': ['mpc', 'model-predictive-control', 'predictive'],
    rl: ['强化学习', 'reinforcement-learning'],
    '强化学习': ['rl', 'reinforcement-learning', 'ppo'],
    il: ['模仿学习', 'imitation-learning'],
    '模仿学习': ['il', 'imitation-learning', 'behavior-cloning'],
    bc: ['行为克隆', 'behavior-cloning'],
    '行为克隆': ['bc', 'behavior-cloning', 'imitation-learning'],
    sim2real: ['仿真到真机', 'domain-randomization', '域随机化'],
    vla: ['视觉语言动作', 'vision-language-action'],
    '运动控制': ['locomotion', 'control', 'gait'],
    locomotion: ['运动控制', '步态', 'legged'],
    humanoid: ['人形', 'biped'],
    '人形': ['humanoid', 'biped'],
    manipulation: ['操作', '灵巧手'],
    '操作': ['manipulation', 'grasp'],
    tactile: ['触觉', '力控'],
    '触觉': ['tactile', 'force'],
    teleoperation: ['遥操', '遥操作'],
    '遥操': ['teleoperation', 'vr'],
  };

  var STOP_WORDS = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do', 'does', 'for', 'from',
    'how', 'i', 'in', 'is', 'it', 'me', 'of', 'on', 'or', 'please', 'tell', 'that', 'the',
    'this', 'to', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'you',
    '什么', '怎么', '怎样', '如何', '为什么', '为何', '哪些', '哪个', '请问', '介绍', '一下', '关于',
    '的', '了', '吗', '呢', '吧', '啊', '和', '与', '在', '有', '是', '我', '你', '他', '她', '它',
    '这', '那', '个', '把', '被', '让', '给', '到', '从', '对', '中', '上', '下',
    '帮', '写', '段', '什', '么', '怎', '样', '如', '何', '为', '哪', '请', '问',
  ]);

  var COPY = {
    zh: {
      panelTitle: '知识问答',
      badgeLocal: '站内检索',
      badgeAi: 'AI 回答',
      badgeFallback: '站内检索',
      langSwitch: 'EN',
      langSwitchLabel: 'Switch to English',
      closeLabel: '关闭问答',
      toggleLabel: '知识问答',
      closeToggleLabel: '收起对话',
      mainToggleLabel: '回到顶部 / 知识问答',
      welcome: '基于站内知识库检索作答，每条回答均附原文链接。',
      welcomeAi: '基于站内文档智能作答，并附原文链接；繁忙时自动切回站内检索。',
      placeholder: '问 WBC、RL、Sim2Real…',
      send: '提问',
      sending: '查找中…',
      readonlyNotice: '基于站内文档回答 · 公开问答匿名记录用于改进知识库',
      sourcesLabel: '参考来源',
      showExcerpts: '显示摘录 ▾',
      hideExcerpts: '收起摘录 ▴',
      defaultPassageLabel: '正文摘录',
      ungrounded: '暂未在站内找到直接相关内容，可换个短词、英文术语，或逛逛知识图谱。',
      greetingReply: '你好！我是机器人知识库只读问答助手。请直接提问站内技术概念、方法或论文（如「什么是全身控制 WBC？」），或点击上方示例问题。',
      localAnswerIntro: '找到以下站内相关内容：',
      fallbackRateLimited: '提问太频繁，已为你切为站内检索结果。',
      fallbackQuota: 'AI 助手当前繁忙，已为你切为站内检索结果。',
      fallbackVerify: '安全校验尚未就绪，已自动切换为站内检索结果。',
      fallbackGeneric: 'AI 助手暂时离线，已自动切换为站内检索结果。',
      cachedTag: '来自常见问答缓存',
      emptyError: '请输入至少 2 个字的问题。',
      tooLongError: '问题请控制在 {max} 字以内。',
      starters: ['什么是全身控制 WBC？', 'Sim2Real 如何缩小仿真到真机差距？', 'VLA 模型与传统控制怎么配合？'],
      fallbackLinksText: '浏览知识图谱',
      metaCommunity: '社区',
      metaRoadmap: '路线',
      metaUpdated: '更新',
      metaHealth: '健康度',
      clearShort: '清空',
      clearLabel: '清空本地聊天记录（云端问答记录不受影响）',
      clearedNotice: '已清空本地聊天记录（云端问答记录不会被删除）。',
    },
    en: {
      panelTitle: 'Knowledge Q&A',
      badgeLocal: 'Wiki Search',
      badgeAi: 'AI Answer',
      badgeFallback: 'Wiki Search',
      langSwitch: '中文',
      langSwitchLabel: '切换为中文',
      closeLabel: 'Close Q&A',
      toggleLabel: 'Knowledge Q&A',
      closeToggleLabel: 'Close Q&A',
      mainToggleLabel: 'Back to top / Knowledge Q&A',
      welcome: 'Answers from the wiki, with direct links to source pages.',
      welcomeAi: 'AI answers grounded in the wiki with source links. Switches to wiki search if busy.',
      placeholder: 'Ask about WBC, RL, Sim2Real…',
      send: 'Ask',
      sending: 'Searching…',
      readonlyNotice: 'Grounded in the wiki · Public Q&A logged anonymously',
      sourcesLabel: 'Sources',
      showExcerpts: 'Show excerpts ▾',
      hideExcerpts: 'Hide excerpts ▴',
      defaultPassageLabel: 'Excerpt',
      ungrounded: 'No direct match found in the wiki. Try shorter keywords or explore the Knowledge Graph.',
      greetingReply: 'Hello! I am the read-only Robotics_Notebooks Q&A assistant. Please ask about robotics concepts, methods, or papers (e.g., "What is Whole-Body Control (WBC)?"), or tap a starter prompt above.',
      localAnswerIntro: 'Relevant sections from the wiki:',
      fallbackRateLimited: 'Too many requests — showing wiki search results instead.',
      fallbackQuota: 'AI assistant is busy right now — showing wiki search results instead.',
      fallbackVerify: 'Verification could not complete — showing wiki search results instead.',
      fallbackGeneric: 'AI assistant is unavailable — showing wiki search results instead.',
      cachedTag: 'From popular Q&A cache',
      emptyError: 'Please enter at least 2 characters.',
      tooLongError: 'Please keep your question within {max} characters.',
      starters: [
        'What is Whole-Body Control (WBC)?',
        'How does Sim2Real bridge domain gaps?',
        'How do VLA models connect with low-level control?',
      ],
      fallbackLinksText: 'Browse Knowledge Graph',
      metaCommunity: 'Community',
      metaRoadmap: 'Roadmap',
      metaUpdated: 'Updated',
      metaHealth: 'Health',
      clearShort: 'Clear',
      clearLabel: 'Clear local chat history (server records stay intact)',
      clearedNotice: 'Local chat history cleared (server records are preserved).',
    },
  };

  function getConfig() {
    var raw = typeof window !== 'undefined' && window.RN_KB_CHAT_CONFIG ? window.RN_KB_CHAT_CONFIG : {};
    var merged = {
      endpoint: sanitizeEndpoint(raw.endpoint !== undefined ? raw.endpoint : DEFAULT_CONFIG.endpoint),
      turnstileSiteKey: String(
        raw.turnstileSiteKey !== undefined ? raw.turnstileSiteKey : DEFAULT_CONFIG.turnstileSiteKey || ''
      ).trim(),
      maxQuestionLength: Number(raw.maxQuestionLength) || DEFAULT_CONFIG.maxQuestionLength,
      requestTimeoutMs: Number(raw.requestTimeoutMs) || DEFAULT_CONFIG.requestTimeoutMs,
    };
    return merged;
  }

  function initPanelResize(panel) {
    var handles = [
      { cls: 'kb-chat-resize-left', dx: -1, dy: 0 },
      { cls: 'kb-chat-resize-top', dx: 0, dy: -1 },
      { cls: 'kb-chat-resize-nw', dx: -1, dy: -1 },
    ];
    handles.forEach(function (cfg) {
      var handle = document.createElement('div');
      handle.className = 'kb-chat-resize-handle ' + cfg.cls;
      handle.setAttribute('aria-hidden', 'true');
      handle.addEventListener('pointerdown', function (e) {
        if (e.button !== 0 || window.innerWidth < 768) return;
        e.preventDefault();
        var rect = panel.getBoundingClientRect();
        var startX = e.clientX;
        var startY = e.clientY;
        var startW = rect.width;
        var startH = rect.height;
        var maxW = Math.min(1080, Math.max(320, window.innerWidth - 48));
        var maxH = Math.min(920, Math.max(380, window.innerHeight - 96));

        function onMove(ev) {
          if (cfg.dx !== 0) {
            var nextW = Math.min(maxW, Math.max(320, startW + (startX - ev.clientX)));
            panel.style.width = nextW + 'px';
          }
          if (cfg.dy !== 0) {
            var nextH = Math.min(maxH, Math.max(380, startH + (startY - ev.clientY)));
            panel.style.height = nextH + 'px';
            panel.style.maxHeight = nextH + 'px';
          }
        }
        function onUp() {
          window.removeEventListener('pointermove', onMove);
          window.removeEventListener('pointerup', onUp);
        }
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
      });
      panel.appendChild(handle);
    });
  }

  function sanitizeEndpoint(urlString) {
    var trimmed = String(urlString || '').trim();
    if (!trimmed) return '';
    try {
      var parsed = new URL(trimmed);
      var isHttps = parsed.protocol === 'https:';
      var isLocalHttp =
        parsed.protocol === 'http:' &&
        (parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost');
      if (!isHttps && !isLocalHttp) return '';
      if (parsed.username || parsed.password) return '';
      return parsed.toString();
    } catch (_err) {
      return '';
    }
  }

  function safeInternalNodeUrl(rawUrl, fallbackId) {
    var candidate = String(rawUrl || '').trim();
    if (SAFE_NODE_URL_RE.test(candidate)) {
      return candidate;
    }
    var id = String(fallbackId || '').trim();
    if (!id || /[<>/"'\\]/.test(id) || id.indexOf('..') >= 0) {
      return '';
    }
    return (id.indexOf('roadmap-') === 0 ? 'roadmap.html?id=' : 'detail.html?id=') + encodeURIComponent(id);
  }

  function detectInitialLang() {
    try {
      if (typeof document !== 'undefined' && document.documentElement) {
        var docLang = String(document.documentElement.getAttribute('lang') || '').toLowerCase();
        if (docLang.indexOf('en') === 0) return 'en';
      }
      if (typeof localStorage !== 'undefined' && localStorage.getItem('robotics-home-language') === 'en') {
        return 'en';
      }
    } catch (_err) {
      /* ignore storage errors */
    }
    return 'zh';
  }

  function tokenizeQuery(text) {
    var matches = String(text || '').toLowerCase().match(WORD_OR_CJK_RE);
    if (!matches) return [];
    var seen = new Set();
    var out = [];
    function pushToken(tok) {
      if (!tok || STOP_WORDS.has(tok) || seen.has(tok)) return;
      seen.add(tok);
      out.push(tok);
    }
    for (var i = 0; i < matches.length; i += 1) {
      var raw = matches[i];
      var isCjk = raw.charCodeAt(0) >= 0x4e00 && raw.charCodeAt(0) <= 0x9fff;
      if (!isCjk || raw.length <= 4) pushToken(raw);
      if (SYNONYMS[raw]) {
        for (var s = 0; s < SYNONYMS[raw].length; s += 1) pushToken(SYNONYMS[raw][s]);
      }
      if (isCjk) {
        if (raw.length === 1) {
          pushToken(raw);
        } else {
          for (var k = 0; k < raw.length - 1; k += 1) {
            var bg = raw.slice(k, k + 2);
            pushToken(bg);
            if (SYNONYMS[bg]) {
              for (var sb = 0; sb < SYNONYMS[bg].length; sb += 1) pushToken(SYNONYMS[bg][sb]);
            }
          }
        }
      }
    }
    return out;
  }

  function cleanMarkdown(text) {
    return String(text || '')
      .replace(CODE_FENCE_RE, ' ')
      .replace(HTML_TAG_RE, ' ')
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(RAW_URL_RE, ' ')
      .replace(/(?:^|\s)\|?(?:\s*:?-{2,}:?\s*\|)+(?:\s*:?-{2,}:?\s*)?\|?/g, ' ')
      .replace(/[`*_>#~|]/g, ' ')
      .replace(/\\([\\`*_{}[\]()#+\-.!|])/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function splitSections(markdown) {
    var withoutCode = String(markdown || '').replace(CODE_FENCE_RE, '\n');
    var lines = withoutCode.split(/\r?\n/);
    var sections = [];
    var heading = '';
    var buffer = [];

    function flush() {
      var cleaned = cleanMarkdown(buffer.join(' '));
      if (cleaned.length >= 8) {
        sections.push({ heading: heading || '概要', text: cleaned });
      }
      buffer = [];
    }

    for (var i = 0; i < lines.length; i += 1) {
      var line = lines[i].trim();
      if (!line) {
        if (buffer.length > 0 && buffer.join(' ').length > 260) flush();
        continue;
      }
      var hm = line.match(/^#{1,4}\s+(.+)$/);
      if (hm) {
        flush();
        heading = cleanMarkdown(hm[1]) || heading;
        continue;
      }
      if (/^---+$/.test(line)) continue;
      buffer.push(line);
    }
    flush();
    return sections;
  }

  var _liteIndex = null;
  var _litePromise = null;
  var _catalogPromise = null;
  var _bodyCache = new Map();
  var _turnstileScriptPromise = null;

  function fetchJsonWithTimeout(url, timeoutMs) {
    var controller = new AbortController();
    var timer = setTimeout(function () {
      controller.abort();
    }, timeoutMs || 15000);
    return fetch(url, { signal: controller.signal })
      .then(function (resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return resp.json();
      })
      .finally(function () {
        clearTimeout(timer);
      });
  }

  function ensureLiteIndex() {
    if (_liteIndex) return Promise.resolve(_liteIndex);
    if (_litePromise) return _litePromise;
    var sharedPromise = (typeof window !== 'undefined' && window.RNSearchIndexLite &&
        typeof window.RNSearchIndexLite.ensure === 'function')
      ? window.RNSearchIndexLite.ensure()
      : fetchJsonWithTimeout('search-index-lite.json', 20000);
    _litePromise = Promise.resolve(sharedPromise)
      .then(function (data) {
        if (!data || !Array.isArray(data.docs)) throw new Error('Invalid lite search index');
        _liteIndex = data;
        return data;
      })
      .catch(function (err) {
        _litePromise = null;
        throw err;
      });
    return _litePromise;
  }

  var _communityIndex = null;
  var _communityIndexPromise = null;

  function ensureCatalog() {
    var promise;
    if (typeof window !== 'undefined' && window.RNSiteCatalog && typeof window.RNSiteCatalog.ensure === 'function') {
      promise = window.RNSiteCatalog.ensure();
    } else {
      if (!_catalogPromise) {
        _catalogPromise = fetchJsonWithTimeout('exports/site-catalog-v1.json', 20000).catch(function (err) {
          _catalogPromise = null;
          throw err;
        });
      }
      promise = _catalogPromise;
    }
    return promise.then(function (catalog) {
      if (catalog && catalog !== state.catalog) {
        state.catalog = catalog;
        state.roadmapByNodeId = buildNodeRoadmapIndex(catalog);
      }
      return catalog;
    });
  }

  function shortenCommunityLabel(label) {
    if (!label) return '';
    var base = String(label).replace(/\s*社区\s*$/, '').trim();
    if (!base || base === '未分类') return '';
    var head = base.split('（', 1)[0].trim();
    return head || base;
  }

  function shortenRoadmapTitle(title) {
    var text = String(title || '').trim();
    if (!text) return '';
    var head = text.split('（', 1)[0].trim();
    return head || text;
  }

  function buildNodeRoadmapIndex(catalog) {
    var byNodeId = new Map();
    var roadmaps = (catalog && catalog.pages && catalog.pages.roadmap_pages) || {};
    Object.keys(roadmaps).forEach(function (roadmapId) {
      var rm = roadmaps[roadmapId];
      if (!rm) return;
      var rmLabel = shortenRoadmapTitle(rm.title || roadmapId);
      if (!byNodeId.has(roadmapId)) byNodeId.set(roadmapId, []);
      var selfList = byNodeId.get(roadmapId);
      if (selfList.indexOf(rmLabel) < 0) selfList.push(rmLabel);
      var stages = Array.isArray(rm.stages) ? rm.stages : [];
      for (var si = 0; si < stages.length; si += 1) {
        var items = Array.isArray(stages[si] && stages[si].related_items) ? stages[si].related_items : [];
        for (var ii = 0; ii < items.length; ii += 1) {
          var itemId = items[ii];
          if (!itemId) continue;
          if (!byNodeId.has(itemId)) byNodeId.set(itemId, []);
          var arr = byNodeId.get(itemId);
          if (arr.indexOf(rmLabel) < 0) arr.push(rmLabel);
        }
      }
    });
    return byNodeId;
  }

  function ensureCommunityIndex() {
    if (_communityIndex) return Promise.resolve(_communityIndex);
    if (_communityIndexPromise) return _communityIndexPromise;
    if (typeof window !== 'undefined' && window.RNCommunityIndex && typeof window.RNCommunityIndex.ensure === 'function') {
      _communityIndexPromise = window.RNCommunityIndex.ensure()
        .then(function (idx) {
          _communityIndex = idx || { pathToCommunity: new Map(), communityLabel: {} };
          _communityIndexPromise = null;
          return _communityIndex;
        })
        .catch(function () {
          _communityIndexPromise = null;
          return { pathToCommunity: new Map(), communityLabel: {} };
        });
      return _communityIndexPromise;
    }
    _communityIndexPromise = fetchJsonWithTimeout('exports/link-graph.json', 20000)
      .then(function (data) {
        var pathToCommunity = new Map();
        var nodes = Array.isArray(data && data.nodes) ? data.nodes : [];
        for (var i = 0; i < nodes.length; i += 1) {
          var node = nodes[i];
          if (node && node.id && node.community) {
            pathToCommunity.set(node.id, node.community);
          }
        }
        var communityLabel = {};
        var communities = Array.isArray(data && data.communities) ? data.communities : [];
        for (var j = 0; j < communities.length; j += 1) {
          var c = communities[j];
          if (c && c.id) communityLabel[c.id] = c.label || c.id;
        }
        _communityIndex = { pathToCommunity: pathToCommunity, communityLabel: communityLabel };
        _communityIndexPromise = null;
        return _communityIndex;
      })
      .catch(function () {
        _communityIndexPromise = null;
        return { pathToCommunity: new Map(), communityLabel: {} };
      });
    return _communityIndexPromise;
  }

  function resolveCitationMeta(item) {
    var id = item && item.id ? String(item.id) : '';
    var pages = (state.catalog && state.catalog.pages) || {};
    var pageMeta =
      (pages.detail_pages && id && pages.detail_pages[id]) ||
      (pages.roadmap_pages && id && pages.roadmap_pages[id]) ||
      null;
    var nodeTitle = (pageMeta && pageMeta.title) || (item && item.title) || id;
    var updated = (pageMeta && pageMeta.updated) || (item && item.updated) || '';
    var healthScore =
      pageMeta && typeof pageMeta.health_score === 'number'
        ? pageMeta.health_score
        : item && typeof item.health_score === 'number'
          ? item.health_score
          : null;
    var path = (pageMeta && pageMeta.path) || (item && item.path) || '';
    var roadmaps =
      (state.roadmapByNodeId && id && state.roadmapByNodeId.get(id)) ||
      (Array.isArray(item && item.roadmaps) ? item.roadmaps : []);
    var communityLabel = (pageMeta && pageMeta.community_label) || (item && item.community_label) || '';
    if (!communityLabel && _communityIndex && path && _communityIndex.pathToCommunity) {
      var cid = _communityIndex.pathToCommunity.get(path);
      if (cid && _communityIndex.communityLabel && _communityIndex.communityLabel[cid]) {
        communityLabel = _communityIndex.communityLabel[cid];
      }
    }
    return {
      title: String(nodeTitle || ''),
      path: String(path || ''),
      community: shortenCommunityLabel(communityLabel),
      roadmaps: roadmaps,
      updated: String(updated || ''),
      healthScore: healthScore,
    };
  }

  function createHealthBarEl(score) {
    var clamped = Math.max(0, Math.min(3, Math.round(score)));
    var wrap = document.createElement('span');
    wrap.className = 'kb-chat-citation-health';
    var bar = document.createElement('span');
    bar.className = 'node-health node-health--' + clamped;
    bar.setAttribute('aria-hidden', 'true');
    for (var i = 0; i < 3; i += 1) {
      var seg = document.createElement('i');
      if (i < clamped) seg.className = 'is-on';
      bar.appendChild(seg);
    }
    var txt = document.createElement('span');
    txt.textContent = t('metaHealth') + ' ' + clamped + '/3';
    wrap.appendChild(bar);
    wrap.appendChild(txt);
    return wrap;
  }

  function fetchPageContent(contentUrl) {
    var match = CONTENT_URL_RE.exec(String(contentUrl || ''));
    if (!match) return Promise.resolve(null);
    var canonicalUrl = 'exports/page-content/' + match[1] + '.json';
    if (_bodyCache.has(canonicalUrl)) return Promise.resolve(_bodyCache.get(canonicalUrl));
    return fetchJsonWithTimeout(canonicalUrl, 12000)
      .then(function (body) {
        if (!body || body.version !== 'v1' || typeof body.id !== 'string' || typeof body.content_markdown !== 'string') {
          return null;
        }
        _bodyCache.set(canonicalUrl, body);
        return body;
      })
      .catch(function () {
        return null;
      });
  }

  function currentPageContextId(catalog) {
    try {
      var params = new URLSearchParams(window.location.search || '');
      var rawId = params.get('id');
      if (!rawId) return '';
      var aliases = (catalog && catalog.pages && catalog.pages.page_aliases) || {};
      return Object.prototype.hasOwnProperty.call(aliases, rawId) ? aliases[rawId] : rawId;
    } catch (_err) {
      return '';
    }
  }

  function rankCandidateDocs(docs, question, currentId) {
    var tokens = tokenizeQuery(question);
    if (!tokens.length) return [];

    var TOP_K = 6;
    var top = [];
    for (var i = 0; i < docs.length; i += 1) {
      var doc = docs[i];
      if (!doc || !doc.id) continue;
      var all = doc._kbAll;
      if (all === undefined) {
        var title = String(doc.title || '').toLowerCase();
        var summary = String(doc.summary || '').toLowerCase();
        var tags = Array.isArray(doc.tags) ? doc.tags.join(' ').toLowerCase() : '';
        var aliases = Array.isArray(doc.search_aliases) ? doc.search_aliases.join(' ').toLowerCase() : '';
        var path = String(doc.path || '').toLowerCase();
        doc._kbTitle = title;
        doc._kbSummary = summary;
        doc._kbTags = tags;
        doc._kbAliases = aliases;
        doc._kbPath = path;
        all = doc._kbAll = aliases + '\n' + title + '\n' + tags + '\n' + summary + '\n' + path;
      }

      var anyHit = false;
      for (var t0 = 0; t0 < tokens.length; t0 += 1) {
        if (all.indexOf(tokens[t0]) >= 0) {
          anyHit = true;
          break;
        }
      }
      if (!anyHit) continue;

      var dAliases = doc._kbAliases;
      var dTitle = doc._kbTitle;
      var dTags = doc._kbTags;
      var dSummary = doc._kbSummary;
      var dPath = doc._kbPath;
      var score = 0;
      for (var t = 0; t < tokens.length; t += 1) {
        var tok = tokens[t];
        var w = tok.length >= 2 ? (tok.length >= 4 ? 3 : 2) : 0.4;
        if (dAliases.indexOf(tok) >= 0) score += w * 4.5;
        if (dTitle.indexOf(tok) >= 0) score += w * 4;
        if (dTags.indexOf(tok) >= 0) score += w * 2.5;
        if (dSummary.indexOf(tok) >= 0) score += w * 2;
        if (dPath.indexOf(tok) >= 0) score += w * 1.5;
      }
      if (currentId && doc.id === currentId && score > 0) {
        score += 3;
      }
      if (score >= 1.5) {
        if (top.length === TOP_K && score <= top[TOP_K - 1].score) continue;
        var pos = top.length;
        while (pos > 0 && top[pos - 1].score < score) pos -= 1;
        if (pos < TOP_K) {
          top.splice(pos, 0, { doc: doc, score: score });
          if (top.length > TOP_K) top.pop();
        }
      }
    }

    var out = new Array(top.length);
    for (var k = 0; k < top.length; k += 1) out[k] = top[k].doc;
    return out;
  }

  function selectLocalExcerpts(bodies, docMetaById, question) {
    var tokens = tokenizeQuery(question);
    if (!tokens.length) return [];
    var candidates = [];

    for (var i = 0; i < bodies.length; i += 1) {
      var body = bodies[i];
      if (!body) continue;
      var meta = docMetaById.get(body.id) || {};
      var title = meta.title || body.id;
      var titleLower = String(title).toLowerCase();
      var sections = splitSections(body.content_markdown);

      for (var s = 0; s < sections.length; s += 1) {
        var sec = sections[s];
        var hLower = sec.heading.toLowerCase();
        var tLower = sec.text.toLowerCase();
        var score = 0;
        for (var k = 0; k < tokens.length; k += 1) {
          var tok = tokens[k];
          var w = tok.length >= 2 ? (tok.length >= 4 ? 3 : 2) : 0.4;
          if (titleLower.indexOf(tok) >= 0) score += w * 1.6;
          if (hLower.indexOf(tok) >= 0) score += w * 1.4;
          if (tLower.indexOf(tok) >= 0) score += w;
        }
        if (score >= 1.5) {
          candidates.push({
            id: body.id,
            title: title,
            heading: sec.heading,
            snippet: sec.text.length > 260 ? sec.text.slice(0, 259) + '…' : sec.text,
            score: score,
          });
        }
      }
    }

    candidates.sort(function (a, b) {
      return b.score - a.score;
    });

    var out = [];
    var seen = new Set();
    for (var c = 0; c < candidates.length && out.length < 4; c += 1) {
      var item = candidates[c];
      var key = item.id + '::' + item.heading;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({
        index: out.length + 1,
        id: item.id,
        title: item.title,
        heading: item.heading,
        snippet: item.snippet,
        url: safeInternalNodeUrl('', item.id),
      });
    }
    return out;
  }

  function ensureTurnstileReady() {
    if (typeof window !== 'undefined' && window.turnstile && typeof window.turnstile.render === 'function') {
      return Promise.resolve(window.turnstile);
    }
    if (_turnstileScriptPromise) return _turnstileScriptPromise;
    _turnstileScriptPromise = new Promise(function (resolve, reject) {
      var script = document.createElement('script');
      script.src = TURNSTILE_SCRIPT_URL;
      script.async = true;
      script.defer = true;
      script.onload = function () {
        if (window.turnstile && typeof window.turnstile.render === 'function') {
          resolve(window.turnstile);
        } else {
          _turnstileScriptPromise = null;
          var err = new Error('Turnstile API missing');
          err.code = 'TURNSTILE_API_MISSING';
          reject(err);
        }
      };
      script.onerror = function () {
        _turnstileScriptPromise = null;
        var err = new Error('Failed to load Turnstile');
        err.code = 'TURNSTILE_SCRIPT_ERROR';
        reject(err);
      };
      document.head.appendChild(script);
    });
    return _turnstileScriptPromise;
  }

  var state = {
    lang: detectInitialLang(),
    open: false,
    busy: false,
    mode: 'local', // 'local' | 'ai' | 'fallback'
    panel: null,
    messagesEl: null,
    badgeEl: null,
    titleEl: null,
    welcomeEl: null,
    startersEl: null,
    catalog: null,
    roadmapByNodeId: null,
    history: [],
    historyRestored: false,
    turnstileEl: null,
    turnstileWidgetId: null,
    prewarmedToken: null,
    turnstileInflight: null,
    inputEl: null,
    sendBtn: null,
    counterEl: null,
    noticeEl: null,
    clearBtn: null,
    langBtn: null,
    closeBtn: null,
  };

  function isCloudConfigured(cfg) {
    return Boolean(cfg.endpoint && cfg.turnstileSiteKey);
  }

  function t(key, vars) {
    var dict = COPY[state.lang] || COPY.zh;
    var text = dict[key] || COPY.zh[key] || '';
    if (!vars) return text;
    Object.keys(vars).forEach(function (k) {
      text = text.replace('{' + k + '}', String(vars[k]));
    });
    return text;
  }

  function syncToggleButtonLabels() {
    var toggle = document.getElementById('kbChatToggle');
    if (toggle) {
      toggle.setAttribute('aria-expanded', state.open ? 'true' : 'false');
      var label = state.open ? t('closeToggleLabel') : t('toggleLabel');
      toggle.setAttribute('aria-label', label);
      toggle.setAttribute('title', label);
    }
    var mainToggle = document.getElementById('fabMainToggle');
    if (mainToggle) {
      var mainLabel = t('mainToggleLabel');
      mainToggle.setAttribute('aria-label', mainLabel);
      mainToggle.setAttribute('title', mainLabel);
    }
  }

  function updatePanelCopy() {
    if (!state.panel) return;
    var cfg = getConfig();
    state.panel.setAttribute('aria-label', t('panelTitle'));
    state.titleEl.textContent = t('panelTitle');
    if (state.mode === 'ai') {
      state.badgeEl.textContent = t('badgeAi');
      state.badgeEl.setAttribute('data-mode', 'ai');
    } else if (state.mode === 'fallback') {
      state.badgeEl.textContent = t('badgeFallback');
      state.badgeEl.setAttribute('data-mode', 'fallback');
    } else {
      state.badgeEl.textContent = t('badgeLocal');
      state.badgeEl.setAttribute('data-mode', 'local');
    }
    state.langBtn.textContent = t('langSwitch');
    state.langBtn.setAttribute('aria-label', t('langSwitchLabel'));
    if (state.clearBtn) {
      state.clearBtn.textContent = t('clearShort');
      state.clearBtn.setAttribute('aria-label', t('clearLabel'));
      state.clearBtn.setAttribute('title', t('clearLabel'));
      state.clearBtn.disabled = state.history.length === 0;
    }
    state.closeBtn.setAttribute('aria-label', t('closeLabel'));
    state.closeBtn.setAttribute('title', t('closeLabel'));
    state.welcomeEl.textContent = isCloudConfigured(cfg) ? t('welcomeAi') : t('welcome');
    state.inputEl.setAttribute('placeholder', t('placeholder'));
    state.inputEl.setAttribute('aria-label', t('placeholder'));
    state.sendBtn.textContent = state.busy ? t('sending') : t('send');
    state.noticeEl.textContent = t('readonlyNotice');
    updateCounter();
    renderStarters();
    syncToggleButtonLabels();
  }

  function renderStarters() {
    if (!state.startersEl) return;
    while (state.startersEl.firstChild) {
      state.startersEl.removeChild(state.startersEl.firstChild);
    }
    var starters = (COPY[state.lang] || COPY.zh).starters || [];
    for (var i = 0; i < starters.length; i += 1) {
      var qText = starters[i];
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'kb-chat-starter-btn';
      btn.textContent = qText;
      btn.addEventListener('click', (function (promptText) {
        return function () {
          if (state.busy) return;
          state.inputEl.value = promptText;
          updateCounter();
          submitQuestion(promptText);
        };
      })(qText));
      state.startersEl.appendChild(btn);
    }
  }

  function updateCounter() {
    if (!state.inputEl || !state.counterEl) return;
    var cfg = getConfig();
    var len = String(state.inputEl.value || '').length;
    state.counterEl.textContent = len + '/' + cfg.maxQuestionLength;
  }

  function isCurrencyDollarExpr(expr) {
    var s = String(expr || '').trim();
    if (!s) return false;
    if (/[\u3400-\u9fff]/.test(s)) return true;
    if (/\*\*/.test(s)) return true;
    return /^\d[\d,]*(?:\.\d+)?(?:[kKmMbB])?\s*(?:[–\-—/]|…|\.{2,3})\s*$/.test(s);
  }

  function normalizeMathText(text) {
    if (typeof window !== 'undefined' && window.RNMath && typeof window.RNMath.normalizeDollarMath === 'function') {
      return window.RNMath.normalizeDollarMath(text);
    }
    return String(text || '').replace(/\$\$[\s\S]+?\$\$|\$\s*([^$]+?)\s*\$/g, function (match, expr) {
      if (expr == null) return match;
      var trimmed = String(expr).trim();
      if (!trimmed || isCurrencyDollarExpr(trimmed)) return match;
      return '\\(' + trimmed.replace(/\\\*/g, '*') + '\\)';
    });
  }

  function containerHasMathDelimiters(container) {
    if (!container) return false;
    var text = container.textContent || '';
    return text.indexOf('$$') >= 0 || text.indexOf('\\[') >= 0 || text.indexOf('\\(') >= 0;
  }

  function loadHeadTag(tagName, attrs) {
    return new Promise(function (resolve, reject) {
      var el = document.createElement(tagName);
      Object.keys(attrs).forEach(function (key) {
        el[key] = attrs[key];
      });
      el.onload = function () {
        resolve();
      };
      el.onerror = function () {
        if (el.parentNode) el.parentNode.removeChild(el);
        reject(new Error('Failed to load ' + (attrs.src || attrs.href || tagName)));
      };
      document.head.appendChild(el);
    });
  }

  function ensureChatKatexLoaded() {
    if (typeof window !== 'undefined' && typeof window.renderMathInElement === 'function') {
      return Promise.resolve();
    }
    if (_katexLoadPromise) return _katexLoadPromise;
    _katexLoadPromise = Promise.all([
      loadHeadTag('link', {
        rel: 'stylesheet',
        href: KATEX_CDN_BASE + 'katex.min.css',
        integrity: 'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+',
        crossOrigin: 'anonymous',
      }),
      loadHeadTag('script', {
        src: KATEX_CDN_BASE + 'katex.min.js',
        integrity: 'sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg',
        crossOrigin: 'anonymous',
      }).then(function () {
        return loadHeadTag('script', {
          src: KATEX_CDN_BASE + 'contrib/auto-render.min.js',
          integrity: 'sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk',
          crossOrigin: 'anonymous',
        });
      }),
    ]).catch(function (err) {
      _katexLoadPromise = null;
      throw err;
    });
    return _katexLoadPromise;
  }

  function renderMessageMath(container) {
    if (!containerHasMathDelimiters(container)) return;
    var renderPromise;
    if (typeof window !== 'undefined' && window.RNMath && typeof window.RNMath.render === 'function') {
      renderPromise = Promise.resolve(window.RNMath.render(container));
    } else {
      renderPromise = ensureChatKatexLoaded().then(function () {
        if (typeof window === 'undefined' || typeof window.renderMathInElement !== 'function') return;
        window.renderMathInElement(container, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
          ],
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
          throwOnError: false,
        });
      });
    }
    renderPromise
      .then(function () {
        if (state.messagesEl) {
          state.messagesEl.scrollTop = state.messagesEl.scrollHeight;
        }
      })
      .catch(function () {});
  }

  function loadChatHistory() {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return [];
      var raw = window.localStorage.getItem(CHAT_HISTORY_KEY);
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(function (entry) {
        return entry && (entry.role === 'user' || entry.role === 'assistant') && typeof entry.text === 'string';
      });
    } catch (_err) {
      return [];
    }
  }

  function saveChatHistory(entries) {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return;
      var list = entries.slice(-MAX_STORED_MESSAGES);
      var json = JSON.stringify(list);
      while (list.length > 1 && json.length > MAX_STORED_BYTES) {
        list.shift();
        json = JSON.stringify(list);
      }
      if (json.length <= MAX_STORED_BYTES) {
        window.localStorage.setItem(CHAT_HISTORY_KEY, json);
      }
    } catch (_err) {
      // ignore quota / private mode storage errors
    }
  }

  function syncClearButton() {
    if (state.clearBtn) {
      state.clearBtn.disabled = state.history.length === 0;
    }
  }

  function recordHistoryEntry(entry) {
    state.history.push(entry);
    if (state.history.length > MAX_STORED_MESSAGES) {
      state.history = state.history.slice(-MAX_STORED_MESSAGES);
    }
    saveChatHistory(state.history);
    syncClearButton();
  }

  function clearLocalHistory() {
    state.history = [];
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.removeItem(CHAT_HISTORY_KEY);
      }
    } catch (_err) {
      // ignore storage errors
    }
    if (state.messagesEl) {
      var msgEls = state.messagesEl.querySelectorAll('.kb-chat-msg');
      for (var i = 0; i < msgEls.length; i += 1) {
        if (msgEls[i].parentNode) {
          msgEls[i].parentNode.removeChild(msgEls[i]);
        }
      }
      state.messagesEl.scrollTop = 0;
    }
    syncClearButton();
  }

  function ensureChatMermaidLoaded() {
    if (typeof window !== 'undefined' && typeof window.mermaid !== 'undefined') {
      return Promise.resolve();
    }
    if (_mermaidLoadPromise) return _mermaidLoadPromise;
    _mermaidLoadPromise = loadHeadTag('script', {
      src: MERMAID_CDN_URL,
      crossOrigin: 'anonymous',
    }).catch(function (err) {
      _mermaidLoadPromise = null;
      throw err;
    });
    return _mermaidLoadPromise;
  }

  var _fallbackLightboxEl = null;

  function openFallbackMermaidLightbox(svgEl) {
    if (!svgEl || typeof document === 'undefined') return;
    if (!_fallbackLightboxEl) {
      var overlay = document.createElement('div');
      overlay.className = 'mermaid-lightbox';
      overlay.hidden = true;
      var backdrop = document.createElement('div');
      backdrop.className = 'mermaid-lightbox-backdrop';
      var panel = document.createElement('div');
      panel.className = 'mermaid-lightbox-panel';
      var head = document.createElement('div');
      head.className = 'mermaid-lightbox-head';
      var title = document.createElement('span');
      title.className = 'mermaid-lightbox-title';
      title.textContent = 'Mermaid';
      var closeBtn = document.createElement('button');
      closeBtn.type = 'button';
      closeBtn.className = 'mermaid-lightbox-close';
      closeBtn.textContent = '×';
      closeBtn.addEventListener('click', function () {
        overlay.hidden = true;
      });
      backdrop.addEventListener('click', function () {
        overlay.hidden = true;
      });
      head.appendChild(title);
      head.appendChild(closeBtn);
      var body = document.createElement('div');
      body.className = 'mermaid-lightbox-body';
      panel.appendChild(head);
      panel.appendChild(body);
      overlay.appendChild(backdrop);
      overlay.appendChild(panel);
      document.body.appendChild(overlay);
      _fallbackLightboxEl = { overlay: overlay, body: body };
    }
    var clone = svgEl.cloneNode(true);
    clone.style.maxWidth = '100%';
    clone.style.maxHeight = '100%';
    clone.style.width = 'auto';
    clone.style.height = 'auto';
    _fallbackLightboxEl.body.textContent = '';
    _fallbackLightboxEl.body.appendChild(clone);
    _fallbackLightboxEl.overlay.hidden = false;
  }

  var MERMAID_HEADER_RE =
    /^(?:flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|gantt|pie|gitGraph|mindmap|timeline|xychart-beta|quadrantChart|journey|sankey-beta|block-beta)\b/;

  function sanitizeMermaidSource(rawText) {
    var src = String(rawText || '');
    var lines = src.replace(/\r\n?/g, '\n').split('\n');
    var firstNonEmpty = '';
    for (var i = 0; i < lines.length; i += 1) {
      var trimmed = lines[i].trim();
      if (trimmed) {
        firstNonEmpty = trimmed;
        break;
      }
    }
    if (/^(?:flowchart|graph)\b/i.test(firstNonEmpty)) {
      src = src.replace(/((?:--|-\.-|==)?>|---)\s*\|([^|\n]+)\|>/g, '$1|$2|');
    }
    return src;
  }

  function extractMermaidBlockSource(fenceInfo, codeLines) {
    var info = String(fenceInfo || '').trim();
    var infoLower = info.toLowerCase();
    var rawText = codeLines.join('\n');
    if (infoLower === 'mermaid' || infoLower === 'mmd' || infoLower === 'mermaidjs') {
      return sanitizeMermaidSource(rawText);
    }
    if (/^mermaid\s+/i.test(info)) {
      var rest = info.replace(/^mermaid\s+/i, '').trim();
      return sanitizeMermaidSource(rest ? rest + '\n' + rawText : rawText);
    }
    if (!infoLower || infoLower === 'text' || infoLower === 'plaintext' || infoLower === 'diagram') {
      var firstNonEmpty = '';
      for (var i = 0; i < codeLines.length; i += 1) {
        var tLine = String(codeLines[i] || '').trim();
        if (tLine) {
          firstNonEmpty = tLine;
          break;
        }
      }
      if (MERMAID_HEADER_RE.test(firstNonEmpty)) {
        return sanitizeMermaidSource(rawText);
      }
    }
    return null;
  }

  function restoreMermaidFallbackIfFailed(nodes) {
    for (var i = 0; i < nodes.length; i += 1) {
      var host = nodes[i];
      if (!host || !host.parentNode) continue;
      var svg = host.querySelector('svg');
      var isErrSvg =
        svg &&
        (svg.getAttribute('aria-roledescription') === 'error' ||
          /Syntax error in text/i.test(svg.textContent || ''));
      if (!svg || isErrSvg) {
        var raw = host.getAttribute('data-raw-mermaid') || host.textContent || '';
        var pre = document.createElement('pre');
        pre.className = 'kb-chat-md-pre';
        var codeEl = document.createElement('code');
        codeEl.textContent = raw;
        pre.appendChild(codeEl);
        host.parentNode.replaceChild(pre, host);
      }
    }
  }

  function renderMessageMermaid(container) {
    if (!container || typeof container.querySelectorAll !== 'function') return;
    var nodes = Array.prototype.slice.call(container.querySelectorAll('.mermaid'));
    if (!nodes.length) return;
    for (var k = 0; k < nodes.length; k += 1) {
      if (!nodes[k].getAttribute('data-raw-mermaid')) {
        nodes[k].setAttribute('data-raw-mermaid', nodes[k].textContent || '');
      }
    }

    var renderPromise;
    if (typeof window !== 'undefined' && window.RNMermaid && typeof window.RNMermaid.render === 'function') {
      renderPromise = Promise.resolve(window.RNMermaid.render(container));
    } else {
      renderPromise = ensureChatMermaidLoaded().then(function () {
        if (typeof window === 'undefined' || !window.mermaid) return;
        var isDark =
          typeof document !== 'undefined' &&
          document.documentElement &&
          document.documentElement.getAttribute('data-theme') === 'dark';
        window.mermaid.initialize({
          startOnLoad: false,
          theme: isDark ? 'dark' : 'default',
          securityLevel: 'strict',
        });
        return Promise.resolve(window.mermaid.run({ nodes: nodes })).then(function () {
          for (var i = 0; i < nodes.length; i += 1) {
            (function (host) {
              var svg = host.querySelector('svg');
              if (!svg || svg.getAttribute('aria-roledescription') === 'error') return;
              host.classList.add('mermaid-zoomable');
              host.setAttribute('tabindex', '0');
              host.setAttribute('role', 'button');
              host.addEventListener('click', function () {
                openFallbackMermaidLightbox(svg);
              });
            })(nodes[i]);
          }
        });
      });
    }

    renderPromise
      .then(function () {
        restoreMermaidFallbackIfFailed(nodes);
        if (state.messagesEl) {
          state.messagesEl.scrollTop = state.messagesEl.scrollHeight;
        }
      })
      .catch(function () {
        restoreMermaidFallbackIfFailed(nodes);
      });
  }

  function appendUserMessage(text, skipSave) {
    var msg = document.createElement('div');
    msg.className = 'kb-chat-msg kb-chat-msg--user';
    msg.textContent = normalizeMathText(text);
    state.messagesEl.appendChild(msg);
    renderMessageMath(msg);
    state.messagesEl.scrollTop = state.messagesEl.scrollHeight;
    if (!skipSave) {
      recordHistoryEntry({ role: 'user', text: String(text || '') });
    }
  }

  function extractCiteParams(item) {
    var rawSnippet = String((item && item.snippet) || '')
      .replace(/…+$/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    var citeText = rawSnippet.slice(0, 64).trim();
    var citeHeading = String((item && item.heading) || '')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 64)
      .trim();
    return { citeText: citeText, citeHeading: citeHeading };
  }

  function buildCitationHref(item) {
    var baseHref = safeInternalNodeUrl(item && item.url, item && item.id);
    if (!baseHref) return '';
    var params = extractCiteParams(item);
    var suffix = '';
    if (params.citeText) {
      suffix += '&cite=' + encodeURIComponent(params.citeText);
    }
    if (params.citeHeading) {
      suffix += '&cite_h=' + encodeURIComponent(params.citeHeading);
    }
    var fullHref = baseHref + suffix;
    return SAFE_NODE_URL_RE.test(fullHref) ? fullHref : baseHref;
  }

  function bindSamePageCitationJump(anchorEl, href, item) {
    if (!anchorEl || !href) return;
    anchorEl.addEventListener('click', function (event) {
      if (typeof window === 'undefined' || typeof window.RNDetailCiteJump !== 'function') return;
      try {
        var cur = new URL(window.location.href);
        var targetUrl = new URL(href, window.location.href);
        if (cur.pathname === targetUrl.pathname && cur.searchParams.get('id') === targetUrl.searchParams.get('id')) {
          event.preventDefault();
          var params = extractCiteParams(item);
          if (window.history && typeof window.history.replaceState === 'function') {
            window.history.replaceState(null, '', targetUrl.pathname + targetUrl.search + targetUrl.hash);
          }
          window.RNDetailCiteJump(params.citeText, params.citeHeading);
        }
      } catch (_err) {
        // fallback to normal link navigation
      }
    });
  }

  function appendPlainWithCitations(parent, text, citationsByIndex) {
    var parts = String(text || '').split(/(\[\d+\])/g);
    for (var i = 0; i < parts.length; i += 1) {
      var part = parts[i];
      if (!part) continue;
      var match = part.match(/^\[(\d+)\]$/);
      if (match) {
        var idx = Number.parseInt(match[1], 10);
        var citation = citationsByIndex.get(idx);
        var href = citation ? buildCitationHref(citation) : '';
        if (citation && href) {
          var pill = document.createElement('a');
          pill.className = 'kb-chat-cite-pill';
          pill.href = href;
          pill.textContent = part;
          pill.title = citation.title || citation.id;
          bindSamePageCitationJump(pill, href, citation);
          parent.appendChild(pill);
          continue;
        }
      }
      parent.appendChild(document.createTextNode(part));
    }
  }

  function appendInlineMarkdown(parent, rawText, citationsByIndex) {
    var tokens = [];
    var prefix = '@@KBCHATTOK_';
    function stash(nodeFactory) {
      var key = prefix + tokens.length + '@@';
      tokens.push({ key: key, build: nodeFactory });
      return key;
    }

    var text = String(rawText || '');
    // 1. Protect math formulas ($$...$$, \[...\], \(...\), and non-currency $...$)
    text = text
      .replace(/\$\$[\s\S]+?\$\$/g, function (match) {
        return stash(function () {
          return document.createTextNode(match);
        });
      })
      .replace(/\\\[[\s\S]+?\\\]/g, function (match) {
        return stash(function () {
          return document.createTextNode(match);
        });
      })
      .replace(/\\\([\s\S]+?\\\)/g, function (match) {
        return stash(function () {
          return document.createTextNode(match);
        });
      })
      .replace(/\$\s*([^$\n]+?)\s*\$/g, function (match, expr) {
        var trimmed = String(expr || '').trim();
        if (!trimmed || isCurrencyDollarExpr(trimmed)) return match;
        var normalized = '\\(' + trimmed.replace(/\\\*/g, '*') + '\\)';
        return stash(function () {
          return document.createTextNode(normalized);
        });
      });

    // 2. Protect inline code
    text = text.replace(/`([^`\n]+)`/g, function (_m, code) {
      return stash(function () {
        var el = document.createElement('code');
        el.className = 'kb-chat-md-code';
        el.textContent = code;
        return el;
      });
    });

    // 3. Protect Markdown links [label](url): allow internal detail/roadmap links only
    text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_m, label, url) {
      var safeHref = safeInternalNodeUrl(url, '');
      return stash(function () {
        if (safeHref) {
          var a = document.createElement('a');
          a.className = 'kb-chat-md-link';
          a.href = safeHref;
          a.textContent = label;
          return a;
        }
        return document.createTextNode(label);
      });
    });

    // 4. Parse bold (**...** / __...__) and italic (*...*)
    text = text
      .replace(/\*\*([^*\n]+)\*\*/g, function (_m, inner) {
        return stash(function () {
          var strong = document.createElement('strong');
          appendInlineMarkdown(strong, inner, citationsByIndex);
          return strong;
        });
      })
      .replace(/__([^_\n]+)__/g, function (_m, inner) {
        return stash(function () {
          var strong = document.createElement('strong');
          appendInlineMarkdown(strong, inner, citationsByIndex);
          return strong;
        });
      })
      .replace(/(^|[\s(（【「])\*([^*\n]+)\*(?=[\s.,!?;:，。！？；：)）】」]|$)/g, function (_m, lead, inner) {
        return (
          lead +
          stash(function () {
            var em = document.createElement('em');
            appendInlineMarkdown(em, inner, citationsByIndex);
            return em;
          })
        );
      });

    var tokenRe = /(@@KBCHATTOK_\d+@@)/g;
    var segments = text.split(tokenRe);
    for (var i = 0; i < segments.length; i += 1) {
      var seg = segments[i];
      if (!seg) continue;
      var tokMatch = seg.match(/^@@KBCHATTOK_(\d+)@@$/);
      if (tokMatch) {
        var tok = tokens[Number.parseInt(tokMatch[1], 10)];
        if (tok) parent.appendChild(tok.build());
      } else {
        appendPlainWithCitations(parent, seg, citationsByIndex);
      }
    }
  }

  function isTableDividerLine(line) {
    return /^\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)?\|?$/.test(line.trim());
  }

  function splitTableCells(line) {
    var trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '');
    return trimmed.split('|').map(function (c) {
      return c.trim();
    });
  }

  function appendCitedText(container, text, citationsByIndex) {
    var body = document.createElement('div');
    body.className = 'kb-chat-md-body';
    var lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];
      var trimmed = line.trim();

      if (!trimmed) {
        i += 1;
        continue;
      }

      // Fenced code block (or Mermaid diagram)
      if (/^```/.test(trimmed)) {
        var fenceInfo = trimmed.slice(3).trim();
        var codeLines = [];
        i += 1;
        while (i < lines.length && !/^```/.test(lines[i].trim())) {
          codeLines.push(lines[i]);
          i += 1;
        }
        if (i < lines.length && /^```/.test(lines[i].trim())) i += 1;
        var mermaidSource = extractMermaidBlockSource(fenceInfo, codeLines);
        if (mermaidSource !== null) {
          var mermaidHost = document.createElement('div');
          mermaidHost.className = 'mermaid';
          mermaidHost.textContent = mermaidSource;
          body.appendChild(mermaidHost);
          continue;
        }
        var pre = document.createElement('pre');
        pre.className = 'kb-chat-md-pre';
        var codeEl = document.createElement('code');
        codeEl.textContent = codeLines.join('\n');
        pre.appendChild(codeEl);
        body.appendChild(pre);
        continue;
      }

      // Horizontal rule
      if (/^(?:---|\*\*\*|___)\s*$/.test(trimmed)) {
        var hr = document.createElement('hr');
        hr.className = 'kb-chat-md-hr';
        body.appendChild(hr);
        i += 1;
        continue;
      }

      // Heading (# .. ####)
      var hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (hMatch) {
        var level = hMatch[1].length;
        var hEl = document.createElement('div');
        hEl.className = 'kb-chat-md-h kb-chat-md-h' + level;
        appendInlineMarkdown(hEl, hMatch[2], citationsByIndex);
        body.appendChild(hEl);
        i += 1;
        continue;
      }

      // Blockquote (> ...)
      if (/^>\s?/.test(trimmed)) {
        var quoteLines = [];
        while (i < lines.length && /^>\s?/.test(lines[i].trim())) {
          quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
          i += 1;
        }
        var bq = document.createElement('blockquote');
        bq.className = 'kb-chat-md-quote';
        appendInlineMarkdown(bq, quoteLines.join('\n'), citationsByIndex);
        body.appendChild(bq);
        continue;
      }

      // Markdown table
      if (
        trimmed.indexOf('|') >= 0 &&
        i + 1 < lines.length &&
        isTableDividerLine(lines[i + 1])
      ) {
        var headers = splitTableCells(trimmed);
        i += 2;
        var rows = [];
        while (i < lines.length && lines[i].trim().indexOf('|') >= 0 && lines[i].trim() !== '') {
          rows.push(splitTableCells(lines[i]));
          i += 1;
        }
        var tableWrap = document.createElement('div');
        tableWrap.className = 'kb-chat-md-table-wrap';
        var table = document.createElement('table');
        table.className = 'kb-chat-md-table';
        var thead = document.createElement('thead');
        var headTr = document.createElement('tr');
        for (var h = 0; h < headers.length; h += 1) {
          var th = document.createElement('th');
          appendInlineMarkdown(th, headers[h], citationsByIndex);
          headTr.appendChild(th);
        }
        thead.appendChild(headTr);
        table.appendChild(thead);

        var tbody = document.createElement('tbody');
        for (var r = 0; r < rows.length; r += 1) {
          var tr = document.createElement('tr');
          for (var c = 0; c < headers.length; c += 1) {
            var td = document.createElement('td');
            appendInlineMarkdown(td, rows[r][c] || '', citationsByIndex);
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        tableWrap.appendChild(table);
        body.appendChild(tableWrap);
        continue;
      }

      // Unordered list (- / * / +)
      if (/^[-*+]\s+/.test(trimmed)) {
        var ul = document.createElement('ul');
        ul.className = 'kb-chat-md-list';
        while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
          var li = document.createElement('li');
          appendInlineMarkdown(li, lines[i].trim().replace(/^[-*+]\s+/, ''), citationsByIndex);
          ul.appendChild(li);
          i += 1;
        }
        body.appendChild(ul);
        continue;
      }

      // Ordered list (1. )
      if (/^\d+\.\s+/.test(trimmed)) {
        var ol = document.createElement('ol');
        ol.className = 'kb-chat-md-list';
        while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
          var oli = document.createElement('li');
          appendInlineMarkdown(oli, lines[i].trim().replace(/^\d+\.\s+/, ''), citationsByIndex);
          ol.appendChild(oli);
          i += 1;
        }
        body.appendChild(ol);
        continue;
      }

      // Normal paragraph (consume consecutive plain lines)
      var paraLines = [];
      while (i < lines.length) {
        var cur = lines[i];
        var curTrim = cur.trim();
        if (
          !curTrim ||
          /^```/.test(curTrim) ||
          /^(?:---|\*\*\*|___)\s*$/.test(curTrim) ||
          /^(#{1,4})\s+/.test(curTrim) ||
          /^>\s?/.test(curTrim) ||
          /^[-*+]\s+/.test(curTrim) ||
          /^\d+\.\s+/.test(curTrim) ||
          (curTrim.indexOf('|') >= 0 && i + 1 < lines.length && isTableDividerLine(lines[i + 1]))
        ) {
          break;
        }
        paraLines.push(cur);
        i += 1;
      }
      var p = document.createElement('p');
      p.className = 'kb-chat-msg-text';
      appendInlineMarkdown(p, paraLines.join('\n'), citationsByIndex);
      body.appendChild(p);
    }

    container.appendChild(body);
  }

  function formatSourcesSummary(docCount, refCount) {
    if (state.lang === 'en') {
      return (
        t('sourcesLabel') +
        ' · ' +
        docCount +
        (docCount === 1 ? ' doc · ' : ' docs · ') +
        refCount +
        (refCount === 1 ? ' ref' : ' refs')
      );
    }
    return t('sourcesLabel') + ' · ' + docCount + ' 篇文档 ' + refCount + ' 处引用';
  }

  function renderCitationsSection(citations, enrichedCitations) {
    var citeWrap = document.createElement('div');
    citeWrap.className = 'kb-chat-citations';
    citeWrap.setAttribute('data-lang', state.lang);

    var groups = [];
    var groupsByKey = new Map();
    var hasSnippets = false;

    for (var j = 0; j < citations.length; j += 1) {
      var item = citations[j];
      if (!item) continue;
      var meta = resolveCitationMeta(item);
      var savedCite = {
        index: item.index || j + 1,
        id: item.id || '',
        title: meta.title || item.title || item.id || '',
        heading: item.heading || '',
        snippet: item.snippet || '',
        url: item.url || '',
        path: meta.path || '',
        community_label: meta.community || item.community_label || '',
        roadmaps: meta.roadmaps || [],
        updated: meta.updated || '',
        health_score: typeof meta.healthScore === 'number' ? meta.healthScore : undefined,
      };
      enrichedCitations.push(savedCite);
      if (item.snippet && String(item.snippet).trim()) {
        hasSnippets = true;
      }

      var href = buildCitationHref(item);
      var fallbackHref = safeInternalNodeUrl(item.url, item.id);
      var groupKey = item.id ? 'id:' + item.id : fallbackHref || 'idx:' + savedCite.index;
      var group = groupsByKey.get(groupKey);
      if (!group) {
        group = {
          key: groupKey,
          meta: meta,
          firstItem: item,
          entries: [],
        };
        groupsByKey.set(groupKey, group);
        groups.push(group);
      }
      group.entries.push({ item: item, savedCite: savedCite, href: href });
    }

    if (groups.length === 0) return null;

    var headRow = document.createElement('div');
    headRow.className = 'kb-chat-citations__head';

    var label = document.createElement('div');
    label.className = 'kb-chat-citations__label';
    label.textContent = formatSourcesSummary(groups.length, enrichedCitations.length);
    headRow.appendChild(label);

    if (hasSnippets) {
      var toggleBtn = document.createElement('button');
      toggleBtn.type = 'button';
      toggleBtn.className = 'kb-chat-citations__toggle';
      toggleBtn.setAttribute('aria-expanded', 'false');
      toggleBtn.textContent = t('showExcerpts');
      toggleBtn.addEventListener('click', function () {
        var nextOpen = citeWrap.classList.toggle('is-snippets-open');
        toggleBtn.setAttribute('aria-expanded', nextOpen ? 'true' : 'false');
        toggleBtn.textContent = nextOpen ? t('hideExcerpts') : t('showExcerpts');
      });
      headRow.appendChild(toggleBtn);
    }
    citeWrap.appendChild(headRow);

    var listEl = document.createElement('div');
    listEl.className = 'kb-chat-citation-list';
    var pendingCommunitySlots = [];

    for (var g = 0; g < groups.length; g += 1) {
      var grp = groups[g];
      var grpMeta = grp.meta;
      var groupEl = document.createElement('div');
      groupEl.className = 'kb-chat-citation-group';

      var groupHead = document.createElement('div');
      groupHead.className = 'kb-chat-citation-group__head';

      var firstHref = grp.entries[0] ? grp.entries[0].href : '';
      var titleEl = document.createElement(firstHref ? 'a' : 'span');
      titleEl.className = 'kb-chat-citation-title';
      titleEl.textContent = grpMeta.title || String(grp.firstItem.id || '');
      if (firstHref) {
        titleEl.href = firstHref;
        bindSamePageCitationJump(titleEl, firstHref, grp.firstItem);
      }
      groupHead.appendChild(titleEl);

      var metaRow = document.createElement('div');
      metaRow.className = 'kb-chat-citation-meta';
      var hasPendingForGroup = false;

      if (grpMeta.community) {
        var commChip = document.createElement('span');
        commChip.className = 'kb-chat-citation-meta-item kb-chat-citation-community';
        commChip.textContent = grpMeta.community;
        commChip.title = t('metaCommunity') + ' · ' + grpMeta.community;
        metaRow.appendChild(commChip);
      } else if (grpMeta.path) {
        hasPendingForGroup = true;
        var savedCites = grp.entries.map(function (entry) {
          return entry.savedCite;
        });
        pendingCommunitySlots.push({ metaRow: metaRow, path: grpMeta.path, savedCites: savedCites });
      }

      if (grpMeta.roadmaps && grpMeta.roadmaps.length > 0) {
        var rmChip = document.createElement('span');
        rmChip.className = 'kb-chat-citation-meta-item kb-chat-citation-roadmap';
        var rmText = grpMeta.roadmaps[0] + (grpMeta.roadmaps.length > 1 ? ' +' + (grpMeta.roadmaps.length - 1) : '');
        rmChip.textContent = rmText;
        rmChip.title = t('metaRoadmap') + ' · ' + grpMeta.roadmaps.join(' / ');
        metaRow.appendChild(rmChip);
      }

      if (grpMeta.updated) {
        var updChip = document.createElement('span');
        updChip.className = 'kb-chat-citation-meta-item kb-chat-citation-updated';
        updChip.textContent = grpMeta.updated;
        updChip.title = t('metaUpdated') + ' ' + grpMeta.updated;
        metaRow.appendChild(updChip);
      }

      if (typeof grpMeta.healthScore === 'number') {
        metaRow.appendChild(createHealthBarEl(grpMeta.healthScore));
      }

      if (metaRow.childNodes.length > 0 || hasPendingForGroup) {
        groupHead.appendChild(metaRow);
      }
      groupEl.appendChild(groupHead);

      var passagesWrap = document.createElement('div');
      passagesWrap.className = 'kb-chat-citation-passages';
      for (var pIdx = 0; pIdx < grp.entries.length; pIdx += 1) {
        var entry = grp.entries[pIdx];
        var passageEl = document.createElement(entry.href ? 'a' : 'div');
        passageEl.className = 'kb-chat-citation-passage';
        if (entry.href) {
          passageEl.href = entry.href;
          bindSamePageCitationJump(passageEl, entry.href, entry.item);
        }

        var passageHead = document.createElement('span');
        passageHead.className = 'kb-chat-citation-passage-head';

        var idxSpan = document.createElement('span');
        idxSpan.className = 'kb-chat-citation-idx';
        idxSpan.textContent = '[' + entry.savedCite.index + ']';
        passageHead.appendChild(idxSpan);

        var headingText =
          entry.item.heading && String(entry.item.heading).trim()
            ? String(entry.item.heading).trim()
            : t('defaultPassageLabel');
        var headingSpan = document.createElement('span');
        headingSpan.className = 'kb-chat-citation-heading';
        headingSpan.textContent = headingText;
        passageHead.appendChild(headingSpan);
        passageEl.appendChild(passageHead);

        if (entry.item.snippet) {
          var snippetEl = document.createElement('span');
          snippetEl.className = 'kb-chat-citation-snippet';
          snippetEl.textContent = normalizeMathText(String(entry.item.snippet));
          passageEl.appendChild(snippetEl);
        }
        passagesWrap.appendChild(passageEl);
      }
      groupEl.appendChild(passagesWrap);
      listEl.appendChild(groupEl);
    }

    citeWrap.appendChild(listEl);

    if (pendingCommunitySlots.length > 0) {
      ensureCommunityIndex().then(function (commIdx) {
        if (!commIdx || !commIdx.pathToCommunity) return;
        var updatedAny = false;
        for (var p = 0; p < pendingCommunitySlots.length; p += 1) {
          var slotInfo = pendingCommunitySlots[p];
          var cid = commIdx.pathToCommunity.get(slotInfo.path);
          var rawLabel = cid && commIdx.communityLabel ? commIdx.communityLabel[cid] : '';
          var shortLabel = shortenCommunityLabel(rawLabel);
          if (!shortLabel) continue;
          for (var sc = 0; sc < slotInfo.savedCites.length; sc += 1) {
            slotInfo.savedCites[sc].community_label = shortLabel;
          }
          updatedAny = true;
          var commEl = document.createElement('span');
          commEl.className = 'kb-chat-citation-meta-item kb-chat-citation-community';
          commEl.textContent = shortLabel;
          commEl.title = t('metaCommunity') + ' · ' + shortLabel;
          slotInfo.metaRow.insertBefore(commEl, slotInfo.metaRow.firstChild);
        }
        if (updatedAny) {
          saveChatHistory(state.history);
        }
      });
    }

    return citeWrap;
  }

  function appendAssistantMessage(options) {
    var msg = document.createElement('div');
    msg.className = 'kb-chat-msg kb-chat-msg--assistant';

    if (options.statusNote) {
      var note = document.createElement('div');
      note.className = 'kb-chat-status-note';
      note.textContent = options.statusNote;
      msg.appendChild(note);
    }

    var citations = Array.isArray(options.citations) ? options.citations : [];
    var citationsByIndex = new Map();
    var enrichedCitations = [];
    for (var i = 0; i < citations.length; i += 1) {
      var c = citations[i];
      if (c && typeof c.index === 'number') {
        citationsByIndex.set(c.index, c);
      }
    }

    appendCitedText(msg, options.text || '', citationsByIndex);

    if (citations.length > 0) {
      var citeWrap = renderCitationsSection(citations, enrichedCitations);
      if (citeWrap) {
        msg.appendChild(citeWrap);
      }
    } else if (options.showGraphLink) {
      var graphLink = document.createElement('a');
      graphLink.className = 'kb-chat-cite-pill';
      graphLink.href = 'graph.html';
      graphLink.textContent = t('fallbackLinksText');
      msg.appendChild(graphLink);
    }

    state.messagesEl.appendChild(msg);
    renderMessageMath(msg);
    renderMessageMermaid(msg);
    state.messagesEl.scrollTop = state.messagesEl.scrollHeight;
    if (!options._restoring) {
      recordHistoryEntry({
        role: 'assistant',
        statusNote: options.statusNote || '',
        text: String(options.text || ''),
        citations: enrichedCitations,
        showGraphLink: Boolean(options.showGraphLink),
      });
    }
  }

  var activeTurnstileHandlers = null;

  function makeTurnstileError(code, message, detail) {
    var err = new Error(message || code);
    err.code = code;
    if (detail) err.detail = String(detail);
    return err;
  }

  function runTurnstileChallenge(siteKey) {
    if (state.turnstileInflight) {
      return state.turnstileInflight;
    }
    var challengePromise = ensureTurnstileReady()
      .then(function (turnstile) {
        return new Promise(function (resolve, reject) {
          if (!state.turnstileEl) {
            reject(makeTurnstileError('TURNSTILE_MOUNT_MISSING', 'Turnstile mount missing'));
            return;
          }
          state.turnstileEl.hidden = false;
          var settled = false;
          var timer = setTimeout(function () {
            if (settled) return;
            settled = true;
            activeTurnstileHandlers = null;
            if (state.turnstileEl) state.turnstileEl.hidden = true;
            reject(makeTurnstileError('TURNSTILE_TIMEOUT', 'Turnstile timeout'));
          }, 25000);

          activeTurnstileHandlers = {
            onSuccess: function (token) {
              if (settled) return;
              settled = true;
              clearTimeout(timer);
              activeTurnstileHandlers = null;
              if (state.turnstileEl) state.turnstileEl.hidden = true;
              resolve(token);
            },
            onError: function (code, errMsg, detail) {
              if (settled) return;
              settled = true;
              clearTimeout(timer);
              activeTurnstileHandlers = null;
              if (state.turnstileEl) state.turnstileEl.hidden = true;
              reject(makeTurnstileError(code || 'TURNSTILE_WIDGET_ERROR', errMsg || 'Turnstile failed', detail));
            },
          };

          try {
            var isNewWidget = state.turnstileWidgetId === null;
            if (isNewWidget) {
              state.turnstileWidgetId = turnstile.render(state.turnstileEl, {
                sitekey: siteKey,
                appearance: 'interaction-only',
                execution: 'execute',
                callback: function (token) {
                  if (activeTurnstileHandlers) activeTurnstileHandlers.onSuccess(token);
                },
                'error-callback': function (errCode) {
                  if (activeTurnstileHandlers) {
                    activeTurnstileHandlers.onError(
                      'TURNSTILE_WIDGET_ERROR',
                      'Turnstile widget error',
                      errCode,
                    );
                  }
                },
                'expired-callback': function () {
                  state.prewarmedToken = null;
                  if (activeTurnstileHandlers) {
                    activeTurnstileHandlers.onError('TURNSTILE_EXPIRED', 'Turnstile expired');
                  }
                },
              });
            } else {
              turnstile.reset(state.turnstileWidgetId);
            }
            setTimeout(function () {
              if (settled) return;
              try {
                turnstile.execute(state.turnstileWidgetId);
              } catch (execErr) {
                if (activeTurnstileHandlers) {
                  activeTurnstileHandlers.onError(
                    'TURNSTILE_EXEC_ERROR',
                    execErr && execErr.message ? execErr.message : 'Turnstile execution failed',
                  );
                }
              }
            }, isNewWidget ? 40 : 0);
          } catch (err) {
            if (activeTurnstileHandlers) {
              activeTurnstileHandlers.onError(
                'TURNSTILE_EXEC_ERROR',
                err && err.message ? err.message : 'Turnstile execution failed',
              );
            }
          }
        });
      })
      .finally(function () {
        if (state.turnstileInflight === challengePromise) {
          state.turnstileInflight = null;
        }
      });

    state.turnstileInflight = challengePromise;
    return challengePromise;
  }

  function warmUpTurnstile(siteKey) {
    if (!siteKey) return;
    if (state.prewarmedToken && state.prewarmedToken.expiresAt > Date.now()) return;
    if (state.turnstileInflight) return;
    runTurnstileChallenge(siteKey).then(
      function (token) {
        if (token) {
          state.prewarmedToken = {
            value: token,
            expiresAt: Date.now() + 240000,
          };
        }
      },
      function () {
        state.prewarmedToken = null;
      },
    );
  }

  function acquireTurnstileToken(siteKey) {
    if (state.prewarmedToken && state.prewarmedToken.expiresAt > Date.now()) {
      var cachedToken = state.prewarmedToken.value;
      state.prewarmedToken = null;
      return Promise.resolve(cachedToken);
    }
    state.prewarmedToken = null;
    return runTurnstileChallenge(siteKey).then(function (token) {
      state.prewarmedToken = null;
      return token;
    });
  }

  function runCloudRequest(cfg, question, contentHashes) {
    return acquireTurnstileToken(cfg.turnstileSiteKey).then(
      function (turnstileToken) {
        var controller = new AbortController();
        var timer = setTimeout(function () {
          controller.abort();
        }, cfg.requestTimeoutMs);
        return fetch(cfg.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question,
            lang: state.lang,
            turnstileToken: turnstileToken,
            contentHashes: contentHashes,
          }),
          signal: controller.signal,
        })
          .then(function (resp) {
            return resp.json().then(function (data) {
              return { status: resp.status, ok: resp.ok, data: data };
            });
          })
          .finally(function () {
            clearTimeout(timer);
          });
      },
      function (err) {
        var code = err && err.code ? String(err.code) : 'TURNSTILE_FAILED';
        var detail = err && err.detail ? String(err.detail) : '';
        if (typeof console !== 'undefined' && typeof console.warn === 'function') {
          console.warn('[kb-chat] turnstile fallback:', code, detail || (err && err.message) || '');
        }
        return { status: 403, ok: false, data: { code: code, detail: detail } };
      },
    );
  }

  function renderLocalParagraphAnswer(question, candidateDocs, catalog, statusNote) {
    var detailPages = (catalog && catalog.pages && catalog.pages.detail_pages) || {};
    var docMetaById = new Map();
    var targetUrls = [];
    for (var i = 0; i < candidateDocs.length && targetUrls.length < 4; i += 1) {
      var doc = candidateDocs[i];
      docMetaById.set(doc.id, doc);
      var meta = Object.prototype.hasOwnProperty.call(detailPages, doc.id) ? detailPages[doc.id] : null;
      if (meta && meta.content_url && CONTENT_URL_RE.test(meta.content_url)) {
        targetUrls.push(meta.content_url);
      }
    }

    return Promise.all(targetUrls.map(fetchPageContent)).then(function (bodies) {
      var citations = selectLocalExcerpts(bodies.filter(Boolean), docMetaById, question);
      if (citations.length === 0) {
        appendAssistantMessage({
          statusNote: statusNote,
          text: t('ungrounded'),
          citations: [],
          showGraphLink: true,
        });
        return;
      }

      var lines = [t('localAnswerIntro')];
      for (var c = 0; c < citations.length; c += 1) {
        var cite = citations[c];
        lines.push('[' + cite.index + '] ' + cite.title + '（' + cite.heading + '）：' + cite.snippet);
      }
      appendAssistantMessage({
        statusNote: statusNote,
        text: lines.join('\n\n'),
        citations: citations,
      });
    });
  }

  function classifyFallbackNote(result) {
    if (!result) return t('fallbackGeneric');
    var code = result.data && result.data.code ? String(result.data.code) : '';
    var detail = result.data && result.data.detail ? String(result.data.detail) : '';
    if (typeof console !== 'undefined' && typeof console.warn === 'function') {
      console.warn('[kb-chat] fallback response:', result.status, code, detail);
    }
    if (result.status === 429 || code === 'RATE_LIMITED' || code === 'CONCURRENT_LIMIT') {
      return t('fallbackRateLimited');
    }
    if (code === 'QUOTA_OR_AI_UNAVAILABLE' || result.status === 503) {
      return t('fallbackQuota');
    }
    if (code.indexOf('TURNSTILE') === 0 || result.status === 403 || code === 'ORIGIN_NOT_ALLOWED') {
      var tag = code || 'HTTP_403';
      if (detail) tag += ':' + detail;
      return t('fallbackVerify') + ' [' + tag + ']';
    }
    return t('fallbackGeneric') + (code ? ' [' + code + ']' : '');
  }

  function setBusy(nextBusy) {
    state.busy = nextBusy;
    if (state.sendBtn) {
      state.sendBtn.disabled = nextBusy;
      state.sendBtn.textContent = nextBusy ? t('sending') : t('send');
    }
    if (state.inputEl) {
      state.inputEl.disabled = nextBusy;
    }
  }

  var GREETING_RE = /^(?:在吗|你好|您好|哈喽|嗨|有人吗|你是谁|介绍一下你|hi|hello|hey|yo|good\s+(?:morning|afternoon|evening))[!！?？。.,，\s]*$/i;

  function submitQuestion(rawText) {
    ensureMounted();
    var cfg = getConfig();
    var question = String(rawText != null ? rawText : state.inputEl.value).trim().replace(/\s+/g, ' ');
    if (question.length < 2) {
      appendAssistantMessage({ text: t('emptyError'), citations: [] });
      return Promise.resolve();
    }
    if (question.length > cfg.maxQuestionLength) {
      appendAssistantMessage({
        text: t('tooLongError', { max: cfg.maxQuestionLength }),
        citations: [],
      });
      return Promise.resolve();
    }

    appendUserMessage(question);
    if (state.inputEl) {
      state.inputEl.value = '';
      updateCounter();
    }
    if (GREETING_RE.test(question)) {
      appendAssistantMessage({ text: t('greetingReply'), citations: [] });
      return Promise.resolve();
    }
    setBusy(true);

    return Promise.all([ensureLiteIndex(), ensureCatalog()])
      .then(function (results) {
        var liteIndex = results[0];
        var catalog = results[1];
        var currentId = currentPageContextId(catalog);
        var candidateDocs = rankCandidateDocs(liteIndex.docs, question, currentId);

        if (candidateDocs.length === 0) {
          appendAssistantMessage({
            text: t('ungrounded'),
            citations: [],
            showGraphLink: true,
          });
          return null;
        }

        var detailPages = (catalog && catalog.pages && catalog.pages.detail_pages) || {};
        var contentHashes = [];
        var seenHashes = new Set();
        for (var i = 0; i < candidateDocs.length && contentHashes.length < 4; i += 1) {
          var meta = Object.prototype.hasOwnProperty.call(detailPages, candidateDocs[i].id)
            ? detailPages[candidateDocs[i].id]
            : null;
          var match = meta && meta.content_url ? CONTENT_URL_RE.exec(meta.content_url) : null;
          if (match && !seenHashes.has(match[1])) {
            seenHashes.add(match[1]);
            contentHashes.push(match[1]);
          }
        }

        if (!isCloudConfigured(cfg) || contentHashes.length === 0) {
          state.mode = 'local';
          updatePanelCopy();
          return renderLocalParagraphAnswer(question, candidateDocs, catalog, '');
        }

        return runCloudRequest(cfg, question, contentHashes).then(function (res) {
          if (res && res.ok && res.data && res.data.ok && typeof res.data.answer === 'string') {
            state.mode = 'ai';
            updatePanelCopy();
            appendAssistantMessage({
              statusNote: res.data.cached ? t('cachedTag') : '',
              text: res.data.answer,
              citations: Array.isArray(res.data.citations) ? res.data.citations : [],
              showGraphLink: res.data.grounded === false,
            });
            return null;
          }
          state.mode = 'fallback';
          updatePanelCopy();
          return renderLocalParagraphAnswer(question, candidateDocs, catalog, classifyFallbackNote(res));
        });
      })
      .catch(function () {
        appendAssistantMessage({
          text: t('fallbackGeneric'),
          citations: [],
          showGraphLink: true,
        });
      })
      .finally(function () {
        setBusy(false);
        if (state.open && state.inputEl) {
          state.inputEl.focus();
        }
      });
  }

  function ensureMounted() {
    if (state.panel) return state.panel;
    var slot = document.getElementById('kbChatSlot');
    if (!slot) {
      slot = document.body;
    }

    var cfg = getConfig();
    state.mode = isCloudConfigured(cfg) ? 'ai' : 'local';

    var panel = document.createElement('section');
    panel.id = 'kbChatPanel';
    panel.className = 'kb-chat-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'false');
    panel.hidden = true;

    initPanelResize(panel);

    var header = document.createElement('div');
    header.className = 'kb-chat-header';

    var titleWrap = document.createElement('div');
    titleWrap.className = 'kb-chat-header__title-wrap';
    var titleEl = document.createElement('span');
    titleEl.className = 'kb-chat-header__title';
    var badgeEl = document.createElement('span');
    badgeEl.className = 'kb-chat-mode-badge';
    titleWrap.appendChild(titleEl);
    titleWrap.appendChild(badgeEl);

    var actions = document.createElement('div');
    actions.className = 'kb-chat-header__actions';
    var clearBtn = document.createElement('button');
    clearBtn.id = 'kbChatClear';
    clearBtn.type = 'button';
    clearBtn.className = 'kb-chat-icon-btn kb-chat-clear-btn';
    clearBtn.addEventListener('click', clearLocalHistory);
    var langBtn = document.createElement('button');
    langBtn.type = 'button';
    langBtn.className = 'kb-chat-icon-btn';
    langBtn.addEventListener('click', function () {
      setLanguage(state.lang === 'zh' ? 'en' : 'zh');
    });
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'kb-chat-icon-btn';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', close);
    actions.appendChild(clearBtn);
    actions.appendChild(langBtn);
    actions.appendChild(closeBtn);

    header.appendChild(titleWrap);
    header.appendChild(actions);

    var messagesEl = document.createElement('div');
    messagesEl.className = 'kb-chat-messages';
    messagesEl.setAttribute('aria-live', 'polite');

    var welcomeEl = document.createElement('div');
    welcomeEl.className = 'kb-chat-welcome';
    var startersEl = document.createElement('div');
    startersEl.className = 'kb-chat-starters';
    messagesEl.appendChild(welcomeEl);
    messagesEl.appendChild(startersEl);

    var turnstileEl = document.createElement('div');
    turnstileEl.id = 'kbChatTurnstile';
    turnstileEl.className = 'kb-chat-turnstile';
    turnstileEl.hidden = true;

    var form = document.createElement('form');
    form.className = 'kb-chat-form';
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      if (!state.busy) submitQuestion();
    });

    var inputRow = document.createElement('div');
    inputRow.className = 'kb-chat-input-row';
    var inputEl = document.createElement('textarea');
    inputEl.id = 'kbChatInput';
    inputEl.className = 'kb-chat-input';
    inputEl.rows = 1;
    inputEl.maxLength = cfg.maxQuestionLength;
    inputEl.addEventListener('input', updateCounter);
    inputEl.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!state.busy) submitQuestion();
      }
    });

    var sendBtn = document.createElement('button');
    sendBtn.type = 'submit';
    sendBtn.className = 'kb-chat-send';
    inputRow.appendChild(inputEl);
    inputRow.appendChild(sendBtn);

    var formFoot = document.createElement('div');
    formFoot.className = 'kb-chat-form-foot';
    var noticeEl = document.createElement('span');
    var counterEl = document.createElement('span');
    formFoot.appendChild(noticeEl);
    formFoot.appendChild(counterEl);

    form.appendChild(inputRow);
    form.appendChild(formFoot);

    panel.appendChild(header);
    panel.appendChild(messagesEl);
    panel.appendChild(turnstileEl);
    panel.appendChild(form);
    slot.appendChild(panel);

    state.panel = panel;
    state.titleEl = titleEl;
    state.badgeEl = badgeEl;
    state.clearBtn = clearBtn;
    state.langBtn = langBtn;
    state.closeBtn = closeBtn;
    state.messagesEl = messagesEl;
    state.welcomeEl = welcomeEl;
    state.startersEl = startersEl;
    state.turnstileEl = turnstileEl;
    state.inputEl = inputEl;
    state.sendBtn = sendBtn;
    state.noticeEl = noticeEl;
    state.counterEl = counterEl;

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && state.open) {
        if (_fallbackLightboxEl && !_fallbackLightboxEl.overlay.hidden) {
          _fallbackLightboxEl.overlay.hidden = true;
          return;
        }
        if (document.querySelector('.mermaid-lightbox:not([hidden])')) {
          return;
        }
        close();
      }
    });

    if (!state.historyRestored) {
      state.historyRestored = true;
      var savedEntries = loadChatHistory();
      if (savedEntries.length > 0) {
        state.history = savedEntries.slice();
        for (var hi = 0; hi < savedEntries.length; hi += 1) {
          var entry = savedEntries[hi];
          if (entry.role === 'user') {
            appendUserMessage(entry.text, true);
          } else {
            appendAssistantMessage({
              statusNote: entry.statusNote || '',
              text: entry.text || '',
              citations: Array.isArray(entry.citations) ? entry.citations : [],
              showGraphLink: Boolean(entry.showGraphLink),
              _restoring: true,
            });
          }
        }
        syncClearButton();
      }
    }

    if (typeof MutationObserver === 'function' && document.documentElement) {
      new MutationObserver(function () {
        var docLang = String(document.documentElement.getAttribute('lang') || '').toLowerCase();
        var nextLang = docLang.indexOf('en') === 0 ? 'en' : 'zh';
        if (nextLang !== state.lang) {
          setLanguage(nextLang);
        }
      }).observe(document.documentElement, { attributes: true, attributeFilter: ['lang'] });
    }

    updatePanelCopy();
    return panel;
  }

  function notifyHostState() {
    if (typeof document === 'undefined' || typeof document.dispatchEvent !== 'function') return;
    if (typeof CustomEvent === 'function') {
      document.dispatchEvent(new CustomEvent('rn-kb-chat:state', { detail: { open: state.open } }));
    }
  }

  function open() {
    var panel = ensureMounted();
    state.open = true;
    panel.hidden = false;
    syncToggleButtonLabels();
    notifyHostState();
    var cfg = getConfig();
    if (cfg.endpoint && cfg.turnstileSiteKey) {
      warmUpTurnstile(cfg.turnstileSiteKey);
    }
    if (state.inputEl) state.inputEl.focus();
  }

  function close() {
    if (!state.panel) return;
    var hadPanelFocus = Boolean(
      typeof document !== 'undefined' && state.panel.contains(document.activeElement)
    );
    state.open = false;
    state.panel.hidden = true;
    syncToggleButtonLabels();
    notifyHostState();
    if (hadPanelFocus || typeof document !== 'undefined') {
      var mainBtn = document.getElementById('fabMainToggle');
      var toggle = document.getElementById('kbChatToggle');
      var focusTarget = mainBtn && !mainBtn.hidden ? mainBtn : toggle;
      if (focusTarget && !focusTarget.hidden && (hadPanelFocus || !mainBtn || document.activeElement === document.body)) {
        focusTarget.focus({ preventScroll: true });
      }
    }
  }

  function toggle() {
    if (state.open) close();
    else open();
  }

  function setLanguage(lang) {
    state.lang = lang === 'en' ? 'en' : 'zh';
    updatePanelCopy();
  }

  function isOpen() {
    return state.open;
  }

  var api = {
    open: open,
    close: close,
    toggle: toggle,
    ask: submitQuestion,
    setLanguage: setLanguage,
    isOpen: isOpen,
    _internals: {
      tokenizeQuery: tokenizeQuery,
      rankCandidateDocs: rankCandidateDocs,
      selectLocalExcerpts: selectLocalExcerpts,
      safeInternalNodeUrl: safeInternalNodeUrl,
      sanitizeEndpoint: sanitizeEndpoint,
      splitSections: splitSections,
    },
  };

  if (typeof window !== 'undefined') {
    window.RNKbChat = api;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})();
