(function () {
  'use strict';

  var DEFAULT_CONFIG = {
    // 部署 Workers Free 网关与 Turnstile 后填写公开配置（严禁在此填写 Secret Key）
    endpoint: 'https://robotics-notebooks-kb-chat.chongliu.workers.dev',
    turnstileSiteKey: '0x4AAAAAAGIWdfoeyhjsNN-h',
    maxQuestionLength: 300,
    requestTimeoutMs: 10000,
  };

  var CONTENT_URL_RE = /^exports\/page-content\/([a-f0-9]{64})\.json$/;
  var SAFE_NODE_URL_RE = /^(?:detail|roadmap)\.html\?id=[A-Za-z0-9%._~!*()-]+$/;
  var WORD_OR_CJK_RE = /[A-Za-z0-9_+\-.\u0370-\u03ff]+|[\u4e00-\u9fff]+/g;
  var CODE_FENCE_RE = /```[\s\S]*?```/g;
  var HTML_TAG_RE = /<[^>]+>/g;
  var RAW_URL_RE = /\b(?:https?:\/\/|javascript:|data:|vbscript:)[^\s)\]<>"']+/gi;
  var TURNSTILE_SCRIPT_URL = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';

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
      readonlyNotice: '基于站内文档回答',
      sourcesLabel: '参考来源',
      ungrounded: '暂未在站内找到直接相关内容，可换个短词、英文术语，或逛逛知识图谱。',
      localAnswerIntro: '找到以下站内相关内容：',
      fallbackRateLimited: '提问太频繁，已为你切为站内检索结果。',
      fallbackQuota: 'AI 助手当前繁忙，已为你切为站内检索结果。',
      fallbackVerify: '安全校验未完成，已为你切为站内检索结果。',
      fallbackGeneric: 'AI 助手暂不可用，已为你切为站内检索结果。',
      cachedTag: '来自常见问答缓存',
      emptyError: '请输入至少 2 个字的问题。',
      tooLongError: '问题请控制在 {max} 字以内。',
      starters: ['什么是全身控制 WBC？', 'Sim2Real 如何缩小仿真到真机差距？', 'VLA 模型与传统控制怎么配合？'],
      fallbackLinksText: '浏览知识图谱',
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
      readonlyNotice: 'Grounded in the wiki',
      sourcesLabel: 'Sources',
      ungrounded: 'No direct match found in the wiki. Try shorter keywords or explore the Knowledge Graph.',
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
    },
  };

  function getConfig() {
    var raw = typeof window !== 'undefined' && window.RN_KB_CHAT_CONFIG ? window.RN_KB_CHAT_CONFIG : {};
    var merged = {
      endpoint: sanitizeEndpoint(raw.endpoint || DEFAULT_CONFIG.endpoint),
      turnstileSiteKey: String(raw.turnstileSiteKey || DEFAULT_CONFIG.turnstileSiteKey || '').trim(),
      maxQuestionLength: Number(raw.maxQuestionLength) || DEFAULT_CONFIG.maxQuestionLength,
      requestTimeoutMs: Number(raw.requestTimeoutMs) || DEFAULT_CONFIG.requestTimeoutMs,
    };
    return merged;
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
    _litePromise = fetchJsonWithTimeout('search-index-lite.json', 20000)
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

  function ensureCatalog() {
    if (typeof window !== 'undefined' && window.RNSiteCatalog && typeof window.RNSiteCatalog.ensure === 'function') {
      return window.RNSiteCatalog.ensure();
    }
    if (_catalogPromise) return _catalogPromise;
    _catalogPromise = fetchJsonWithTimeout('exports/site-catalog-v1.json', 20000).catch(function (err) {
      _catalogPromise = null;
      throw err;
    });
    return _catalogPromise;
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

    var scored = [];
    for (var i = 0; i < docs.length; i += 1) {
      var doc = docs[i];
      if (!doc || !doc.id) continue;
      var title = String(doc.title || '').toLowerCase();
      var summary = String(doc.summary || '').toLowerCase();
      var tags = Array.isArray(doc.tags) ? doc.tags.join(' ').toLowerCase() : '';
      var aliases = Array.isArray(doc.search_aliases) ? doc.search_aliases.join(' ').toLowerCase() : '';
      var path = String(doc.path || '').toLowerCase();

      var score = 0;
      for (var t = 0; t < tokens.length; t += 1) {
        var tok = tokens[t];
        var w = tok.length >= 2 ? (tok.length >= 4 ? 3 : 2) : 0.4;
        if (aliases.indexOf(tok) >= 0) score += w * 4.5;
        if (title.indexOf(tok) >= 0) score += w * 4;
        if (tags.indexOf(tok) >= 0) score += w * 2.5;
        if (summary.indexOf(tok) >= 0) score += w * 2;
        if (path.indexOf(tok) >= 0) score += w * 1.5;
      }
      if (currentId && doc.id === currentId && score > 0) {
        score += 3;
      }
      if (score >= 1.5) {
        scored.push({ doc: doc, score: score });
      }
    }

    scored.sort(function (a, b) {
      return b.score - a.score;
    });
    return scored.slice(0, 6).map(function (entry) {
      return entry.doc;
    });
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
          reject(new Error('Turnstile API missing'));
        }
      };
      script.onerror = function () {
        _turnstileScriptPromise = null;
        reject(new Error('Failed to load Turnstile'));
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
    turnstileEl: null,
    turnstileWidgetId: null,
    inputEl: null,
    sendBtn: null,
    counterEl: null,
    noticeEl: null,
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

  function appendUserMessage(text) {
    var msg = document.createElement('div');
    msg.className = 'kb-chat-msg kb-chat-msg--user';
    msg.textContent = text;
    state.messagesEl.appendChild(msg);
    state.messagesEl.scrollTop = state.messagesEl.scrollHeight;
  }

  function appendCitedText(container, text, citationsByIndex) {
    var p = document.createElement('p');
    p.className = 'kb-chat-msg-text';
    var parts = String(text || '').split(/(\[\d+\])/g);
    for (var i = 0; i < parts.length; i += 1) {
      var part = parts[i];
      if (!part) continue;
      var match = part.match(/^\[(\d+)\]$/);
      if (match) {
        var idx = Number.parseInt(match[1], 10);
        var citation = citationsByIndex.get(idx);
        var href = citation ? safeInternalNodeUrl(citation.url, citation.id) : '';
        if (citation && href) {
          var pill = document.createElement('a');
          pill.className = 'kb-chat-cite-pill';
          pill.href = href;
          pill.textContent = part;
          pill.title = citation.title || citation.id;
          p.appendChild(pill);
          continue;
        }
      }
      p.appendChild(document.createTextNode(part));
    }
    container.appendChild(p);
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
    for (var i = 0; i < citations.length; i += 1) {
      var c = citations[i];
      if (c && typeof c.index === 'number') {
        citationsByIndex.set(c.index, c);
      }
    }

    appendCitedText(msg, options.text || '', citationsByIndex);

    if (citations.length > 0) {
      var citeWrap = document.createElement('div');
      citeWrap.className = 'kb-chat-citations';

      var label = document.createElement('div');
      label.className = 'kb-chat-citations__label';
      label.textContent = t('sourcesLabel');
      citeWrap.appendChild(label);

      for (var j = 0; j < citations.length; j += 1) {
        var item = citations[j];
        if (!item) continue;
        var href = safeInternalNodeUrl(item.url, item.id);
        var card = document.createElement(href ? 'a' : 'div');
        card.className = 'kb-chat-citation-card';
        if (href) card.href = href;

        var titleEl = document.createElement('span');
        titleEl.className = 'kb-chat-citation-title';
        titleEl.textContent = '[' + (item.index || j + 1) + '] ' + String(item.title || item.id || '');
        card.appendChild(titleEl);

        if (item.heading) {
          var headingEl = document.createElement('span');
          headingEl.className = 'kb-chat-citation-heading';
          headingEl.textContent = String(item.heading);
          card.appendChild(headingEl);
        }

        if (item.snippet) {
          var snippetEl = document.createElement('span');
          snippetEl.className = 'kb-chat-citation-snippet';
          snippetEl.textContent = String(item.snippet);
          card.appendChild(snippetEl);
        }
        citeWrap.appendChild(card);
      }
      msg.appendChild(citeWrap);
    } else if (options.showGraphLink) {
      var graphLink = document.createElement('a');
      graphLink.className = 'kb-chat-cite-pill';
      graphLink.href = 'graph.html';
      graphLink.textContent = t('fallbackLinksText');
      msg.appendChild(graphLink);
    }

    state.messagesEl.appendChild(msg);
    state.messagesEl.scrollTop = state.messagesEl.scrollHeight;
  }

  function acquireTurnstileToken(siteKey) {
    return ensureTurnstileReady().then(function (turnstile) {
      return new Promise(function (resolve, reject) {
        if (!state.turnstileEl) {
          reject(new Error('Turnstile mount missing'));
          return;
        }
        state.turnstileEl.hidden = false;
        var settled = false;
        var timer = setTimeout(function () {
          if (settled) return;
          settled = true;
          reject(new Error('Turnstile timeout'));
        }, 12000);

        function finishOk(token) {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          resolve(token);
        }
        function finishErr(errMsg) {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          reject(new Error(errMsg || 'Turnstile failed'));
        }

        try {
          if (state.turnstileWidgetId === null) {
            state.turnstileWidgetId = turnstile.render(state.turnstileEl, {
              sitekey: siteKey,
              appearance: 'interaction-only',
              execution: 'execute',
              callback: finishOk,
              'error-callback': function () {
                finishErr('Turnstile error');
              },
              'expired-callback': function () {
                finishErr('Turnstile expired');
              },
            });
          } else {
            turnstile.reset(state.turnstileWidgetId);
          }
          turnstile.execute(state.turnstileWidgetId);
        } catch (err) {
          finishErr(err && err.message ? err.message : 'Turnstile execution failed');
        }
      });
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
      function () {
        return { status: 403, ok: false, data: { code: 'TURNSTILE_FAILED' } };
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
    if (result.status === 429 || code === 'RATE_LIMITED' || code === 'CONCURRENT_LIMIT') {
      return t('fallbackRateLimited');
    }
    if (code === 'QUOTA_OR_AI_UNAVAILABLE' || result.status === 503) {
      return t('fallbackQuota');
    }
    if (code.indexOf('TURNSTILE') === 0 || result.status === 403) {
      return t('fallbackVerify');
    }
    return t('fallbackGeneric');
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
        close();
      }
    });

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
