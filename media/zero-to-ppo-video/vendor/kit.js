/* Shared toolkit for the interactive paper demos (assets/js/demos/*.js).
 *
 * _layouts/paper.html loads this file first on any note whose front matter
 * declares `demos: [...]`, then the bundles themselves. The bundles may only
 * assume `window.PaperDemoKit` — everything else (Canvas, DOM) is native,
 * because the site's CSP allows no third-party scripts and
 * scripts/sanitize_paper_html.py strips <script>/<canvas>/<input> out of
 * #paper-body, so a note can ship nothing but the placeholder
 *
 *     <div class="paper-demo" data-demo="ppo-clip"></div>
 *
 * and every widget has to be built at runtime. Styling: assets/css/paper-demos.css.
 *
 * Formulas: write LaTeX and let the KaTeX the page already loads typeset it —
 * `$…$` inside any demo string (card title/sub, note(), verdictBox, explainer
 * cues, demo-table cells via el()/table().row()), K.svgMath() for a bare
 * formula on an SVG storyboard and K.svgRich() for a storyboard label that
 * mixes prose with `$…$`. See AGENTS.md, and texToPlain() below for what a
 * reader sees when that CDN is blocked.
 *
 * Usage from a bundle:
 *
 *     (function () {
 *       var K = window.PaperDemoKit;
 *       function buildFoo(host) { var root = K.card(host, {...}); ... }
 *       K.mount({ 'paper-foo': buildFoo });
 *     })();
 */

(function () {
  'use strict';

  // ─── tiny DOM kit ────────────────────────────────────────────────────────
  function needsRichMarkup(str) {
    return typeof str === 'string' && (/\$[^$]+\$/.test(str) || /\*\*/.test(str) || /`[^`]+`/.test(str));
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null && text !== '') {
      if (needsRichMarkup(text)) rich(node, text);
      else node.textContent = text;
    }
    return node;
  }

  function fmt(x, digits) {
    var d = digits == null ? 2 : digits;
    if (!isFinite(x)) return '∞';
    var s = x.toFixed(d);
    if (/^-0(\.0*)?$/.test(s)) s = s.slice(1); // avoid "-0.00"
    return s;
  }

  function clamp(x, lo, hi) {
    return x < lo ? lo : x > hi ? hi : x;
  }

  // ─── LaTeX / KaTeX ───────────────────────────────────────────────────────
  /* Formulas in a demo are written as LaTeX and rendered by the KaTeX that
     every page already loads (_layouts/default.html, pinned + SRI) — the
     demos add no script of their own, so the CSP is untouched.
     `$…$` inside any demo string goes through here (see rich()); a formula
     drawn on an SVG storyboard goes through svgMath().
     If the KaTeX CDN is blocked the formula degrades to readable plain text
     (texToPlain) instead of leaking raw TeX, and re-renders if KaTeX shows
     up late — the script is `defer`red, so it is normally already there. */

  /* Enough of a TeX subset for the fallback to stay readable: the demos only
     ever write single-level formulas. */
  var TEX_FRAC = /\\[tdc]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g;
  var TEX_FRAC_ANY = /\\[tdc]?frac/;

  var TEX_PLAIN = [
    [/_\s*\{([^{}]*)\}/g, '_$1'], [/\^\s*\{([^{}]*)\}/g, '^$1'],
    [/\\(?:text|texttt|mathrm|mathbf|mathit|mathbb|mathcal|mathsf|operatorname|bm|boldsymbol|overline)\s*\{([^{}]*)\}/g, '$1'],
    [/\\\{/g, '｛'], [/\\\}/g, '｝'], [/\\_/g, '_'],
    [/\\[tdc]?frac\s*(\d)\s*(\d)/g, '$1/$2'],
    [/\\sqrt\s*\{([^{}]*)\}/g, '√($1)'],
    [/\\begin\{[bp]?matrix\}([\s\S]*?)\\end\{[bp]?matrix\}/g, function (m, body) {
      return '[' + body.replace(/\\\\/g, '; ').replace(/&/g, ',') + ']';
    }],
    [/\\([%#&])/g, '$1'],
    [/\\(?:qquad|quad)/g, '  '],
    [/\\(?:bigg?|Bigg?)[lr]?/g, ''],
    [/\\hat\s*\{?([A-Za-z])\}?/g, '$1̂'],
    [/\\dot\s*\{?([A-Za-z])\}?/g, '$1̇'],
    [/\\(?:left(?!arrow)|right(?!arrow)|,|;|!|\s)/g, ' '],
    [/\\(?:cdots|ldots|dots)/g, '⋯'],
    [/\\cdot/g, '·'], [/\\times/g, '×'], [/\\approx/g, '≈'],
    [/\\le(?:q)?(?![A-Za-z])/g, '≤'], [/\\ge(?:q)?(?![A-Za-z])/g, '≥'], [/\\neq(?![A-Za-z])/g, '≠'],
    [/\\to(?![A-Za-z])/g, '→'], [/\\rightarrow(?![A-Za-z])/g, '→'], [/\\leftarrow(?![A-Za-z])/g, '←'],
    [/\\in(?![A-Za-z])/g, '∈'], [/\\sum(?![A-Za-z])/g, 'Σ'], [/\\infty(?![A-Za-z])/g, '∞'],
    [/\\mid(?![A-Za-z])/g, '|'], [/\\sim(?![A-Za-z])/g, '~'], [/\\pm(?![A-Za-z])/g, '±'],
    /* ‖·‖ 与 KL 里的分隔符：没有这一条，\|a - b\|^2 会把原始 TeX 漏出来。
       \lVert / \rVert 与 \lvert / \rvert 是同两个符号的「成对」写法，一起认掉。 */
    [/\\\|/g, '‖'], [/\\[lr]Vert/g, '‖'], [/\\[lr]?vert(?![A-Za-z])/g, '|'],
    [/\\bar\s*\{?([A-Za-z])\}?/g, '$1\u0304'], [/\\ell(?![A-Za-z])/g, 'ℓ'],
    [/\\propto(?![A-Za-z])/g, '∝'],
    [/\\(cos|sin|tanh|tan|arg)(?![a-z])/g, '$1'],
    [/\\min(?![A-Za-z])/g, 'min'], [/\\max(?![A-Za-z])/g, 'max'], [/\\exp(?![A-Za-z])/g, 'exp'], [/\\log(?![A-Za-z])/g, 'log'],
    [/\\partial/g, '∂'], [/\\nabla/g, '∇'], [/\\odot(?![A-Za-z])/g, '⊙'], [/\\div(?![A-Za-z])/g, '÷'],
    [/\\ast(?![A-Za-z])/g, '*'], [/\\longrightarrow(?![A-Za-z])/g, '⟶'],
    [/\\star(?![A-Za-z])/g, '⋆'], [/\\equiv(?![A-Za-z])/g, '≡'], [/\\Leftrightarrow(?![A-Za-z])/g, '⇔'],
    [/\\(?:Longrightarrow|Rightarrow)(?![A-Za-z])/g, '⇒'], [/\\textstyle(?![A-Za-z])/g, ''],
    [/\\alpha/g, 'α'], [/\\beta/g, 'β'], [/\\gamma/g, 'γ'], [/\\delta/g, 'δ'],
    [/\\epsilon/g, 'ε'], [/\\varepsilon/g, 'ε'], [/\\theta/g, 'θ'], [/\\lambda/g, 'λ'],
    [/\\mu/g, 'μ'], [/\\pi/g, 'π'], [/\\sigma/g, 'σ'], [/\\tau/g, 'τ'],
    [/\\eta/g, 'η'], [/\\zeta/g, 'ζ'], [/\\xi/g, 'ξ'], [/\\rho/g, 'ρ'], [/\\kappa/g, 'κ'], [/\\nu/g, 'ν'], [/\\chi/g, 'χ'],
    [/\\phi/g, 'φ'], [/\\varphi/g, 'φ'], [/\\psi/g, 'ψ'], [/\\omega/g, 'ω'],
    [/\\Delta/g, 'Δ'], [/\\Gamma/g, 'Γ'], [/\\Lambda/g, 'Λ'], [/\\Theta/g, 'Θ'], [/\\Pi/g, 'Π'],
    [/\\Phi/g, 'Φ'], [/\\Psi/g, 'Ψ'], [/\\Sigma/g, 'Σ'], [/\\Omega/g, 'Ω'],
    [/\^\{?\\?circ\}?/g, '°'],
    /* Accents and roots over a Greek letter: only a single glyph is left by now. */
    [/\\hat\s*\{?([^{}\s\\])\}?/g, '$1\u0302'], [/\\bar\s*\{?([^{}\s\\])\}?/g, '$1\u0304'],
    [/\\dot\s*\{?([^{}\s\\])\}?/g, '$1\u0307'], [/\\ddot\s*\{?([^{}\s\\])\}?/g, '$1\u0308'],
    [/\\sqrt\s*\{([^{}]*)\}/g, '√($1)'],
    /* A fraction whose parts only lost their braces above. */
    [/_\s*\{([^{}]*)\}/g, '_$1'], [/\^\s*\{([^{}]*)\}/g, '^$1'], [TEX_FRAC, '($1)/($2)'],
    [/[{}]/g, ''], [/\s{2,}/g, ' ']
  ];

  function texToPlain(tex) {
    var s = String(tex);
    /* Sub/superscript braces come off first (TEX_PLAIN[0..1]) so that a
       fraction whose parts carry them still matches TEX_FRAC — and they peel
       from the inside out, or a nested one (π_{θ_{old}}) keeps its braces and
       takes the surrounding \frac down with it. Then the fractions. */
    for (var sub = 0; sub < 3 && /[_^]\s*\{/.test(s); sub++) {
      s = s.replace(TEX_PLAIN[0][0], TEX_PLAIN[0][1]).replace(TEX_PLAIN[1][0], TEX_PLAIN[1][1]);
    }
    for (var pass = 0; pass < 3 && TEX_FRAC_ANY.test(s); pass++) {
      s = s.replace(TEX_FRAC, '($1)/($2)');
    }
    for (var i = 2; i < TEX_PLAIN.length; i++) s = s.replace(TEX_PLAIN[i][0], TEX_PLAIN[i][1]);
    return s.trim();
  }

  function katexReady() {
    return !!(window.katex && typeof window.katex.render === 'function');
  }

  /* iOS WebKit paints KaTeX's HTML output (a stack of position:relative
     offsets) at the SVG origin when it sits inside a <foreignObject> — the
     same compositing bug assets/js/mermaid-config.js works around by asking
     Mermaid for native MathML there. MathML creates no layer, so use it. */
  function isIos() {
    if (document.documentElement.classList.contains('ios')) return true;
    if (typeof navigator === 'undefined') return false;
    return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  }

  var pendingTex = [];
  var texWatch = null;

  function queueTex(job) {
    pendingTex.push(job);
    if (texWatch !== null) return;
    var tries = 0;
    texWatch = window.setInterval(function () {
      if (katexReady()) {
        window.clearInterval(texWatch);
        texWatch = null;
        var queue = pendingTex;
        pendingTex = [];
        queue.forEach(function (j) { renderTex(j.target, j.tex, j.opts); });
      } else if (++tries > 50) {
        // Give up rather than leave a timer running for the rest of the visit.
        window.clearInterval(texWatch);
        texWatch = null;
        pendingTex = [];
      }
    }, 100);
  }

  function renderTex(target, tex, opts) {
    var o = opts || {};
    if (katexReady()) {
      try {
        window.katex.render(String(tex), target, {
          displayMode: !!o.display,
          output: o.output || 'htmlAndMathml',
          /* Throw rather than paint KaTeX's red raw-TeX error on the page:
             a typo in a formula then reads as plain text instead of shouting
             at the reader. (`%`, `#`, `&` must be escaped inside \text{}.) */
          throwOnError: true,
          strict: false
        });
        return true;
      } catch (e) {
        /* A malformed formula must not take the whole demo down with it. */
      }
    }
    target.textContent = texToPlain(tex);
    if (!katexReady()) queueTex({ target: target, tex: tex, opts: o });
    return false;
  }

  /* An inline formula for the HTML parts of a demo (titles, cues, notes). */
  function tex(str, opts) {
    var o = opts || {};
    var cls = 'demo-tex' + (o.display ? ' demo-tex-display' : '') + (o.cls ? ' ' + o.cls : '');
    var span = el('span', cls);
    span.setAttribute('data-tex', String(str));
    renderTex(span, str, { display: o.display, output: o.output });
    return span;
  }

  function rerenderDemoTex() {
    document.querySelectorAll('.demo-tex[data-tex]').forEach(function (span) {
      renderTex(span, span.getAttribute('data-tex'), {
        display: span.classList.contains('demo-tex-display'),
        output: isIos() ? 'mathml' : 'htmlAndMathml'
      });
    });
  }

  document.addEventListener('katex-ready', rerenderDemoTex);

  /* The tiny markup every human-readable demo string may use:
     `**bold**`, `` `code` `` and `$LaTeX$`. A code run and a formula never
     nest inside each other, but either can sit inside a bold run
     (`**$r_t(\theta)$**`, ``**`disc_reward_weight: 1.0`**``). */
  function rich(parent, str, opts) {
    var s = String(str);
    var re = /\*\*|`[^`]+`|\$[^$]+\$/g;
    var target = parent, at = 0, m;
    while ((m = re.exec(s)) !== null) {
      if (m.index > at) target.appendChild(document.createTextNode(s.slice(at, m.index)));
      at = m.index + m[0].length;
      if (m[0] === '**') {
        if (target === parent) {
          target = el('b');
          parent.appendChild(target);
        } else {
          target = parent;
        }
      } else if (m[0].charAt(0) === '`') {
        target.appendChild(el('code', 'demo-code', m[0].slice(1, -1)));
      } else {
        target.appendChild(tex(m[0].slice(1, -1), opts));
      }
    }
    if (at < s.length) target.appendChild(document.createTextNode(s.slice(at)));
    return parent;
  }

  /* What a screen reader (an aria-label) should hear for a rich() string:
     the markup comes off and each `$…$` reads as its texToPlain(). */
  function richToPlain(str) {
    return String(str)
      .replace(/\$([^$]+)\$/g, function (m, t) { return texToPlain(t); })
      .replace(/\*\*/g, '')
      .replace(/`([^`]+)`/g, '$1');
  }

  function card(host, opts) {
    host.innerHTML = '';
    var root = el('div', 'demo-card');
    var head = el('div', 'demo-head');
    head.appendChild(el('span', 'demo-badge', opts.badge || '交互演示'));
    head.appendChild(rich(el('h4', 'demo-title'), opts.title));
    if (opts.sub) head.appendChild(rich(el('p', 'demo-sub'), opts.sub));
    root.appendChild(head);
    host.appendChild(root);
    return root;
  }

  /* A recorded explainer video with a download button. `<video>` never survives
     scripts/sanitize_paper_html.py, so the note ships a placeholder whose
     `data-src` / `data-poster` (relative to the note page) point at the files
     and whose fallback is a plain download link; this builds the player.
     Options: { title, sub, src, poster, size, fileName }. */
  function video(host, opts) {
    var o = opts || {};
    var src = o.src || host.getAttribute('data-src');
    var poster = o.poster || host.getAttribute('data-poster');
    var fileName = o.fileName || src.split('/').pop();
    var root = card(host, { title: o.title, sub: o.sub, badge: '讲解视频' });
    var frame = el('div', 'demo-video');
    var player = document.createElement('video');
    player.controls = true;
    player.preload = 'none';
    player.setAttribute('playsinline', '');
    if (poster) player.poster = poster;
    player.setAttribute('aria-label', richToPlain(o.title || fileName));
    var source = document.createElement('source');
    source.src = src;
    source.type = 'video/mp4';
    player.appendChild(source);
    frame.appendChild(player);
    root.appendChild(frame);

    var row = el('div', 'demo-buttons demo-video-actions');
    var link = el('a', 'demo-btn', '下载视频' + (o.size ? '（mp4，' + o.size + '）' : '（mp4）'));
    link.href = src;
    link.setAttribute('download', fileName);
    row.appendChild(link);
    root.appendChild(row);
    return player;
  }

  function controlsRow(parent) {
    var row = el('div', 'demo-controls');
    parent.appendChild(row);
    return row;
  }

  /* A labelled range input. `format` renders the value, `onInput` gets the number. */
  function slider(parent, opts) {
    var wrap = el('div', 'demo-control');
    var label = el('div', 'demo-control-label');
    label.appendChild(el('span', null, opts.label));
    var valueEl = el('span', 'demo-control-value');
    label.appendChild(valueEl);
    wrap.appendChild(label);

    var input = document.createElement('input');
    input.type = 'range';
    input.min = String(opts.min);
    input.max = String(opts.max);
    input.step = String(opts.step);
    input.value = String(opts.value);
    if (opts.ariaLabel || opts.label) input.setAttribute('aria-label', opts.ariaLabel || richToPlain(opts.label));
    wrap.appendChild(input);
    parent.appendChild(wrap);

    var api = {
      input: input,
      get: function () {
        return parseFloat(input.value);
      },
      set: function (v, silent) {
        input.value = String(v);
        valueEl.textContent = opts.format ? opts.format(api.get()) : String(api.get());
        if (!silent && opts.onInput) opts.onInput(api.get());
      },
      refresh: function () {
        valueEl.textContent = opts.format ? opts.format(api.get()) : String(api.get());
      }
    };

    input.addEventListener('input', function () {
      api.refresh();
      if (opts.onInput) opts.onInput(api.get());
    });
    api.refresh();
    return api;
  }

  function button(parent, text, onClick) {
    var b = el('button', 'demo-btn', text);
    b.type = 'button';
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  }

  function checkbox(parent, text, checked, onChange) {
    var label = el('label', 'demo-toggle');
    var input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !!checked;
    label.appendChild(input);
    label.appendChild(el('span', null, text));
    parent.appendChild(label);
    input.addEventListener('change', function () {
      onChange(input.checked);
    });
    return input;
  }

  function statsRow(parent) {
    var row = el('div', 'demo-readout');
    parent.appendChild(row);
    var api = {
      root: row,
      add: function (key) {
        var box = el('div', 'demo-stat');
        box.appendChild(el('span', 'demo-stat-k', key));
        var v = el('span', 'demo-stat-v', '—');
        box.appendChild(v);
        row.appendChild(box);
        return {
          set: function (text, tone) {
            v.textContent = '';
            if (needsRichMarkup(text)) rich(v, text);
            else v.textContent = text;
            v.className = 'demo-stat-v' + (tone ? ' is-' + tone : '');
          }
        };
      }
    };
    return api;
  }

  function verdictBox(parent) {
    var box = el('div', 'demo-verdict');
    parent.appendChild(box);
    return {
      set: function (html, tone) {
        box.textContent = '';
        if (html instanceof Node) {
          box.appendChild(html);
        } else {
          // same tiny markup as note(): **bold** and $LaTeX$
          rich(box, html);
        }
        box.className = 'demo-verdict' + (tone ? ' is-' + tone : '');
      }
    };
  }

  function legend(parent, items) {
    var row = el('div', 'demo-legend');
    var spans = [];
    items.forEach(function (it) {
      var s = el('span', null, it.text);
      row.appendChild(s);
      spans.push({ span: s, key: it.key });
    });
    parent.appendChild(row);
    return function (palette) {
      spans.forEach(function (s) {
        s.span.style.color = palette[s.key] || palette.muted;
      });
    };
  }

  function note(parent, lines) {
    var box = el('div', 'demo-note');
    lines.forEach(function (line) {
      // Lines use a tiny markup: **bold** and $LaTeX$ (see rich()).
      box.appendChild(rich(el('p'), line));
    });
    parent.appendChild(box);
  }

  // ─── canvas kit ──────────────────────────────────────────────────────────
  function palette() {
    var cs = getComputedStyle(document.documentElement);
    function v(name, fallback) {
      var got = cs.getPropertyValue(name);
      return (got && got.trim()) || fallback;
    }
    return {
      text: v('--text', '#d4d4d4'),
      muted: v('--demo-muted', '#8a8a8a'),
      grid: v('--demo-grid', '#37373a'),
      border: v('--demo-border', '#3c3c3c'),
      surface: v('--demo-surface', '#212124'),
      surface2: v('--demo-surface-2', '#2a2a2e'),
      accent: v('--demo-accent', '#569cd6'),
      good: v('--demo-good', '#4ec9a4'),
      bad: v('--demo-bad', '#e06c75'),
      warn: v('--demo-warn', '#d7a75f')
    };
  }

  function stage(parent, height) {
    var box = el('div', 'demo-stage');
    var canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    box.appendChild(canvas);
    parent.appendChild(box);
    return { box: box, canvas: canvas, height: height };
  }

  function stageGrid(parent) {
    var grid = el('div', 'demo-stage-grid');
    parent.appendChild(grid);
    return grid;
  }

  /* Prepare a HiDPI canvas and return a drawing context in CSS pixels. */
  function begin(st) {
    var canvas = st.canvas;
    var cssW = Math.max(240, canvas.parentElement.clientWidth || 300);
    var cssH = st.height;
    var dpr = window.devicePixelRatio || 1;
    canvas.style.height = cssH + 'px';
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    var P = palette();
    ctx.fillStyle = P.surface2;
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif';
    ctx.textBaseline = 'middle';
    return { ctx: ctx, w: cssW, h: cssH, P: P };
  }

  /* Linear scales for a plot area inset by `pad`. */
  function plot(g, pad, xd, yd) {
    var x0 = pad.l,
      x1 = g.w - pad.r,
      y0 = g.h - pad.b,
      y1 = pad.t;
    return {
      x0: x0,
      x1: x1,
      y0: y0,
      y1: y1,
      sx: function (v) {
        return x0 + ((v - xd[0]) / (xd[1] - xd[0])) * (x1 - x0);
      },
      sy: function (v) {
        return y0 + ((v - yd[0]) / (yd[1] - yd[0])) * (y1 - y0);
      },
      ux: function (px) {
        return xd[0] + ((px - x0) / (x1 - x0)) * (xd[1] - xd[0]);
      },
      xd: xd,
      yd: yd
    };
  }

  function line(ctx, pts, color, width, dash) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    for (var i = 0; i < pts.length; i++) {
      if (i === 0) ctx.moveTo(pts[i][0], pts[i][1]);
      else ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.stroke();
    ctx.restore();
  }

  function dot(ctx, x, y, r, fill, stroke) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }
    ctx.restore();
  }

  function text(ctx, str, x, y, color, align, font) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.textAlign = align || 'left';
    if (font) ctx.font = font;
    ctx.fillText(str, x, y);
    ctx.restore();
  }

  /* Value label for a bar: above its top, or tucked just inside when the bar
     is so tall that the label would collide with the chart title. */
  function barLabel(g, p, x, yTop, str, color) {
    var above = yTop - 9;
    if (above < p.y1 + 12) {
      text(g.ctx, str, x, yTop + 12, g.P.surface2, 'center', '11px monospace');
    } else {
      text(g.ctx, str, x, above, color || g.P.text, 'center', '11px monospace');
    }
  }

  function axes(g, p, opts) {
    var ctx = g.ctx,
      P = g.P;
    ctx.save();
    ctx.strokeStyle = P.grid;
    ctx.lineWidth = 1;
    (opts.yTicks || []).forEach(function (t) {
      var y = Math.round(p.sy(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p.x0, y);
      ctx.lineTo(p.x1, y);
      ctx.stroke();
      text(ctx, opts.yFmt ? opts.yFmt(t) : String(t), p.x0 - 6, y, P.muted, 'right', '11px monospace');
    });
    (opts.xTicks || []).forEach(function (t) {
      var x = Math.round(p.sx(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, p.y1);
      ctx.lineTo(x, p.y0);
      ctx.stroke();
      text(ctx, opts.xFmt ? opts.xFmt(t) : String(t), x, p.y0 + 13, P.muted, 'center', '11px monospace');
    });
    ctx.restore();
    if (opts.xLabel) text(ctx, opts.xLabel, p.x1, p.y0 + 27, P.muted, 'right', '11px sans-serif');
    if (opts.yLabel) text(ctx, opts.yLabel, p.x0 - 4, p.y1 - 9, P.muted, 'right', '11px sans-serif');
  }

  // ─── re-render plumbing (theme switch + container resize) ────────────────
  var renderers = [];

  function registerRenderer(fn) {
    renderers.push(fn);
    return fn;
  }

  function renderAll() {
    for (var i = 0; i < renderers.length; i++) {
      try {
        renderers[i]();
      } catch (e) {
        /* one broken demo must not take down the rest of the page */
      }
    }
  }

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderAll, 120);
  });

  if (window.MutationObserver) {
    new MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        if (records[i].attributeName === 'data-theme') {
          renderAll();
          return;
        }
      }
    }).observe(document.documentElement, { attributes: true });
  }

  function niceTicks(lo, hi, count) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var raw = span / count;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
    var ticks = [];
    for (var t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
      ticks.push(Math.abs(t) < 1e-9 ? 0 : t);
    }
    return ticks;
  }


  /* A row of mutually exclusive buttons (the "which variant am I looking at"
     control). `onPick` gets the chosen item's value. */
  function buttonGroup(parent, opts) {
    var wrap = el('div', 'demo-control demo-buttons');
    if (opts.label) {
      wrap = el('div', 'demo-control');
      var lab = el('div', 'demo-control-label');
      lab.appendChild(el('span', null, opts.label));
      wrap.appendChild(lab);
      var row = el('div', 'demo-buttons');
      wrap.appendChild(row);
      parent.appendChild(wrap);
      wrap = row;
    } else {
      parent.appendChild(wrap);
    }

    var btns = [];
    var api = {
      pick: function (value, silent) {
        btns.forEach(function (b) {
          b.node.className = 'demo-btn' + (b.value === value ? ' is-active' : '');
        });
        api.value = value;
        if (!silent && opts.onPick) opts.onPick(value);
      }
    };
    opts.items.forEach(function (it) {
      var node = button(wrap, it.label, function () {
        api.pick(it.value);
      });
      btns.push({ node: node, value: it.value });
    });
    api.pick(opts.value, true);
    return api;
  }

  /* A horizontally scrollable table. `rows` are rebuilt by the caller through
     the returned api; the wrapper keeps the phone layout from overflowing. */
  function table(parent) {
    var wrap = el('div', 'demo-table-wrap');
    var node = el('table', 'demo-table');
    wrap.appendChild(node);
    parent.appendChild(wrap);
    return {
      node: node,
      clear: function () {
        node.innerHTML = '';
      },
      row: function (cells, isHead) {
        var tr = el('tr');
        cells.forEach(function (c, i) {
          var tag = i === 0 || isHead ? 'th' : 'td';
          var cls = c && typeof c === 'object' ? c.cls : null;
          var cell = el(tag, cls);
          if (c == null) {
            /* empty cell */
          } else if (typeof c === 'object' && c.text != null) {
            if (needsRichMarkup(c.text)) rich(cell, c.text);
            else cell.textContent = c.text;
          } else if (typeof c === 'string' || typeof c === 'number') {
            if (needsRichMarkup(String(c))) rich(cell, String(c));
            else cell.textContent = String(c);
          }
          tr.appendChild(cell);
        });
        node.appendChild(tr);
        return tr;
      }
    };
  }

  /* −/+ spinner used inside demo tables for hand-editable numbers. */
  function stepper(cell, opts) {
    var box = el('span', 'demo-stepper');
    var minus = el('button', null, '−');
    minus.type = 'button';
    minus.setAttribute('aria-label', (opts.label || '') + ' 减小');
    var value = el('span', 'demo-stepper-v', fmt(opts.get(), opts.digits));
    var plus = el('button', null, '+');
    plus.type = 'button';
    plus.setAttribute('aria-label', (opts.label || '') + ' 增大');
    function bump(dir) {
      opts.set(opts.get() + dir * opts.step);
      value.textContent = fmt(opts.get(), opts.digits);
      if (opts.onChange) opts.onChange();
    }
    minus.addEventListener('click', function () {
      bump(-1);
    });
    plus.addEventListener('click', function () {
      bump(1);
    });
    box.appendChild(minus);
    box.appendChild(value);
    box.appendChild(plus);
    cell.appendChild(box);
    return {
      refresh: function () {
        value.textContent = fmt(opts.get(), opts.digits);
      }
    };
  }

  // ─── SVG storyboard kit (for the narrated explainer animations) ──────────
  /* The explainers draw on SVG rather than Canvas: the scenes are mostly text
     and boxes, and SVG keeps them crisp and selectable. Colors go through
     inline style (not presentation attributes) so `var(--demo-*)` resolves in
     every browser and the site's theme switch needs no repaint hook. */

  var SVG_NS = 'http://www.w3.org/2000/svg';

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function svgText(x, y, str, cls, size, anchor) {
    var node = svgEl('text', { x: x, y: y, 'font-size': size || 12 });
    if (cls) node.setAttribute('class', cls);
    if (anchor) node.setAttribute('text-anchor', anchor);
    node.textContent = str;
    return node;
  }

  /* A real formula on a storyboard. SVG has no math typesetting, so the
     KaTeX output is hosted in a <foreignObject>; the box is sized generously
     and the formula aligned inside it, which keeps the call site looking like
     svgText (x, y, anchor) with no measure-then-place round trip.
     `y` is the baseline svgText would use, so a label and a formula on the
     same row line up.

     svgMath returns the <foreignObject>, with `setTex` for formulas whose
     numbers change while the scene plays, `setX` / `setY` for a label that
     follows something around the scene, and `setCls` / `setTone` for its
     colour — `paint()` cannot help here, HTML takes `color`, not `fill`.
     Options: { size, anchor, cls, w, h, display }. svgBox is the part it
     shares with svgRich below. */
  function svgBox(x, y, o, defaultW, baseCls) {
    var size = o.size || 12;
    var w = o.w || defaultW;
    var h = o.h || size * (o.display ? 3.4 : 2.4);
    var anchor = o.anchor || 'start';
    var fo = svgEl('foreignObject', {
      x: anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x,
      /* Centre the box on where the text's optical middle would be, so the
         formula sits on the row rather than hanging below its baseline. */
      y: y - 0.34 * size - h / 2,
      width: w,
      height: h,
      class: 'demo-x-fo'
    });
    var box = document.createElement('div');
    box.style.fontSize = size + 'px';
    box.style.justifyContent = anchor === 'middle' ? 'center' : anchor === 'end' ? 'flex-end' : 'flex-start';
    fo.appendChild(box);

    fo.texBox = box;
    /* The box is wider than the formula, so a call site that moves a label
       around (a band that follows the camera) must go through this rather
       than set `x` itself. */
    fo.setX = function (nx) {
      fo.setAttribute('x', anchor === 'middle' ? nx - w / 2 : anchor === 'end' ? nx - w : nx);
      return fo;
    };
    /* `ny` is a baseline, like svgText's `y`. */
    fo.setY = function (ny) {
      fo.setAttribute('y', ny - 0.34 * size - h / 2);
      return fo;
    };
    fo.setCls = function (cls) {
      box.className = baseCls + (cls ? ' ' + cls : '');
      return fo;
    };
    fo.setTone = function (color) {
      box.style.color = color || '';
      return fo;
    };
    fo.setCls(o.cls);
    return fo;
  }

  function svgMath(x, y, str, opts) {
    var o = opts || {};
    var fo = svgBox(x, y, o, 300, 'demo-x-tex');
    var output = isIos() ? 'mathml' : 'htmlAndMathml';
    fo.setTex = function (next) {
      if (fo.texSource === next) return fo;
      fo.texSource = next;
      renderTex(fo.texBox, next, { display: o.display, output: output });
      return fo;
    };
    fo.setTex(str);
    return fo;
  }

  /* A storyboard label that mixes prose and formulas — the svgText of a line
     like "在 $\mathcal{D}_{\text{loco}}$ 里检索". It takes the same markup as
     rich() (`$…$`, `**bold**`, `` `code` ``) in svgMath's <foreignObject>,
     so the prose keeps the scene's own font (inside KaTeX's \text{} it
     inherits `KaTeX_Main, Times New Roman, serif`, which turns CJK into a
     serif face wherever one is installed) and only the `$…$` runs are typeset.
     Same call shape and methods as svgMath, with `setText` for a label whose
     numbers change while the scene plays. */
  function svgRich(x, y, str, opts) {
    var o = opts || {};
    var fo = svgBox(x, y, o, 760, 'demo-x-tex demo-x-rich');
    var output = isIos() ? 'mathml' : 'htmlAndMathml';
    /* The flex box would trim the spaces around each formula (every text run
       becomes its own flex item), so the runs sit in one inline span. */
    var line = document.createElement('span');
    fo.texBox.appendChild(line);
    fo.setText = function (next) {
      if (fo.textSource === next) return fo;
      fo.textSource = next;
      line.textContent = '';
      rich(line, next, { output: output });
      return fo;
    };
    fo.setText(str);
    return fo;
  }

  function paint(node, fill, stroke) {
    if (fill) node.style.fill = fill;
    if (stroke) node.style.stroke = stroke;
    return node;
  }

  /* Progress of `t` through the window [a, b], clamped to 0…1. */
  function seg(t, a, b) {
    return clamp((t - a) / (b - a), 0, 1);
  }

  function ease(x) {
    return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
  }

  function setOpacity(node, v) {
    node.style.opacity = v;
  }

  /* A scene canvas on the shared 800×420 grid the explainer CSS is sized for. */
  function sceneSvg(label, viewBox) {
    return svgEl('svg', {
      viewBox: viewBox || '0 0 800 420',
      class: 'demo-x-svg',
      role: 'img',
      'aria-label': label
    });
  }

  /* An arrowhead <marker>; ids are document-global, so pass a bundle-unique one. */
  function arrowMarker(svg, id, color) {
    var defs = svgEl('defs', {});
    var marker = svgEl('marker', {
      id: id, viewBox: '0 0 10 10', refX: 9, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse'
    });
    marker.appendChild(paint(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z' }), color));
    defs.appendChild(marker);
    svg.appendChild(defs);
    return 'url(#' + id + ')';
  }

  var X_COLORS = {
    accent: 'var(--demo-accent)',
    good: 'var(--demo-good)',
    bad: 'var(--demo-bad)',
    warn: 'var(--demo-warn)',
    muted: 'var(--demo-muted)',
    grid: 'var(--demo-grid)',
    border: 'var(--demo-border)',
    surface: 'var(--demo-surface)',
    surface2: 'var(--demo-surface-2)',
    ink: 'var(--text)',
    ink2: 'var(--text-secondary)'
  };

  var TAU = Math.PI * 2;

  /* ── a schematic stick figure ──
     Angles are global and measured in degrees from straight-down, positive
     towards +x (the direction the character faces), so a limb is one call.
     `pose(x, y, P)` puts the hip at (x, y); P holds the five global angles
     { lean, armA: [shoulder, elbow], armB, legA: [hip, knee], legB }.
     `farAlpha` (optional) fades armB / legB, so in a gait you can tell the
     far-side limbs from the near ones. */
  function limbPt(x, y, len, ang) {
    var r = (ang * Math.PI) / 180;
    return [x + len * Math.sin(r), y + len * Math.cos(r)];
  }

  function stickFigure(color, w, dashed, farAlpha) {
    var g = svgEl('g', {});
    function bone() {
      var ln = paint(svgEl('line', { 'stroke-width': w, 'stroke-linecap': 'round' }), null, color);
      if (dashed) ln.setAttribute('stroke-dasharray', '5 4');
      g.appendChild(ln);
      return ln;
    }
    var spine = bone(),
      armA1 = bone(), armA2 = bone(), armB1 = bone(), armB2 = bone(),
      legA1 = bone(), legA2 = bone(), legB1 = bone(), legB2 = bone();
    if (farAlpha != null) {
      [armB1, armB2, legB1, legB2].forEach(function (ln) { ln.style.opacity = farAlpha; });
    }
    var head = paint(svgEl('circle', { r: 8.5, fill: 'none', 'stroke-width': w }), null, color);
    if (dashed) head.setAttribute('stroke-dasharray', '5 4');
    g.appendChild(head);

    function put(ln, p, q) {
      ln.setAttribute('x1', p[0].toFixed(1));
      ln.setAttribute('y1', p[1].toFixed(1));
      ln.setAttribute('x2', q[0].toFixed(1));
      ln.setAttribute('y2', q[1].toFixed(1));
    }

    function pose(x, y, P) {
      var hip = [x, y];
      var sh = limbPt(x, y, 42, 180 - P.lean);
      put(spine, hip, sh);
      var hd = limbPt(sh[0], sh[1], 15, 180 - P.lean);
      head.setAttribute('cx', hd[0].toFixed(1));
      head.setAttribute('cy', hd[1].toFixed(1));
      [[armA1, armA2, P.armA], [armB1, armB2, P.armB]].forEach(function (a) {
        var e = limbPt(sh[0], sh[1], 20, a[2][0]);
        put(a[0], sh, e);
        put(a[1], e, limbPt(e[0], e[1], 18, a[2][1]));
      });
      [[legA1, legA2, P.legA], [legB1, legB2, P.legB]].forEach(function (l) {
        var k = limbPt(x, y, 24, l[2][0]);
        put(l[0], hip, k);
        put(l[1], k, limbPt(k[0], k[1], 24, l[2][1]));
      });
    }

    return { el: g, pose: pose };
  }

  /* 一段普通走路：手脚对称摆动、膝盖在摆动相折叠。`ph` 是 0…1 的相位。 */
  function poseWalk(ph) {
    var s = Math.sin(ph * TAU),
      c = Math.cos(ph * TAU);
    return {
      lean: 6,
      armA: [-22 * s - 8, -22 * s + 16],
      armB: [22 * s + 8, 22 * s + 32],
      legA: [22 * s, 22 * s - 30 * Math.max(0, c)],
      legB: [-22 * s, -22 * s - 30 * Math.max(0, -c)]
    };
  }

  /* `M x y L x y …` through a polyline, and the point a fraction `u` along it —
     enough to draw a flow-chart edge and walk a token down it. */
  function polyPath(pts) {
    return pts.map(function (p, i) { return (i ? 'L ' : 'M ') + p[0] + ' ' + p[1]; }).join(' ');
  }

  function pointOn(pts, u) {
    var lens = [], total = 0, i;
    for (i = 1; i < pts.length; i++) {
      var d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      lens.push(d);
      total += d;
    }
    var want = u * total;
    for (i = 0; i < lens.length; i++) {
      if (want <= lens[i] || i === lens.length - 1) {
        var k = lens[i] ? want / lens[i] : 0;
        return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * k, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * k];
      }
      want -= lens[i];
    }
    return pts[pts.length - 1];
  }

  /* 幕号两位数：01 … 09、10（十幕的讲解动画不能显示成 010） */
  function sceneNo(i) {
    return (i + 1 < 10 ? '0' : '') + (i + 1);
  }

  function clockText(sec) {
    var m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  /* A narrated, auto-playing storyboard.
   *
   *   explainer(host, {
   *     title, sub, ariaLabel, notes: ['取数依据：…'],
   *     scenes: [{ title, dur, cues: [{ at, s }], build: function () {
   *       return { el: <svg>, draw: function (t) {…} };
   *     } }]
   *   })
   *
   * Cue text may use **bold** spans and $LaTeX$ formulas. Playback starts
   * when the frame scrolls into view (a note is long — an animation that
   * finished above the fold helps nobody) and pauses when it leaves. With
   * prefers-reduced-motion each scene is shown at its final frame and
   * nothing animates. */
  function explainer(host, opts) {
    var specs = opts.scenes;
    var root = card(host, { title: opts.title, sub: opts.sub });

    var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var total = specs.reduce(function (a, sc) { return a + sc.dur; }, 0);

    var chipRow = el('div', 'demo-x-chips');
    root.appendChild(chipRow);

    var frame = el('div', 'demo-x-frame');
    var head = el('div', 'demo-x-head');
    var noEl = el('span', 'demo-x-no', '01');
    var titleEl = rich(el('span', 'demo-x-title'), specs[0].title);
    var clockEl = el('span', 'demo-x-clock', '0:00 / ' + clockText(total));
    [noEl, titleEl, clockEl].forEach(function (n) { head.appendChild(n); });
    frame.appendChild(head);

    var scroller = el('div', 'demo-x-scroll');
    var stageEl = el('div', 'demo-x-stage');
    scroller.appendChild(stageEl);
    frame.appendChild(scroller);

    var track = el('div', 'demo-x-track');
    var fill = el('div', 'demo-x-track-fill');
    track.appendChild(fill);
    frame.appendChild(track);

    var cueBox = el('div', 'demo-x-cues');
    frame.appendChild(cueBox);
    root.appendChild(frame);

    var scenes = specs.map(function (spec) {
      var built = spec.build();
      var wrap = el('div', 'demo-x-scene');
      wrap.appendChild(built.el);
      wrap.hidden = true;
      stageEl.appendChild(wrap);
      return { spec: spec, wrap: wrap, draw: built.draw };
    });

    var chips = specs.map(function (spec, i) {
      var b = el('button', 'demo-x-chip');
      b.type = 'button';
      b.appendChild(el('span', 'demo-x-chip-n', sceneNo(i)));
      rich(b, spec.title);
      b.addEventListener('click', function () { go(i, true); });
      chipRow.appendChild(b);
      return b;
    });

    var row = controlsRow(root);
    var playBtn = button(row, '播放', function () {
      if (!playing && idx === scenes.length - 1 && time >= scenes[idx].spec.dur) {
        go(0, true);
        return;
      }
      setPlaying(!playing);
    });
    button(row, '上一幕', function () { go(idx - 1, true); });
    button(row, '下一幕', function () { go(idx + 1, true); });
    button(row, '从头播', function () { go(0, true); });

    if (opts.notes) note(root, opts.notes);

    var idx = 0, time = 0, playing = false, last = 0, started = false, autoPaused = false, raf = null;

    function buildCues() {
      cueBox.textContent = '';
      scenes[idx].spec.cues.forEach(function (cue) {
        var p = rich(el('p', 'demo-x-cue'), cue.s);
        p.setAttribute('data-at', cue.at);
        cueBox.appendChild(p);
      });
    }

    function paintFrame() {
      var sc = scenes[idx];
      sc.draw(Math.min(time, sc.spec.dur));
      fill.style.width = (Math.min(time / sc.spec.dur, 1) * 100).toFixed(1) + '%';
      var before = 0;
      for (var k = 0; k < idx; k++) before += scenes[k].spec.dur;
      clockEl.textContent = clockText(before + Math.min(time, sc.spec.dur)) + ' / ' + clockText(total);
      var cues = cueBox.children;
      for (k = 0; k < cues.length; k++) {
        cues[k].classList.toggle('is-on', time >= parseFloat(cues[k].getAttribute('data-at')));
      }
      playBtn.textContent = playing ? '暂停' : '播放';
    }

    function go(next, fromUser) {
      idx = clamp(next, 0, scenes.length - 1);
      time = reduceMotion ? scenes[idx].spec.dur : 0;
      scenes.forEach(function (sc, i) { sc.wrap.hidden = i !== idx; });
      noEl.textContent = sceneNo(idx);
      titleEl.textContent = '';
      rich(titleEl, scenes[idx].spec.title);
      chips.forEach(function (c, i) { c.setAttribute('aria-current', i === idx ? 'true' : 'false'); });
      buildCues();
      if (fromUser && !reduceMotion) {
        playing = true;
        ensureLoop();
      }
      paintFrame();
    }

    function tick(now) {
      if (!playing) {
        raf = null;
        return;
      }
      if (!last) last = now;
      var dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      time += dt;
      if (time >= scenes[idx].spec.dur) {
        if (idx < scenes.length - 1) {
          go(idx + 1);
          raf = window.requestAnimationFrame(tick);
          return;
        }
        time = scenes[idx].spec.dur;
        playing = false;
      }
      paintFrame();
      raf = playing ? window.requestAnimationFrame(tick) : null;
    }

    function ensureLoop() {
      if (raf === null) {
        last = 0;
        raf = window.requestAnimationFrame(tick);
      }
    }

    function setPlaying(v) {
      playing = v;
      if (v) ensureLoop();
      paintFrame();
    }

    root.addEventListener('keydown', function (e) {
      if (e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        setPlaying(!playing);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        go(idx + 1, true);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        go(idx - 1, true);
      }
    });
    root.setAttribute('tabindex', '0');
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', opts.ariaLabel || opts.title);

    go(0);

    if (!reduceMotion && typeof window.IntersectionObserver === 'function') {
      var io = new window.IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            if (!started) {
              started = true;
              setPlaying(true);
            } else if (autoPaused) {
              autoPaused = false;
              setPlaying(true);
            }
          } else if (playing) {
            autoPaused = true;
            setPlaying(false);
          }
        });
      }, { threshold: 0.4 });
      io.observe(frame);
    }

    return root;
  }

  // ─── deterministic randomness (demos must look the same on every reload) ──
  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rng) {
    var u = 1 - rng(),
      v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function softmax(logits) {
    var m = Math.max.apply(null, logits);
    var e = logits.map(function (x) {
      return Math.exp(x - m);
    });
    var s = e.reduce(function (a, b) {
      return a + b;
    }, 0);
    return e.map(function (x) {
      return x / s;
    });
  }

  // ─── bootstrap ───────────────────────────────────────────────────────────
  /* Build every placeholder this bundle knows about. Safe to call from several
     bundles on the same page: each host is claimed once via data-demo-ready. */
  function mount(builders) {
    function init() {
      var hosts = document.querySelectorAll('.paper-demo[data-demo]');
      for (var i = 0; i < hosts.length; i++) {
        var host = hosts[i];
        var builder = builders[host.getAttribute('data-demo')];
        if (!builder || host.getAttribute('data-demo-ready') === '1') continue;
        try {
          builder(host);
          host.setAttribute('data-demo-ready', '1');
        } catch (e) {
          host.innerHTML = '';
          host.appendChild(el('p', 'demo-fallback', '（交互演示加载失败，请刷新页面重试）'));
        }
      }
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  }

  window.PaperDemoKit = {
    el: el,
    fmt: fmt,
    clamp: clamp,
    card: card,
    video: video,
    controlsRow: controlsRow,
    slider: slider,
    button: button,
    checkbox: checkbox,
    buttonGroup: buttonGroup,
    statsRow: statsRow,
    verdictBox: verdictBox,
    legend: legend,
    note: note,
    table: table,
    stepper: stepper,
    palette: palette,
    stage: stage,
    stageGrid: stageGrid,
    begin: begin,
    plot: plot,
    line: line,
    dot: dot,
    text: text,
    axes: axes,
    barLabel: barLabel,
    niceTicks: niceTicks,
    registerRenderer: registerRenderer,
    renderAll: renderAll,
    tex: tex,
    texToPlain: texToPlain,
    rich: rich,
    richToPlain: richToPlain,
    svgEl: svgEl,
    svgText: svgText,
    svgMath: svgMath,
    svgRich: svgRich,
    paint: paint,
    seg: seg,
    ease: ease,
    setOpacity: setOpacity,
    sceneSvg: sceneSvg,
    arrowMarker: arrowMarker,
    stickFigure: stickFigure,
    poseWalk: poseWalk,
    polyPath: polyPath,
    pointOn: pointOn,
    xColors: X_COLORS,
    explainer: explainer,
    mulberry32: mulberry32,
    gauss: gauss,
    softmax: softmax,
    mount: mount
  };
})();
