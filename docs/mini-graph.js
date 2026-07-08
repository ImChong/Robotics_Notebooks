(function() {
  var miniWrap = document.getElementById('mini-graph-wrap');
  var miniSvg  = document.getElementById('mini-graph-svg');
  var statsEl  = document.getElementById('mini-graph-stats');
  var expandEl = document.getElementById('mini-graph-expand');
  var tooltip  = document.getElementById('mini-graph-tooltip');
  if (!miniWrap || !miniSvg) return;

  var PREVIEW_TOP_N = 40;
  var isMobile = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
  var pinnedNode = null;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function toDetailId(path) {
    return String(path).replace(/\//g, '-').replace('.md', '');
  }

  function pageUrlForNode(d) {
    var detailId = d.detail_id || toDetailId(d.id);
    if (d.type === 'roadmap_page' || (d.id && String(d.id).indexOf('roadmap/') === 0)) {
      return 'roadmap.html?id=' + encodeURIComponent(detailId);
    }
    return 'detail.html?id=' + encodeURIComponent(detailId);
  }

  function miniGraphTheme() {
    var dark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
      background: dark ? '#0d1117' : '#eef2f7',
      edge: dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.10)',
      edge3d: dark ? '#aab6c4' : '#3c4756',
      label: dark ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.70)',
      stats: dark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.62)',
      link: dark ? '#60a5fa' : '#2563eb',
      linkBorder: dark ? 'rgba(96,165,250,0.30)' : 'rgba(37,99,235,0.30)'
    };
  }

  function tooltipSummary(raw) {
    return window.RNGraphTooltip && window.RNGraphTooltip.formatTooltipSummary
      ? window.RNGraphTooltip.formatTooltipSummary(raw, 100)
      : (raw || '');
  }

  function tooltipHtml(d, nodeFill, communityLabelMap) {
    var summary = tooltipSummary(d.summary);
    var detailUrl = pageUrlForNode(d);
    var communityColor = d.community ? nodeFill(d) : '';
    var linkLabel = (d.type === 'roadmap_page' || (d.id && String(d.id).indexOf('roadmap/') === 0))
      ? '打开路线页 →' : '打开详情页 →';
    if (window.RNGraphTooltip && window.RNGraphTooltip.buildNodeTooltipHtml) {
      return window.RNGraphTooltip.buildNodeTooltipHtml({
        type: d.type || '',
        title: d.label || d.id,
        summary: summary,
        communityColor: communityColor,
        linkHtml: '<a class="tt-link" href="' + escapeHtml(detailUrl) + '">' + linkLabel + '</a>'
      });
    }
    return '';
  }

  function showTooltip(ev, d, nodeFill, communityLabelMap) {
    if (!tooltip) return;
    tooltip.innerHTML = tooltipHtml(d, nodeFill, communityLabelMap);
    tooltip.setAttribute('aria-hidden', 'false');
    tooltip.style.width = '';
    tooltip.style.transform = '';
    if (isMobile) {
      tooltip.classList.add('tt-pinned');
      tooltip.style.left = '';
      tooltip.style.top = '';
      tooltip.style.right = '20px';
      tooltip.style.bottom = '20px';
      pinnedNode = d;
      tooltip.classList.remove('hidden');
    } else {
      tooltip.classList.remove('tt-pinned');
      tooltip.style.right = '';
      tooltip.style.bottom = '';
      moveTooltip(ev);
      tooltip.classList.remove('hidden');
    }
  }

  function moveTooltip(ev) {
    if (!tooltip || tooltip.classList.contains('hidden')) return;
    var x = ev.clientX + 14;
    var y = ev.clientY - 10;
    var tw = tooltip.offsetWidth;
    var th = tooltip.offsetHeight;
    tooltip.style.left = (x + tw > window.innerWidth - 20 ? x - tw - 28 : x) + 'px';
    tooltip.style.top = (y + th > window.innerHeight - 20 ? y - th : y) + 'px';
    tooltip.style.transform = '';
  }

  function hideTooltip() {
    if (!tooltip) return;
    tooltip.classList.add('hidden');
    tooltip.setAttribute('aria-hidden', 'true');
  }

  if (tooltip) {
    tooltip.addEventListener('click', function(ev) {
      var link = ev.target.closest && ev.target.closest('.tt-link');
      if (!link) return;
      var href = link.getAttribute('href');
      if (!href) return;
      window.location.href = href;
      setTimeout(function() {
        pinnedNode = null;
        hideTooltip();
      }, 100);
    });
  }

  var TABLEAU10 = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac'];

  fetch('exports/link-graph.json').then(function(r){ return r.json(); }).then(function(gd) {
    var palette = (typeof d3 !== 'undefined' && d3.schemeTableau10) ? d3.schemeTableau10 : TABLEAU10;
    var communityFill = {};
    var communityLabelMap = {};
    var namedColorIdx = 0;
    (gd.communities || []).slice().sort(function (a, b) {
      if (a.id === 'community-other') return 1;
      if (b.id === 'community-other') return -1;
      return (b.size || 0) - (a.size || 0);
    }).forEach(function(c) {
      communityFill[c.id] = c.id === 'community-other'
        ? '#94a3b8'
        : palette[namedColorIdx++ % palette.length];
      communityLabelMap[c.id] = c.label || c.id;
    });
    function nodeFill(d) {
      if (d.community && communityFill[d.community]) return communityFill[d.community];
      return palette[palette.length - 1];
    }

    // 节点半径继承 graph view 的标尺（graph-node-size.js），度数基准为全图
    var degreeMap = window.RNGraphNodeSize.computeDegreeMap(gd.edges);
    var maxDegree = window.RNGraphNodeSize.maxDegreeOf(degreeMap);
    function nodeRadius(d) {
      return window.RNGraphNodeSize.radiusForDegree(d._degree, maxDegree);
    }

    var topIds = new Set(
      gd.nodes.slice().sort(function(a,b){ return (degreeMap[b.id]||0)-(degreeMap[a.id]||0); })
      .slice(0, PREVIEW_TOP_N).map(function(n){ return n.id; })
    );

    var nodes = gd.nodes.filter(function(n){ return topIds.has(n.id); }).map(function(n){
      return {
        id: n.id,
        label: n.label || n.id,
        type: n.type || '',
        community: n.community || '',
        summary: n.summary || '',
        _degree: degreeMap[n.id] || 0
      };
    });
    var nodeIdSet = new Set(nodes.map(function(n){ return n.id; }));
    var edges = gd.edges.filter(function(e){
      return nodeIdSet.has(e.source) && nodeIdSet.has(e.target);
    }).map(function(e){ return {source:e.source, target:e.target}; });

    var W = miniWrap.clientWidth || 700, H = 480;
    miniSvg.setAttribute('viewBox','0 0 '+W+' '+H);

    var svg = d3.select(miniSvg);
    var g = svg.append('g');
    var lineLayer = g.append('g');
    var nodeLayer = g.append('g');

    var line;
    var label;

    function applyMiniGraphTheme() {
      var theme = miniGraphTheme();
      miniWrap.style.background = theme.background;
      if (statsEl) statsEl.style.color = theme.stats;
      if (expandEl) {
        expandEl.style.color = theme.link;
        expandEl.style.borderBottom = '1px solid ' + theme.linkBorder;
      }
      if (line) line.attr('stroke', theme.edge);
      if (label) label.attr('fill', theme.label);
      applyMode3dTheme();
    }

    var zoom = d3.zoom().scaleExtent([0.3,4]).on('zoom',function(ev){ g.attr('transform',ev.transform); });
    svg.call(zoom).on('dblclick.zoom',null);

    var sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(function(d){ return d.id; }).distance(60).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-200).distanceMax(300))
      .force('center', d3.forceCenter(W/2, H/2).strength(0.08))
      .force('collision', d3.forceCollide().radius(function(d){ return nodeRadius(d) + 4; }).strength(0.6))
      .alphaDecay(0.03);

    line = lineLayer.selectAll('line').data(edges).join('line')
      .attr('stroke-width',1);

    var nodeG = nodeLayer.selectAll('g').data(nodes).join('g')
      .attr('class', 'mini-graph-node')
      .style('cursor','pointer')
      .on('click', function(ev, d) {
        if (isMobile) {
          ev.stopPropagation();
          if (pinnedNode === d) {
            pinnedNode = null;
            hideTooltip();
          } else {
            showTooltip(ev, d, nodeFill, communityLabelMap);
          }
          return;
        }
        window.location.href = 'graph.html?focus=' + encodeURIComponent(d.id);
      })
      .on('mouseenter', function(ev, d) {
        if (isMobile) return;
        d3.select(this).select('circle').attr('fill-opacity', 1).attr('r', nodeRadius(d) * 1.3);
        showTooltip(ev, d, nodeFill, communityLabelMap);
      })
      .on('mousemove', function(ev) {
        if (isMobile && pinnedNode) return;
        if (!isMobile || !pinnedNode) moveTooltip(ev);
      })
      .on('mouseleave', function(ev, d) {
        if (isMobile) return;
        d3.select(this).select('circle').attr('fill-opacity', 0.9).attr('r', nodeRadius(d));
        if (!isMobile || !pinnedNode) hideTooltip();
      });

    nodeG.append('circle')
      .attr('r', nodeRadius)
      .attr('fill', function(d){ return nodeFill(d); })
      .attr('fill-opacity', 0.9);

    label = nodeG.append('text')
      .text(function(d){ return d.label.length>12 ? d.label.slice(0,12)+'…' : d.label; })
      .attr('dy', function(d){ return nodeRadius(d)+11; })
      .attr('text-anchor','middle')
      .attr('font-size','10px')
      .attr('pointer-events','none');

    applyMiniGraphTheme();

    var observer = new MutationObserver(function() {
      applyMiniGraphTheme();
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    sim.on('tick', function() {
      line
        .attr('x1',function(d){ return d.source.x; }).attr('y1',function(d){ return d.source.y; })
        .attr('x2',function(d){ return d.target.x; }).attr('y2',function(d){ return d.target.y; });
      nodeG.attr('transform', function(d){ return 'translate('+d.x+','+d.y+')'; });
    });

    sim.on('end', function() {
      var allN = nodes.filter(function(n){ return n.x!=null; });
      if (!allN.length) return;
      var xs=allN.map(function(n){return n.x;}), ys=allN.map(function(n){return n.y;});
      var x0=Math.min.apply(null,xs), x1=Math.max.apply(null,xs);
      var y0=Math.min.apply(null,ys), y1=Math.max.apply(null,ys);
      var pad=40, cx=(x0+x1)/2, cy=(y0+y1)/2;
      var scale=Math.min(3, Math.max(0.3, Math.min(W/(x1-x0+pad), H/(y1-y0+pad))));
      svg.transition().duration(600).call(zoom.transform,
        d3.zoomIdentity.translate(W/2-scale*cx, H/2-scale*cy).scale(scale));
    });


    if (window.RNGraphTooltip) {
      var tooltipApi = {
        isMobile: isMobile,
        getPinned: function() { return pinnedNode; },
        clearPin: function() { pinnedNode = null; },
        hide: hideTooltip
      };
      window.RNGraphTooltip.bindBlankDismiss(miniSvg, tooltipApi, {
        nodeSelector: '.mini-graph-node',
        tooltipEl: tooltip
      });
      window.RNGraphTooltip.bindOutsideDismiss(miniSvg, tooltipApi, {
        tooltipEl: tooltip,
        dismissRootEl: document.querySelector('main')
      });
    }

    // ── 2D / 3D 预览切换：3D 库（vendor/3d-force-graph.min.js，1.3MB）懒加载，
    //    仅在首次切到 3D 时注入；实例常驻，来回切换只 pause/resume 动画 ──
    var btn2d = document.getElementById('miniGraphMode2d');
    var btn3d = document.getElementById('miniGraphMode3d');
    var wrap3d = document.getElementById('mini-graph-3d');
    var graph3d = null;
    var loading3d = false;
    var lastPointer3d = { clientX: 0, clientY: 0 };

    function trackPointer3d(ev) {
      lastPointer3d.clientX = ev.clientX;
      lastPointer3d.clientY = ev.clientY;
    }

    function showTooltip3d(d) {
      showTooltip(lastPointer3d, d, nodeFill, communityLabelMap);
    }

    function bindMiniGraph3dTooltipDismiss() {
      if (!window.RNGraphTooltip || !wrap3d) return;
      var tooltipApi = {
        isMobile: isMobile,
        getPinned: function() { return pinnedNode; },
        clearPin: function() { pinnedNode = null; },
        hide: hideTooltip
      };
      window.RNGraphTooltip.bindBlankDismiss(wrap3d, tooltipApi, {
        tooltipEl: tooltip
      });
      window.RNGraphTooltip.bindOutsideDismiss(wrap3d, tooltipApi, {
        tooltipEl: tooltip,
        dismissRootEl: document.querySelector('main')
      });
    }

    function applyMode3dTheme() {
      if (!graph3d) return;
      var theme = miniGraphTheme();
      graph3d.backgroundColor(theme.background);
      graph3d.linkColor(function () { return miniGraphTheme().edge3d; });
    }

    function setModeButtons(mode) {
      var in3d = mode === '3d';
      if (btn2d) {
        btn2d.classList.toggle('is-active', !in3d);
        btn2d.setAttribute('aria-pressed', in3d ? 'false' : 'true');
      }
      if (btn3d) {
        btn3d.classList.toggle('is-active', in3d);
        btn3d.setAttribute('aria-pressed', in3d ? 'true' : 'false');
      }
    }

    function init3d() {
      var theme = miniGraphTheme();
      // 独立节点/边副本：2D d3 力模拟已把 edges 的端点替换为节点对象引用，
      // 且节点带 x/y/vx/vy 等模拟状态，直接共享会与 3D 力模拟争用
      var nodes3d = nodes.map(function (n) {
        return {
          id: n.id,
          label: n.label,
          type: n.type,
          community: n.community,
          summary: n.summary || '',
          _degree: n._degree
        };
      });
      var links3d = edges.map(function (e) {
        return {
          source: typeof e.source === 'object' ? e.source.id : e.source,
          target: typeof e.target === 'object' ? e.target.id : e.target
        };
      });
      // controlType: orbit —— trackball 不支持 autoRotate（设了会静默无效）
      graph3d = window.ForceGraph3D({ controlType: 'orbit' })(wrap3d)
        .width(miniWrap.clientWidth || 700)
        .height(miniWrap.clientHeight || 480)
        .backgroundColor(theme.background)
        .showNavInfo(false)
        // 40 节点几秒即稳定；默认 cooldownTime 15s 会让 engineStop 的取景来得太晚
        .cooldownTime(4000)
        .nodeColor(function (d) { return nodeFill(d); })
        .nodeVal(function (d) { return Math.max(1, d._degree); })
        .linkColor(function () { return miniGraphTheme().edge3d; })
        .linkOpacity(0.35)
        .onNodeHover(function (node) {
          if (isMobile) return;
          if (node) showTooltip3d(node);
          else hideTooltip();
        })
        .onNodeClick(function (d) {
          if (isMobile) {
            if (pinnedNode === d) {
              pinnedNode = null;
              hideTooltip();
            } else {
              showTooltip3d(d);
            }
            return;
          }
          window.location.href = 'graph.html?focus=' + encodeURIComponent(d.id);
        })
        .graphData({ nodes: nodes3d, links: links3d });
      wrap3d.addEventListener('mousemove', trackPointer3d);
      bindMiniGraph3dTooltipDismiss();
      // 力模拟稳定后一次性取景填满容器（与 2D sim.on('end') 的自动 fit 对齐）；
      // 只做首次，避免用户拖拽节点触发的后续 engineStop 抢走相机
      var fitted3d = false;
      graph3d.onEngineStop(function () {
        if (fitted3d) return;
        fitted3d = true;
        graph3d.zoomToFit(600, 40);
      });
      var controls3d = graph3d.controls();
      if (controls3d) {
        controls3d.autoRotate = true;
        controls3d.autoRotateSpeed = 0.9;
      }
    }

    function show3d() {
      hideTooltip();
      pinnedNode = null;
      miniSvg.style.display = 'none';
      wrap3d.hidden = false;
      setModeButtons('3d');
      if (graph3d) {
        graph3d.resumeAnimation();
        return;
      }
      if (typeof window.ForceGraph3D === 'function') {
        init3d();
        return;
      }
      if (loading3d) return;
      loading3d = true;
      wrap3d.innerHTML = '<div class="mini-graph-3d-hint">正在加载 3D 组件…</div>';
      var script = document.createElement('script');
      script.src = 'vendor/3d-force-graph.min.js';
      script.onload = function () {
        loading3d = false;
        wrap3d.innerHTML = '';
        if (!wrap3d.hidden) init3d();
      };
      script.onerror = function () {
        loading3d = false;
        wrap3d.innerHTML = '<div class="mini-graph-3d-hint">3D 组件加载失败，请刷新重试</div>';
      };
      document.head.appendChild(script);
    }

    function show2d() {
      hideTooltip();
      pinnedNode = null;
      wrap3d.hidden = true;
      miniSvg.style.display = '';
      setModeButtons('2d');
      if (graph3d) graph3d.pauseAnimation();
    }

    if (btn2d && btn3d && wrap3d) {
      btn3d.addEventListener('click', show3d);
      btn2d.addEventListener('click', show2d);
      // 视口尺寸变化（手机横竖屏 / 窗口缩放）时同步 3D 画布尺寸
      window.addEventListener('resize', function () {
        if (!graph3d) return;
        graph3d
          .width(miniWrap.clientWidth || 700)
          .height(miniWrap.clientHeight || 480);
      });
    }

    statsEl.textContent = '预览：全站连接度 Top-' + PREVIEW_TOP_N + ' 枢纽节点';
  }).catch(function(){});
})();
