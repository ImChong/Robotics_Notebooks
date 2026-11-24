(function() {
  var miniWrap = document.getElementById('mini-graph-wrap');
  var miniSvg  = document.getElementById('mini-graph-svg');
  var statsEl  = document.getElementById('mini-graph-stats');
  var expandEl = document.getElementById('mini-graph-expand');
  var tooltip  = document.getElementById('mini-graph-tooltip');
  if (!miniWrap || !miniSvg) return;

  var PREVIEW_TOP_N = 50;
  var isMobile = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
  var lastPointerWasTouch = false;
  document.addEventListener('pointerdown', function (ev) {
    lastPointerWasTouch = ev.pointerType === 'touch';
  }, true);
  document.addEventListener('touchstart', function () {
    lastPointerWasTouch = true;
  }, { capture: true, passive: true });
  var pinnedNode = null;
  var tooltipNode = null;

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
    var classic = document.documentElement.getAttribute('data-ui') === 'classic';
    if (classic) {
      return {
        background: dark ? '#191919' : '#f7f6f3',
        edge: dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.10)',
        label: dark ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.70)',
        stats: dark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.62)',
        link: dark ? '#5b9cf6' : '#2d74da',
        linkBorder: dark ? 'rgba(91,156,246,0.30)' : 'rgba(45,116,218,0.30)'
      };
    }
    return {
      background: dark ? '#111317' : '#ffffff',
      edge: dark ? 'rgba(239,239,241,0.09)' : 'rgba(17,18,20,0.11)',
      label: dark ? 'rgba(239,239,241,0.78)' : 'rgba(17,18,20,0.74)',
      stats: dark ? '#8f949e' : '#595b61',
      link: dark ? '#5ca4ff' : '#1947c2',
      linkBorder: dark ? 'rgba(92,164,255,0.35)' : 'rgba(25,71,194,0.35)'
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
        communityLabel: d.community ? communityLabelMap[d.community] : '',
        health: d,
        hasRepo: d.has_repo,
        linkHtml: '<a class="tt-link" href="' + escapeHtml(detailUrl) + '">' + linkLabel + '</a>'
      });
    }
    return '';
  }

  function applyTooltipPosition(ev) {
    if (!tooltip || !ev || !Number.isFinite(ev.clientX) || !Number.isFinite(ev.clientY)) return false;
    var x = ev.clientX + 14;
    var y = ev.clientY - 10;
    var tw = tooltip.offsetWidth;
    var th = tooltip.offsetHeight;
    tooltip.style.left = Math.max(12, (x + tw > window.innerWidth - 20 ? x - tw - 28 : x)) + 'px';
    tooltip.style.top = Math.max(12, (y + th > window.innerHeight - 20 ? y - th : y)) + 'px';
    tooltip.style.transform = '';
    return true;
  }

  function showTooltip(ev, d, nodeFill, communityLabelMap, pinned) {
    if (!tooltip || !d) return;
    var shouldPin = Boolean(pinned || isMobile);
    var dockCorner = isMobile || window.innerWidth <= 768 || !ev || !Number.isFinite(ev.clientX) || !Number.isFinite(ev.clientY);
    if (!shouldPin) {
      if (!ev || !Number.isFinite(ev.clientX) || !Number.isFinite(ev.clientY)) return;
      // 3D 地形每次鼠标微移都会回调 onNodeHover：同一节点仅更新坐标，不重绘 DOM / KaTeX
      if (tooltipNode === d && !tooltip.classList.contains('hidden') && !tooltip.classList.contains('tt-pinned')) {
        moveTooltip(ev);
        return;
      }
    }
    tooltipNode = d;
    tooltip.innerHTML = tooltipHtml(d, nodeFill, communityLabelMap);
    if (window.RNGraphTooltip && window.RNGraphTooltip.renderMath) {
      window.RNGraphTooltip.renderMath(tooltip);
    }
    tooltip.setAttribute('aria-hidden', 'false');
    tooltip.style.width = '';
    tooltip.style.transform = '';
    if (shouldPin) {
      tooltip.classList.add('tt-pinned');
      pinnedNode = d;
      if (dockCorner) {
        tooltip.style.left = '';
        tooltip.style.top = '';
        tooltip.style.right = '20px';
        tooltip.style.bottom = 'calc(84px + env(safe-area-inset-bottom, 0px))';
        tooltip.classList.remove('hidden');
      } else {
        tooltip.style.right = '';
        tooltip.style.bottom = '';
        if (applyTooltipPosition(ev)) tooltip.classList.remove('hidden');
      }
    } else {
      tooltip.classList.remove('tt-pinned');
      tooltip.style.right = '';
      tooltip.style.bottom = '';
      // 先完成定位再移除 hidden，避免在定位前闪现到其他位置
      if (applyTooltipPosition(ev)) tooltip.classList.remove('hidden');
    }
  }

  function moveTooltip(ev) {
    if (!tooltip || tooltip.classList.contains('hidden')) return;
    applyTooltipPosition(ev);
  }

  function hideTooltip() {
    tooltipNode = null;
    if (!tooltip) return;
    tooltip.classList.add('hidden');
    tooltip.classList.remove('tt-pinned');
    tooltip.style.right = '';
    tooltip.style.bottom = '';
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

  function fallbackFullGraphPromise() {
    if (window.RNLinkGraph) {
      if (typeof window.RNLinkGraph.ensureSummaries === 'function') {
        return window.RNLinkGraph.ensureSummaries();
      }
      if (typeof window.RNLinkGraph.ensure === 'function') {
        return window.RNLinkGraph.ensure();
      }
    }
    return fetch('exports/link-graph.json').then(function (r) { return r.json(); }).then(function (gd) {
      return window.RNExportFormats ? window.RNExportFormats.expandLinkGraph(gd) : gd;
    });
  }

  // 优先读取 14 KB 预裁剪首页预览子图 exports/mini-graph.json，避免首页首屏下载 708 KB 全图与 716 KB 全量摘要；
  // 预裁剪产物缺失或无 fetch（单元测试）时无缝回退共享加载器。
  var graphDataPromise = (typeof window.fetch === 'function')
    ? fetch('exports/mini-graph.json')
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .catch(fallbackFullGraphPromise)
    : ((window.RNLinkGraph && typeof window.RNLinkGraph.ensure === 'function')
        ? window.RNLinkGraph.ensure()
        : Promise.reject(new Error('No graph data source')));
  graphDataPromise.then(function(gd) {
    var palette = (window.RNGraphTooltip && window.RNGraphTooltip.getCommunityColors)
      ? window.RNGraphTooltip.getCommunityColors()
      : ((typeof d3 !== 'undefined' && d3.schemeTableau10) ? d3.schemeTableau10 : TABLEAU10);
    var communityFill = (window.RNGraphTooltip && typeof window.RNGraphTooltip.buildCommunityColorMap === 'function')
      ? window.RNGraphTooltip.buildCommunityColorMap(gd.communities || [])
      : {};
    var communityLabelMap = {};
    var hasColorMap = Object.keys(communityFill).length > 0;
    var namedColorIdx = 0;
    (gd.communities || []).slice().sort(function (a, b) {
      if (a.id === 'community-other') return 1;
      if (b.id === 'community-other') return -1;
      var sizeDiff = (b.size || 0) - (a.size || 0);
      if (sizeDiff !== 0) return sizeDiff;
      return String(a.label || a.id).localeCompare(String(b.label || b.id), 'zh-CN');
    }).forEach(function(c) {
      if (!hasColorMap) {
        communityFill[c.id] = c.id === 'community-other'
          ? '#94a3b8'
          : palette[namedColorIdx++ % palette.length];
      }
      communityLabelMap[c.id] = c.label || c.id;
    });
    function nodeFill(d) {
      if (d.community && communityFill[d.community]) return communityFill[d.community];
      return palette[palette.length - 1];
    }

    var isPrePruned = typeof gd.max_degree === 'number' && gd.nodes && gd.nodes.length <= PREVIEW_TOP_N &&
      (!gd.nodes.length || typeof gd.nodes[0]._degree === 'number');
    var degreeMap;
    var maxDegree;
    var nodes;
    var edges;
    function nodeRadius(d) {
      return window.RNGraphNodeSize.radiusForDegree(d._degree, maxDegree);
    }

    if (isPrePruned) {
      maxDegree = Math.max(1, gd.max_degree);
      nodes = gd.nodes.map(function (n) {
        return {
          id: n.id,
          detail_id: n.detail_id,
          label: n.label || n.id,
          type: n.type || '',
          community: n.community || '',
          summary: n.summary || '',
          health_score: n.health_score,
          health_gaps: n.health_gaps,
          has_repo: n.has_repo,
          added: n.added || '',
          _degree: n._degree || 0
        };
      });
      var prunedIds = new Set(nodes.map(function (n) { return n.id; }));
      edges = (gd.edges || []).filter(function (e) {
        return e.source !== e.target && prunedIds.has(e.source) && prunedIds.has(e.target);
      }).map(function (e) { return { source: e.source, target: e.target }; });
    } else {
      // 节点半径继承 graph view 的标尺（graph-node-size.js），度数基准为全图
      degreeMap = window.RNGraphNodeSize.computeDegreeMap(gd.edges);
      maxDegree = window.RNGraphNodeSize.maxDegreeOf(degreeMap);

      // 邻接表：预览取 Top-N 诱导子图后，可能出现「全局度数高但邻居都不在 Top-N」的孤立点
      var adj = {};
      (gd.edges || []).forEach(function (e) {
        if (e.source === e.target) return;
        if (!adj[e.source]) adj[e.source] = [];
        if (!adj[e.target]) adj[e.target] = [];
        adj[e.source].push(e.target);
        adj[e.target].push(e.source);
      });
      function hasNeighborInSet(id, idSet) {
        var neighbors = adj[id] || [];
        for (var i = 0; i < neighbors.length; i++) {
          if (idSet.has(neighbors[i])) return true;
        }
        return false;
      }

      var ranked = gd.nodes.slice().sort(function (a, b) {
        return (degreeMap[b.id] || 0) - (degreeMap[a.id] || 0);
      });
      var topIds = new Set(
        ranked.slice(0, PREVIEW_TOP_N).map(function (n) { return n.id; })
      );
      // 剔除诱导子图孤立点（迭代至稳定，避免级联），再按度数序回填可连通候选
      var dropped = true;
      while (dropped) {
        dropped = false;
        Array.from(topIds).forEach(function (id) {
          if (!hasNeighborInSet(id, topIds)) {
            topIds.delete(id);
            dropped = true;
          }
        });
      }
      for (var ri = 0; ri < ranked.length && topIds.size < PREVIEW_TOP_N; ri++) {
        var cand = ranked[ri].id;
        if (topIds.has(cand)) continue;
        if (hasNeighborInSet(cand, topIds)) topIds.add(cand);
      }

      nodes = gd.nodes.filter(function(n){ return topIds.has(n.id); }).map(function(n){
        return {
          id: n.id,
          detail_id: n.detail_id,
          label: n.label || n.id,
          type: n.type || '',
          community: n.community || '',
          summary: n.summary || '',
          health_score: n.health_score,
          health_gaps: n.health_gaps,
          has_repo: n.has_repo,
          added: n.added || '',
          _degree: degreeMap[n.id] || 0
        };
      });
      var nodeIdSet = new Set(nodes.map(function(n){ return n.id; }));
      edges = gd.edges.filter(function(e){
        return e.source !== e.target && nodeIdSet.has(e.source) && nodeIdSet.has(e.target);
      }).map(function(e){ return {source:e.source, target:e.target}; });
    }
    var nodeById = Object.create(null);
    nodes.forEach(function(n){ nodeById[n.id] = n; });
    var latestMiniAddedTs = nodes.reduce(function (mx, n) {
      var ts = n && n.added ? Date.parse(String(n.added)) : 0;
      return isFinite(ts) && ts > mx ? ts : mx;
    }, 0);

    function edgeEndId(end) {
      return (end && typeof end === 'object') ? end.id : end;
    }

    var previewAdj = Object.create(null);
    edges.forEach(function (e) {
      var s = edgeEndId(e.source);
      var t = edgeEndId(e.target);
      if (!previewAdj[s]) previewAdj[s] = new Set();
      if (!previewAdj[t]) previewAdj[t] = new Set();
      previewAdj[s].add(t);
      previewAdj[t].add(s);
    });

    var W = miniWrap.clientWidth || 500, H = miniWrap.clientHeight || 420;
    miniSvg.setAttribute('viewBox','0 0 '+W+' '+H);

    var svg = d3.select(miniSvg);
    var g = svg.append('g');
    var lineLayer = g.append('g');
    var nodeLayer = g.append('g');

    var line;
    var label;
    var nodeG;
    var previewMode = '2d';
    var perspCtrl = null;
    var hoveredNode2d = null;

    function apply2dHoverState(activeNode) {
      if (!line || !nodeG) return;
      var theme = miniGraphTheme();
      if (!activeNode) {
        line
          .classed('mini-edge-hover', false)
          .classed('mini-edge-hover-dim', false)
          .attr('stroke', theme.edge)
          .attr('stroke-width', 1)
          .attr('stroke-opacity', null);
        nodeG
          .classed('mini-node-hover', false)
          .classed('mini-node-hover-neighbor', false)
          .classed('mini-node-hover-dim', false);
        nodeG.select('circle')
          .attr('fill-opacity', 0.9)
          .attr('r', nodeRadius);
        return;
      }
      var activeId = activeNode.id;
      var activeColor = nodeFill(activeNode);
      var neighbors = previewAdj[activeId] || new Set();
      line
        .classed('mini-edge-hover', function (e) {
          return edgeEndId(e.source) === activeId || edgeEndId(e.target) === activeId;
        })
        .classed('mini-edge-hover-dim', function (e) {
          return edgeEndId(e.source) !== activeId && edgeEndId(e.target) !== activeId;
        })
        .attr('stroke', function (e) {
          return (edgeEndId(e.source) === activeId || edgeEndId(e.target) === activeId)
            ? activeColor : theme.edge;
        })
        .attr('stroke-width', function (e) {
          return (edgeEndId(e.source) === activeId || edgeEndId(e.target) === activeId) ? 2 : 1;
        })
        .attr('stroke-opacity', function (e) {
          return (edgeEndId(e.source) === activeId || edgeEndId(e.target) === activeId) ? 0.92 : null;
        });
      line.filter('.mini-edge-hover').raise();
      nodeG
        .classed('mini-node-hover', function (n) { return n.id === activeId; })
        .classed('mini-node-hover-neighbor', function (n) { return n.id !== activeId && neighbors.has(n.id); })
        .classed('mini-node-hover-dim', function (n) { return n.id !== activeId && !neighbors.has(n.id); });
      nodeG.select('circle')
        .attr('fill-opacity', function (n) {
          return (n.id === activeId || neighbors.has(n.id)) ? 1 : 0.9;
        })
        .attr('r', function (n) {
          if (n.id === activeId) return nodeRadius(n) * 1.3;
          if (neighbors.has(n.id)) return nodeRadius(n) * 1.08;
          return nodeRadius(n);
        });
    }

    function applyMiniGraphTheme() {
      var theme = miniGraphTheme();
      miniWrap.style.background = previewMode === '3d' ? '' : theme.background;
      if (statsEl) statsEl.style.color = theme.stats;
      if (expandEl) {
        expandEl.style.color = theme.link;
        expandEl.style.borderBottom = '1px solid ' + theme.linkBorder;
      }
      if (line) line.attr('stroke', theme.edge);
      if (label) label.attr('fill', theme.label);
      if (previewMode === '2d' && (hoveredNode2d || pinnedNode)) {
        apply2dHoverState(hoveredNode2d || pinnedNode);
      }
      if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.refreshTheme();
    }

    var zoom = d3.zoom().scaleExtent([0.3,4])
      // 桌面滚轮/拖拽 + 触屏单指平移与双指捏合缩放统一启用
      .filter(function(ev){ return (!ev.ctrlKey || ev.type === 'wheel') && !ev.button; })
      .on('start', function(){ svg.style('cursor', 'grabbing'); })
      .on('zoom', function(ev){ g.attr('transform', ev.transform); })
      .on('end', function(){ svg.style('cursor', 'grab'); });
    svg.style('cursor', 'grab').call(zoom).on('dblclick.zoom',null);

    var sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(function(d){ return d.id; }).distance(60).strength(0.32))
      .force('charge', d3.forceManyBody().strength(-200).distanceMax(300))
      .force('center', d3.forceCenter(W/2, H/2).strength(0.08))
      .force('collision', d3.forceCollide().radius(function(d){ return nodeRadius(d) + 5; }).strength(0.65).iterations(1))
      .force('x', d3.forceX(W/2).strength(0.06))
      .force('y', d3.forceY(H/2).strength(0.06))
      .velocityDecay(0.52)
      .alphaDecay(0.04)
      .alphaMin(0.006)
      .stop();

    var entranceActive = false;
    var entranceTimer = null;
    var entranceObserver = null;
    function finishEntranceImmediately() {
      if (!entranceActive) return;
      entranceActive = false;
      if (entranceObserver) {
        entranceObserver.disconnect();
        entranceObserver = null;
      }
      if (entranceTimer) {
        entranceTimer.stop();
        entranceTimer = null;
      }
      nodes.forEach(function (n) {
        if (n._fx != null) n.x = n._fx;
        if (n._fy != null) n.y = n._fy;
        n._fx = null;
        n._fy = null;
      });
      if (nodeG) nodeG.select('circle').attr('r', nodeRadius);
      if (label) label.attr('opacity', null);
      if (line) line.attr('stroke-opacity', null);
      renderPositions();
    }

    line = lineLayer.selectAll('line').data(edges).join('line')
      .attr('stroke-width',1);

    var nodeDrag2d = d3.drag()
      .filter(function(ev) {
        // 双指触控交给 d3.zoom 做捏合缩放，单指触控与鼠标左键可拖拽节点
        if (ev.touches && ev.touches.length > 1) return false;
        return !ev.button;
      })
      .on('start', function(ev, d) {
        // 仅在入场动画尚未完成时收尾，已落稳后绝不覆盖任何节点坐标；
        // 按下瞬间不重启力模拟（首次真正拖动 >3px 时再柔和加热），避免按下瞬间其他节点跳变
        finishEntranceImmediately();
        var srcEv = ev.sourceEvent;
        d._dragIsTouch = Boolean(srcEv && ((srcEv.type && srcEv.type.indexOf('touch') === 0) || srcEv.pointerType === 'touch'));
        d._dragStartX = ev.x;
        d._dragStartY = ev.y;
        d._dragMoved = false;
        d._dragReheated = false;
        d.fx = d.x;
        d.fy = d.y;
        d3.select(this).style('cursor', 'grabbing');
      })
      .on('drag', function(ev, d) {
        var thresh = d._dragIsTouch ? 7 : 3;
        if (Math.hypot(ev.x - d._dragStartX, ev.y - d._dragStartY) > thresh) {
          d._dragMoved = true;
          if (!pinnedNode) hideTooltip();
          if (!d._dragReheated) {
            d._dragReheated = true;
            sim.alphaTarget(0.18).restart();
          }
        }
        if (d._dragMoved) {
          d.fx = ev.x;
          d.fy = ev.y;
          d.x = ev.x;
          d.y = ev.y;
          renderPositions();
        }
      })
      .on('end', function(ev, d) {
        d3.select(this).style('cursor', 'grab');
        if (d._dragReheated && !ev.active) {
          if (sim.alpha() < 0.18) sim.alpha(0.18);
          sim.alphaTarget(0);
        }
        d._dragReheated = false;
        d.fx = null;
        d.fy = null;
      });

    nodeG = nodeLayer.selectAll('g').data(nodes).join('g')
      .attr('class', 'mini-graph-node')
      .style('cursor','grab')
      .call(nodeDrag2d)
      .on('click', function(ev, d) {
        if (ev.defaultPrevented || d._dragMoved) {
          d._dragMoved = false;
          return;
        }
        ev.stopPropagation();
        // 点击/轻触节点先固定显示漂浮卡片，点击卡片内链接再跳转详情页
        if (pinnedNode === d) {
          pinnedNode = null;
          apply2dHoverState(null);
          hideTooltip();
        } else {
          showTooltip(ev, d, nodeFill, communityLabelMap, true);
          apply2dHoverState(d);
        }
      })
      .on('mouseenter', function(ev, d) {
        if (isMobile || lastPointerWasTouch || pinnedNode) return;
        hoveredNode2d = d;
        apply2dHoverState(d);
        showTooltip(ev, d, nodeFill, communityLabelMap, false);
      })
      .on('mousemove', function(ev) {
        if (isMobile || lastPointerWasTouch || pinnedNode) return;
        moveTooltip(ev);
      })
      .on('mouseleave', function() {
        if (isMobile || lastPointerWasTouch) return;
        hoveredNode2d = null;
        if (pinnedNode) return;
        apply2dHoverState(null);
        hideTooltip();
      });

    svg.on('mouseleave', function() {
      if (isMobile || !hoveredNode2d) return;
      hoveredNode2d = null;
      apply2dHoverState(null);
      if (!pinnedNode) hideTooltip();
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

    // ── 搜索联动：首页搜索框输入时，背景图谱高亮相关节点、其余淡出 ──
    function tokenizeMini(q) {
      return String(q || '').toLowerCase().split(/[^a-z0-9一-鿿]+/i)
        .filter(function(t){ return t.length >= 2; });
    }
    var telemetryModeEl = document.getElementById('miniGraphTelemetryMode');
    var telemetryLinkageEl = document.getElementById('miniGraphTelemetryLinkage');
    var miniHighlightSet = null;
    function clearHighlight() {
      miniHighlightSet = null;
      nodeG.classed('mini-node-dim', false).classed('mini-node-hit', false);
      line.classed('mini-edge-dim', false);
      if (telemetryLinkageEl) telemetryLinkageEl.textContent = '实时搜索联动';
      if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.syncData(false);
    }
    function highlightNodes(idList, query) {
      var idSet = {};
      (idList || []).forEach(function(id){ idSet[id] = true; });
      var tokens = tokenizeMini(query);
      var hitSet = {};
      var hitCount = 0;
      nodes.forEach(function(n) {
        var hit = !!idSet[n.id];
        if (!hit && tokens.length) {
          var lab = String(n.label || '').toLowerCase();
          for (var i = 0; i < tokens.length; i++) {
            if (lab.indexOf(tokens[i]) >= 0) { hit = true; break; }
          }
        }
        if (hit) { hitSet[n.id] = true; hitCount++; }
      });
      // 预览图只含前 50 度数节点，命中可能为空；此时不淡出，避免整图变暗却无高亮的困惑
      if (!hitCount) { clearHighlight(); return; }
      miniHighlightSet = new Set(Object.keys(hitSet));
      nodeG.classed('mini-node-dim', function(d){ return !hitSet[d.id]; });
      nodeG.classed('mini-node-hit', function(d){ return !!hitSet[d.id]; });
      line.classed('mini-edge-dim', function(d){
        var s = (d.source && d.source.id) || d.source;
        var t = (d.target && d.target.id) || d.target;
        return !(hitSet[s] && hitSet[t]);
      });
      if (telemetryLinkageEl) telemetryLinkageEl.textContent = hitCount + ' 个节点命中';
      if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.syncData(false);
    }
    window.RNMiniGraph = {
      highlight: highlightNodes,
      clear: clearHighlight,
      getMode: function () { return previewMode; },
      getPerspectiveState: function () { return perspCtrl ? perspCtrl.getDebugState() : null; },
      getPerspectiveTargets: function (limit) { return perspCtrl && perspCtrl.getScreenTargets ? perspCtrl.getScreenTargets(limit) : []; },
      getZoomTransform: function () { return typeof d3 !== 'undefined' ? d3.zoomTransform(miniSvg) : null; },
    };
    var pending = window.__miniGraphPendingQuery;
    if (pending && (pending.query || (pending.ids && pending.ids.length))) {
      highlightNodes(pending.ids || [], pending.query || '');
    }

    applyMiniGraphTheme();

    var observer = new MutationObserver(function() {
      applyMiniGraphTheme();
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-ui']
    });

    function renderPositions() {
      line
        .attr('x1',function(d){ return d.source.x; }).attr('y1',function(d){ return d.source.y; })
        .attr('x2',function(d){ return d.target.x; }).attr('y2',function(d){ return d.target.y; });
      nodeG.attr('transform', function(d){ return 'translate('+d.x+','+d.y+')'; });
    }

    sim.on('tick', renderPositions);
    sim.on('end', function() {
      window.RNGraphNodeSize.resolveOverlaps2D(nodes, nodeRadius, { padding: 5, maxPasses: 24 });
      renderPositions();
    });

    // 布局离屏算到收敛（预热 + 0.7 开场，tick 数与原先实时模拟一致，终态相同）：
    // 每帧限时 8ms 分帧计算，避免一次性同步计算形成长任务；算完前隐藏画布，算完直接取景，
    // 用户看到的动态改由下方「入场」负责，不再在首屏外空跑。
    g.attr('visibility', 'hidden');
    sim.alpha(1);
    for (var wi = 0; wi < 12; wi++) sim.tick();
    nodes.forEach(function (n) { n.vx = 0; n.vy = 0; });
    sim.alpha(0.7);
    var layoutDone = false;
    var layoutTimer = d3.timer(function () {
      var budgetEnd = performance.now() + 8;
      while (sim.alpha() >= sim.alphaMin() && performance.now() < budgetEnd) sim.tick();
      if (sim.alpha() >= sim.alphaMin()) return;
      layoutDone = true;
      layoutTimer.stop();
      onLayoutReady();
      if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.syncData(true);
    });
    function finishLayoutNow() {
      if (layoutDone) return;
      layoutDone = true;
      layoutTimer.stop();
      while (sim.alpha() >= sim.alphaMin()) sim.tick();
      onLayoutReady();
    }

    function fit2dViewport() {
      if (!nodes.length) return { cx: W / 2, cy: H / 2 };
      var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      nodes.forEach(function (n) {
        var nx = n._fx != null ? n._fx : n.x;
        var ny = n._fy != null ? n._fy : n.y;
        var r = nodeRadius(n);
        if (nx - r < minX) minX = nx - r;
        if (nx + r > maxX) maxX = nx + r;
        if (ny - r < minY) minY = ny - r;
        if (ny + r + 16 > maxY) maxY = ny + r + 16;
      });
      var cx = (minX + maxX) / 2;
      var cy = (minY + maxY) / 2;
      var spanX = Math.max(1, maxX - minX);
      var spanY = Math.max(1, maxY - minY);
      var padX = 96;
      var topPad = 48;
      var botPad = 56;
      var availW = Math.max(120, W - padX);
      var availH = Math.max(120, H - topPad - botPad);
      var scale = Math.min(2.2, Math.max(0.3, Math.min(availW / spanX, availH / spanY)));
      var targetCy = topPad + availH / 2;
      svg.call(zoom.transform, d3.zoomIdentity.translate(W / 2 - scale * cx, targetCy - scale * cy).scale(scale));
      return { cx: cx, cy: cy };
    }

    function syncViewportSize() {
      var nextW = miniWrap.clientWidth;
      var nextH = miniWrap.clientHeight;
      if (!nextW || !nextH) return;
      if (Math.abs(nextW - W) > 2 || Math.abs(nextH - H) > 2) {
        W = nextW;
        H = nextH;
        miniSvg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        sim.force('center', d3.forceCenter(W / 2, H / 2).strength(0.08));
        sim.force('x', d3.forceX(W / 2).strength(0.06));
        sim.force('y', d3.forceY(H / 2).strength(0.06));
        if (layoutDone) fit2dViewport();
      }
      if (perspCtrl) perspCtrl.syncViewport();
    }

    function onLayoutReady() {
      window.RNGraphNodeSize.resolveOverlaps2D(nodes, nodeRadius, { padding: 5, maxPasses: 24 });
      var center = fit2dViewport();
      var cx = center.cx, cy = center.cy;
      g.attr('visibility', null);

      // ── 入场：区块进入视口时，节点自取景中心由内到外错开扩散到终态位置并长大，连线随端点拉出、渐显 ──
      var ENTRANCE_MS = 900;
      var ENTRANCE_STAGGER_MS = 380;
      var reduceMotion = false;
      try { reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (_e) { /* ignore */ }
      if (reduceMotion || typeof IntersectionObserver === 'undefined') {
        renderPositions();
        return;
      }
      var maxDist = 1;
      nodes.forEach(function (n) {
        n._fx = n.x;
        n._fy = n.y;
        n._dist = Math.hypot(n.x - cx, n.y - cy);
        if (n._dist > maxDist) maxDist = n._dist;
      });
      nodes.forEach(function (n) { n._delay = n._dist / maxDist * ENTRANCE_STAGGER_MS; });

      var setEntranceFrame = function (elapsed) {
        var done = true;
        nodes.forEach(function (n) {
          var t = Math.max(0, Math.min(1, (elapsed - n._delay) / ENTRANCE_MS));
          if (t < 1) done = false;
          n._e = d3.easeCubicOut(t);
          n.x = cx + (n._fx - cx) * n._e;
          n.y = cy + (n._fy - cy) * n._e;
        });
        renderPositions();
        // 圆点用 r 长大、标签只淡入不缩放：逐帧缩放文字会反复重新栅格化字形，实测掉到 ~20fps
        nodeG.select('circle').attr('r', function(d){ return nodeRadius(d) * d._e; });
        label.attr('opacity', function(d){ return Math.max(0, (d._e - 0.6) / 0.4); });
        line.attr('stroke-opacity', function(d){ return Math.min(d.source._e, d.target._e); });
        return done;
      };

      entranceActive = true;
      setEntranceFrame(0);
      entranceObserver = new IntersectionObserver(function (entries) {
        if (!entries.some(function (en) { return en.isIntersecting; })) return;
        if (entranceObserver) {
          entranceObserver.disconnect();
          entranceObserver = null;
        }
        entranceTimer = d3.timer(function (elapsed) {
          if (!setEntranceFrame(elapsed)) return;
          if (entranceTimer) {
            entranceTimer.stop();
            entranceTimer = null;
          }
          entranceActive = false;
          nodes.forEach(function (n) {
            n.x = n._fx;
            n.y = n._fy;
            n._fx = null;
            n._fy = null;
          });
          renderPositions();
          label.attr('opacity', null);
          line.attr('stroke-opacity', null);
          if (hoveredNode2d || pinnedNode) apply2dHoverState(hoveredNode2d || pinnedNode);
        });
      }, { threshold: 0.35 });
      entranceObserver.observe(miniWrap);
    }


    if (window.RNGraphTooltip) {
      var tooltipApi = {
        isMobile: true,
        getPinned: function() { return pinnedNode; },
        clearPin: function() {
          pinnedNode = null;
          apply2dHoverState(null);
          if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.syncData(false);
        },
        hide: hideTooltip
      };
      window.RNGraphTooltip.bindBlankDismiss(miniSvg, tooltipApi, {
        nodeSelector: '.mini-graph-node',
        tooltipEl: tooltip
      });
      // 用外层容器 miniWrap 做外部点击判定：2D SVG 与 3D 舞台同属其内，避免 3D 点击节点被 2D 的 outside 监听误关
      window.RNGraphTooltip.bindOutsideDismiss(miniWrap, tooltipApi, {
        tooltipEl: tooltip,
        dismissRootEl: document.querySelector('main')
      });
    }

    // ── 2D / 3D 连接度地形视图切换（复用 graph-perspective.js，不加载 1.3MB 3D 库）──
    var btn2d = document.getElementById('miniGraphMode2d');
    var btn3d = document.getElementById('miniGraphMode3d');
    var stageEl = document.getElementById('mini-graph-perspective-stage');

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
      if (expandEl) {
        expandEl.setAttribute('href', in3d ? 'graph.html?view=perspective' : 'graph.html');
      }
      if (telemetryModeEl) {
        telemetryModeEl.textContent = in3d ? '3D 地形' : '2D 力导图';
      }
    }

    var dirtyFrom3dDrag = false;
    perspCtrl = (window.RNGraphPerspective && typeof window.RNGraphPerspective.createController === 'function' && stageEl)
      ? window.RNGraphPerspective.createController({
          wrapEl: miniWrap,
          stageEl: stageEl,
          nodeCount: nodes.length,
          persist: false,
          usePreviewTerrainHeight: true,
          allowTouchOrbit: true,
          defaultDistance: 1.92,
          defaultParamsByTier: {
            full: { contours: 14, pitch: 34, yaw: -16 },
            lite: { contours: 12, pitch: 32, yaw: -12 },
            reduced: { contours: 12, pitch: 36, yaw: -12 },
          },
          getViewportSize: function () {
            return {
              width: miniWrap.clientWidth || W || 500,
              height: miniWrap.clientHeight || H || 420,
            };
          },
          getSearchMatchIds: function () { return miniHighlightSet; },
          isNodeNew: function (d) {
            if (!latestMiniAddedTs || !d || !d.added) return false;
            var ts = Date.parse(String(d.added));
            return isFinite(ts) && (latestMiniAddedTs - ts) <= 14 * 86400000;
          },
          getGraphData: function () {
            var terrainNodes = entranceActive
              ? nodes.map(function (n) {
                  if (n._fx != null && n._fy != null && (n.x !== n._fx || n.y !== n._fy)) {
                    return Object.assign({}, n, { x: n._fx, y: n._fy });
                  }
                  return n;
                })
              : nodes;
            return {
              nodes: terrainNodes,
              edges: edges,
              focusIds: miniHighlightSet,
              sidebarNode: pinnedNode,
              getNodeColor: nodeFill,
              getNodeAlpha: function (d) {
                return (!miniHighlightSet || miniHighlightSet.has(d.id)) ? 1 : 0.16;
              },
            };
          },
          isDark: function () {
            return document.documentElement.getAttribute('data-theme') !== 'light';
          },
          getBgColor: function () {
            return miniGraphTheme().background;
          },
          getLabelText: function (d) { return d.label || d.id; },
          isLabelsVisible: function () { return true; },
          isEdgesVisible: function () { return true; },
          onNodeHover: function (d, ev) {
            if (isMobile || lastPointerWasTouch || pinnedNode) return;
            if (d) showTooltip(ev, d, nodeFill, communityLabelMap, false);
            else hideTooltip();
          },
          onNodeClick: function (d, ev) {
            // 点击/轻触节点先固定显示漂浮卡片，点击卡片内链接再跳转详情页
            if (pinnedNode === d) {
              pinnedNode = null;
              hideTooltip();
            } else {
              showTooltip(ev, d, nodeFill, communityLabelMap, true);
            }
            if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.syncData(false);
          },
          onBackgroundClick: function () {
            if (pinnedNode) {
              pinnedNode = null;
              hideTooltip();
              if (perspCtrl && perspCtrl.isEnabled()) perspCtrl.syncData(false);
            }
          },
          onNodeDragStart: function () {
            pinnedNode = null;
            hideTooltip();
          },
          onNodeDrag: function (d, x, y) {
            dirtyFrom3dDrag = true;
            var orig = nodeById[d.id];
            if (orig) {
              orig.x = x;
              orig.y = y;
            }
            d.x = x;
            d.y = y;
          },
          onNodeDragEnd: function () {
            renderPositions();
          },
          onStateChange: function (st) {
            previewMode = (st && st.enabled) ? '3d' : '2d';
            setModeButtons(previewMode);
            applyMiniGraphTheme();
          },
        })
      : null;

    function show3d() {
      finishLayoutNow();
      finishEntranceImmediately();
      sim.stop();
      hideTooltip();
      pinnedNode = null;
      hoveredNode2d = null;
      apply2dHoverState(null);
      if (!perspCtrl || !perspCtrl.setEnabled(true, { silent: true })) {
        previewMode = '2d';
        setModeButtons('2d');
      }
    }

    function show2d() {
      hideTooltip();
      pinnedNode = null;
      if (perspCtrl) {
        perspCtrl.setEnabled(false);
      } else {
        previewMode = '2d';
        setModeButtons('2d');
        applyMiniGraphTheme();
      }
      nodes.forEach(function (n) {
        n.fx = null;
        n.fy = null;
        n.vx = 0;
        n.vy = 0;
      });
      window.RNGraphNodeSize.resolveOverlaps2D(nodes, nodeRadius, { padding: 5, maxPasses: dirtyFrom3dDrag ? 60 : 24 });
      dirtyFrom3dDrag = false;
      renderPositions();
    }

    if (btn2d && btn3d) {
      if (!perspCtrl || !perspCtrl.isSupported()) {
        btn3d.disabled = true;
        btn3d.setAttribute('aria-disabled', 'true');
      }
      btn3d.addEventListener('click', show3d);
      btn2d.addEventListener('click', show2d);
    }

    // 视口尺寸变化（手机横竖屏 / 窗口缩放 / 新旧版切换）时同步 2D SVG 与 3D 舞台
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(syncViewportSize).observe(miniWrap);
    }
    window.addEventListener('resize', syncViewportSize);
    window.addEventListener('rn:uichange', function () {
      applyMiniGraphTheme();
      window.requestAnimationFrame(syncViewportSize);
    });

    statsEl.textContent = '预览：全站连接度 Top-' + PREVIEW_TOP_N + ' 枢纽节点';
  }).catch(function(){});
})();
