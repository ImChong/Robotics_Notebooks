(function() {
  var miniWrap = document.getElementById('mini-graph-wrap');
  var miniSvg  = document.getElementById('mini-graph-svg');
  var statsEl  = document.getElementById('mini-graph-stats');
  var expandEl = document.getElementById('mini-graph-expand');
  if (!miniWrap || !miniSvg) return;

  function miniGraphTheme() {
    var dark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
      background: dark ? '#0d1117' : '#eef2f7',
      edge: dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.10)',
      label: dark ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.70)', // 提高对比度
      stats: dark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.62)',
      link: dark ? '#60a5fa' : '#2563eb', // 使用更标准的蓝色
      linkBorder: dark ? 'rgba(96,165,250,0.30)' : 'rgba(37,99,235,0.30)'
    };
  }

  var TYPE_COLOR = {
    concept:'#60a5fa', method:'#34d399', task:'#f472b6',
    entity:'#fbbf24', comparison:'#c084fc', query:'#94a3b8',
    formalization:'#fb923c', '':'#64748b'
  };

  fetch('exports/link-graph.json').then(function(r){ return r.json(); }).then(function(gd) {
    fetch('exports/graph-stats.json').then(function(r){ return r.json(); }).then(function(stats) {
      var totalNodes = gd.nodes.length, totalEdges = gd.edges.length;
      var degreeMap = {};
      gd.edges.forEach(function(e) {
        degreeMap[e.source] = (degreeMap[e.source]||0)+1;
        degreeMap[e.target] = (degreeMap[e.target]||0)+1;
      });

      // Top-40 by degree
      var topIds = new Set(
        gd.nodes.slice().sort(function(a,b){ return (degreeMap[b.id]||0)-(degreeMap[a.id]||0); })
        .slice(0,40).map(function(n){ return n.id; })
      );

      var nodes = gd.nodes.filter(function(n){ return topIds.has(n.id); }).map(function(n){
        return { id:n.id, label:n.label||n.id, type:n.type||'', _degree:degreeMap[n.id]||0 };
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
      }

      var zoom = d3.zoom().scaleExtent([0.3,4]).on('zoom',function(ev){ g.attr('transform',ev.transform); });
      svg.call(zoom).on('dblclick.zoom',null);

      var sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(function(d){ return d.id; }).distance(60).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-200).distanceMax(300))
        .force('center', d3.forceCenter(W/2, H/2).strength(0.08))
        .force('collision', d3.forceCollide().radius(12).strength(0.6))
        .alphaDecay(0.03);

      var line = lineLayer.selectAll('line').data(edges).join('line')
        .attr('stroke-width',1);

      var nodeG = nodeLayer.selectAll('g').data(nodes).join('g')
        .style('cursor','pointer')
        .on('click', function(ev, d) {
          window.location.href = 'graph.html?focus=' + encodeURIComponent(d.id);
        })
        .on('mouseenter', function(ev, d) {
          d3.select(this).select('circle').attr('fill-opacity', 1).attr('r', function(){
            return Math.max(5, Math.min(14, 3+Math.sqrt(d._degree)*2)) * 1.3;
          });
        })
        .on('mouseleave', function(ev, d) {
          d3.select(this).select('circle').attr('fill-opacity', 0.9).attr('r',
            Math.max(5, Math.min(14, 3+Math.sqrt(d._degree)*2)));
        });

      nodeG.append('circle')
        .attr('r', function(d){ return Math.max(5, Math.min(14, 3+Math.sqrt(d._degree)*2)); })
        .attr('fill', function(d){ return TYPE_COLOR[d.type]||TYPE_COLOR['']; })
        .attr('fill-opacity', 0.9);

      var label = nodeG.append('text')
        .text(function(d){ return d.label.length>12 ? d.label.slice(0,12)+'…' : d.label; })
        .attr('dy', function(d){ return Math.max(5, Math.min(14, 3+Math.sqrt(d._degree)*2))+11; })
        .attr('text-anchor','middle')
        .attr('font-size','10px') // 稍大一点以提高清晰度
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

      var orphans = (stats.orphan_nodes||[]).length;
      statsEl.textContent = totalNodes + ' 节点 · ' + totalEdges + ' 条边 · 孤儿 ' + orphans + ' 个 | 显示 Top-40';
    }).catch(function(){});
  }).catch(function(){});
})();
