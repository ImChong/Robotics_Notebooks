(function () {
  'use strict';

  /**
   * 知识图谱 3D 视图封装（基于 3d-force-graph / Three.js）。
   * 由 graph.html 在切换到 3D 模式时初始化；与 2D 力模拟共享 nodes 数组。
   */
  function isAvailable() {
    return typeof window.ForceGraph3D === 'function';
  }

  function edgeEndpointId(endpoint) {
    return typeof endpoint === 'object' ? endpoint.id : endpoint;
  }

  function create(opts) {
    if (!isAvailable()) return null;
    opts = opts || {};

    var container = opts.container;
    var nodes = opts.nodes || [];
    var edges = opts.edges || [];
    var getNodeColor = opts.getNodeColor || function () { return '#64748b'; };
    var getNodeRadius = opts.getNodeRadius || function () { return 6; };
    var getLinkColor = opts.getLinkColor || function () { return 'rgba(255,255,255,0.07)'; };
    var getLinkWidth = opts.getLinkWidth || function () { return 1; };
    var isDark = opts.isDark || function () { return true; };
    var onNodeClick = opts.onNodeClick;
    var onNodeHover = opts.onNodeHover;
    var getVisibleNodeIds = opts.getVisibleNodeIds || function () { return new Set(); };
    var hasActiveFilter = opts.hasActiveFilter || function () { return false; };
    var edgeHighlightsWithNode = opts.edgeHighlightsWithNode;
    var getChargeStrength = opts.getChargeStrength || function () { return -800; };
    var getMagneticConfig = opts.getMagneticConfig || function () { return null; };

    var sidebarNodeId = null;
    var sidebarDirect = new Set();
    var sidebarSecondary = new Set();
    var hoverNodeId = null;
    var nodeById = new Map();

    function rebuildNodeIndex() {
      nodeById = new Map(nodes.map(function (n) { return [n.id, n]; }));
    }
    rebuildNodeIndex();

    function backgroundColor() {
      return isDark() ? '#0d1117' : '#eef2f7';
    }

    function buildLinks() {
      return edges
        .map(function (e) {
          var sId = edgeEndpointId(e.source);
          var tId = edgeEndpointId(e.target);
          var source = nodeById.get(sId);
          var target = nodeById.get(tId);
          if (!source || !target) return null;
          return { source: source, target: target, _ref: e };
        })
        .filter(Boolean);
    }

    function nodeOpacityFor(d) {
      var visible = getVisibleNodeIds();
      var filtered = hasActiveFilter();
      var ok = visible.has(d.id);

      if (sidebarNodeId) {
        if (d.id === sidebarNodeId || sidebarDirect.has(d.id)) return 1;
        if (sidebarSecondary.has(d.id)) return 0.4;
        return 0.05;
      }

      if (hoverNodeId) {
        if (!filtered) {
          return hoverNodeId === d.id || isNeighborOf(hoverNodeId, d.id) ? 1 : 0.15;
        }
        if (!ok) return 0.08;
        return hoverNodeId === d.id || isNeighborOf(hoverNodeId, d.id) ? 1 : 0.15;
      }

      if (!filtered) return 1;
      return ok ? 1 : 0.08;
    }

    function isNeighborOf(nodeId, otherId) {
      for (var i = 0; i < edges.length; i++) {
        var e = edges[i];
        var s = edgeEndpointId(e.source);
        var t = edgeEndpointId(e.target);
        if (s === nodeId && t === otherId) return true;
        if (t === nodeId && s === otherId) return true;
      }
      return false;
    }

    function linkOpacityFor(l) {
      var s = edgeEndpointId(l.source);
      var t = edgeEndpointId(l.target);
      var visible = getVisibleNodeIds();
      var filtered = hasActiveFilter();

      if (sidebarNodeId && edgeHighlightsWithNode) {
        var ref = l._ref;
        if (!ref) return 0.18;
        if (!filtered) {
          return edgeHighlightsWithNode(ref, sidebarNodeId) ? 0.85 : 0.18;
        }
        if (!visible.has(s) || !visible.has(t)) return 0.18;
        return edgeHighlightsWithNode(ref, sidebarNodeId) ? 0.85 : 0.18;
      }

      if (hoverNodeId) {
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode) {
          return edgeHighlightsWithNode(eRef, hoverNodeId) ? 0.7 : 0.3;
        }
      }

      if (!filtered) return 0.55;
      if (!visible.has(s) || !visible.has(t)) return 0.12;
      return 0.55;
    }

    function linkColorFor(l) {
      if (sidebarNodeId) {
        var ref = l._ref;
        if (ref && edgeHighlightsWithNode && edgeHighlightsWithNode(ref, sidebarNodeId)) {
          var node = nodeById.get(sidebarNodeId);
          return node ? getNodeColor(node) : getLinkColor();
        }
      }
      if (hoverNodeId) {
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode && edgeHighlightsWithNode(eRef, hoverNodeId)) {
          var hNode = nodeById.get(hoverNodeId);
          return hNode ? getNodeColor(hNode) : getLinkColor();
        }
      }
      return getLinkColor();
    }

    function linkWidthFor(l) {
      var base = getLinkWidth();
      if (sidebarNodeId) {
        var ref = l._ref;
        if (ref && edgeHighlightsWithNode && edgeHighlightsWithNode(ref, sidebarNodeId)) {
          return base * 1.5;
        }
      }
      if (hoverNodeId) {
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode && edgeHighlightsWithNode(eRef, hoverNodeId)) {
          return base * 1.8;
        }
      }
      return base;
    }

    nodes.forEach(function (n) {
      if (n.z == null) n.z = (Math.random() - 0.5) * 40;
    });

    var graph = window.ForceGraph3D()(container)
      .graphData({ nodes: nodes, links: buildLinks() })
      .backgroundColor(backgroundColor())
      .showNavInfo(false)
      .nodeId('id')
      .nodeLabel(function (d) {
        return (d._info && d._info.title) || d.label || d.id;
      })
      .nodeColor(function (d) { return getNodeColor(d); })
      .nodeVal(function (d) {
        var r = getNodeRadius(d);
        return Math.max(0.5, (r * r) / 16);
      })
      .nodeOpacity(function (d) { return nodeOpacityFor(d); })
      .linkColor(function (l) { return linkColorFor(l); })
      .linkWidth(function (l) { return linkWidthFor(l); })
      .linkOpacity(function (l) { return linkOpacityFor(l); })
      .linkDirectionalParticles(0)
      .enableNodeDrag(true)
      .onNodeClick(function (node, ev) {
        if (onNodeClick) onNodeClick(node, ev);
      })
      .onNodeHover(function (node, ev) {
        hoverNodeId = node ? node.id : null;
        refreshAppearance();
        if (onNodeHover) onNodeHover(node, ev);
      })
      .onBackgroundClick(function () {
        hoverNodeId = null;
        refreshAppearance();
        if (opts.onBackgroundClick) opts.onBackgroundClick();
      });

    var chargeForce = graph.d3Force('charge');
    if (chargeForce && chargeForce.strength) {
      chargeForce.strength(getChargeStrength());
    }
    var linkForce = graph.d3Force('link');
    if (linkForce) {
      if (linkForce.distance) linkForce.distance(80);
      if (linkForce.strength) linkForce.strength(0.35);
    }

    function applyMagneticForces() {
      var magneticConfig = getMagneticConfig();
      if (!magneticConfig || !magneticConfig.enabled) {
        graph.d3Force('x', null);
        graph.d3Force('y', null);
        graph.d3Force('z', null);
        return;
      }

      var centers = magneticConfig.centers || {};
      var getKey = magneticConfig.getKey || function () { return ''; };
      var forceZ = window.d3 && window.d3.forceZ;

      graph.d3Force('x', window.d3.forceX(function (d) {
        var key = getKey(d);
        return centers[key] ? centers[key].x : 0;
      }).strength(0.18));
      graph.d3Force('y', window.d3.forceY(function (d) {
        var key = getKey(d);
        return centers[key] ? centers[key].y : 0;
      }).strength(0.18));
      if (forceZ) {
        graph.d3Force('z', forceZ(function (d) {
          var key = getKey(d);
          return centers[key] ? centers[key].z : 0;
        }).strength(0.18));
      } else {
        graph.d3Force('z', null);
      }
    }

    function refreshAppearance() {
      graph
        .nodeColor(function (d) { return getNodeColor(d); })
        .nodeOpacity(function (d) { return nodeOpacityFor(d); })
        .linkColor(function (l) { return linkColorFor(l); })
        .linkWidth(function (l) { return linkWidthFor(l); })
        .linkOpacity(function (l) { return linkOpacityFor(l); });
    }

    function randomizePositions() {
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        n.fx = null;
        n.fy = null;
        n.fz = null;
        n.vx = 0;
        n.vy = 0;
        n.vz = 0;
        n.x = (Math.random() - 0.5) * 160;
        n.y = (Math.random() - 0.5) * 160;
        n.z = (Math.random() - 0.5) * 160;
      }
    }

    return {
      show: function () {
        container.hidden = false;
        graph.width(container.clientWidth);
        graph.height(container.clientHeight);
        graph.backgroundColor(backgroundColor());
        refreshAppearance();
        graph.resumeAnimation();
      },

      hide: function () {
        container.hidden = true;
        graph.pauseAnimation();
      },

      resize: function () {
        if (container.hidden) return;
        graph.width(container.clientWidth);
        graph.height(container.clientHeight);
      },

      syncFilters: function () {
        refreshAppearance();
      },

      refreshColors: function () {
        graph.backgroundColor(backgroundColor());
        refreshAppearance();
      },

      updateForces: function () {
        var charge = graph.d3Force('charge');
        if (charge && charge.strength) charge.strength(getChargeStrength());
        applyMagneticForces();
        graph.d3ReheatSimulation();
      },

      restartSimulation: function () {
        randomizePositions();
        rebuildNodeIndex();
        refreshAppearance();
        graph.graphData({ nodes: nodes, links: buildLinks() });
        this.updateForces();
      },

      fitToScreen: function (ms) {
        var duration = ms == null ? 650 : ms;
        graph.zoomToFit(duration, 120);
      },

      fitToVisible: function (ms) {
        var visible = getVisibleNodeIds();
        var list = nodes.filter(function (n) { return visible.has(n.id); });
        if (!list.length) {
          this.fitToScreen(ms);
          return;
        }
        var duration = ms == null ? 650 : ms;
        graph.zoomToFit(duration, 140, function (n) { return visible.has(n.id); });
      },

      focusNode: function (node, ms) {
        if (!node || node.x == null) return;
        var duration = ms == null ? 700 : ms;
        var dist = 220;
        var z = node.z != null ? node.z : 0;
        graph.cameraPosition(
          { x: node.x, y: node.y, z: z + dist },
          node,
          duration
        );
      },

      applySidebarHighlight: function (d, direct, secondary) {
        sidebarNodeId = d.id;
        sidebarDirect = direct || new Set();
        sidebarSecondary = secondary || new Set();
        hoverNodeId = null;
        refreshAppearance();
      },

      clearSidebarHighlight: function () {
        sidebarNodeId = null;
        sidebarDirect = new Set();
        sidebarSecondary = new Set();
        refreshAppearance();
      },

      destroy: function () {
        if (graph._destructor) graph._destructor();
      },
    };
  }

  /** 3D 磁吸模式：在球面上均匀分布聚类中心 */
  function buildMagneticCenters3D(categories) {
    var centers = {};
    var count = categories.length || 1;
    var radius = 120;
    var golden = Math.PI * (3 - Math.sqrt(5));
    categories.forEach(function (cat, i) {
      var y = 1 - (i / Math.max(count - 1, 1)) * 2;
      var r = Math.sqrt(Math.max(0, 1 - y * y));
      var theta = golden * i;
      centers[cat] = {
        x: radius * Math.cos(theta) * r,
        y: radius * y,
        z: radius * Math.sin(theta) * r,
      };
    });
    return centers;
  }

  window.RNGraph3D = {
    isAvailable: isAvailable,
    create: create,
    buildMagneticCenters3D: buildMagneticCenters3D,
  };
})();
