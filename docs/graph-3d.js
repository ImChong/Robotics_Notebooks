(function () {
  'use strict';

  /**
   * 知识图谱 3D 视图（3d-force-graph）。
   * 使用独立节点副本，避免与 2D d3 力模拟争用节点对象。
   */
  function isAvailable() {
    return typeof window.ForceGraph3D === 'function';
  }

  function edgeEndpointId(endpoint) {
    return typeof endpoint === 'object' ? endpoint.id : endpoint;
  }

  /**
   * 页面加载的是 2D 版 d3（含 forceX/forceY，但无 forceZ），3d-force-graph 内部
   * 的 d3-force-3d 又不对外暴露。手写一个等价的 z 轴定向力供磁吸布局使用。
   */
  function createForceZ(getZ, strength) {
    var nodesRef = [];
    function force(alpha) {
      for (var i = 0; i < nodesRef.length; i++) {
        var n = nodesRef[i];
        if (n.z == null) continue;
        n.vz += (getZ(n) - n.z) * strength * alpha;
      }
    }
    force.initialize = function (ns) { nodesRef = ns; };
    return force;
  }

  function cloneNodeFor3D(source) {
    var copy = Object.assign({}, source);
    if (source._info) copy._info = Object.assign({}, source._info);
    copy.x = source.x != null ? source.x : (Math.random() - 0.5) * 80;
    copy.y = source.y != null ? source.y : (Math.random() - 0.5) * 80;
    copy.z = source.z != null ? source.z : (Math.random() - 0.5) * 80;
    copy.vx = 0;
    copy.vy = 0;
    copy.vz = 0;
    // 不要把 fx/fy/fz 设为 null：3d-force-graph 在拖拽结束时只有当「拖拽前该轴为
    // undefined」才解除钉固（源码判断 void 0 === 轴值）。设成 null 会被判定为
    // 「拖拽前已钉固」，从而保留落点、不回弹。删除属性（保持 undefined），
    // 自由节点松手后即可回弹到力学平衡位置（与 2D drag end 清空 fx/fy 行为一致）。
    delete copy.fx;
    delete copy.fy;
    delete copy.fz;
    return copy;
  }

  function measureContainerSize(container) {
    var w = container.clientWidth;
    var h = container.clientHeight;
    if (w > 0 && h > 0) return { width: w, height: h };
    var wrap = container.parentElement;
    if (wrap) {
      w = wrap.clientWidth;
      h = wrap.clientHeight;
    }
    return {
      width: Math.max(w || 0, 1200),
      height: Math.max(h || 0, 800),
    };
  }

  function captureBundledThree(graph) {
    if (!graph || !graph.scene) return null;
    var scene = graph.scene();
    if (!scene) return null;
    var captured = null;
    scene.traverse(function (obj) {
      if (!captured && obj.isMesh && obj.geometry && obj.material) {
        captured = {
          SphereGeometry: obj.geometry.constructor,
          MeshLambertMaterial: obj.material.constructor,
          Mesh: obj.constructor,
          AmbientLight: null,
          DirectionalLight: null,
        };
      }
      if (captured && obj.isLight) {
        if (obj.type === 'AmbientLight') captured.AmbientLight = obj.constructor;
        if (obj.type === 'DirectionalLight') captured.DirectionalLight = obj.constructor;
      }
    });
    if (!captured || !captured.SphereGeometry || !captured.Mesh) return null;
    return captured;
  }

  function ensureSceneLights(scene, T) {
    if (!scene || !T) return;
    var hasLight = false;
    scene.traverse(function (obj) { if (obj.isLight) hasLight = true; });
    if (hasLight) return;
    if (T.AmbientLight) scene.add(new T.AmbientLight(0xffffff, 0.92));
    if (T.DirectionalLight) {
      var dir = new T.DirectionalLight(0xffffff, 0.88);
      dir.position.set(90, 140, 110);
      scene.add(dir);
    }
  }

  function create(opts) {
    if (!isAvailable()) return null;
    opts = opts || {};

    var container = opts.container;
    var sourceNodes = opts.nodes || [];
    var edges = opts.edges || [];
    var getNodeColor = opts.getNodeColor || function () { return '#64748b'; };
    var getNodeRadius = opts.getNodeRadius || function () { return 6; };
    var getLinkColor = opts.getLinkColor || function () { return 'rgba(255,255,255,0.07)'; };
    var getLinkWidth = opts.getLinkWidth || function () { return 1; };
    var isDark = opts.isDark || function () { return true; };
    var onNodeClick = opts.onNodeClick;
    var onNodeHover = opts.onNodeHover;
    var onPointerMove = opts.onPointerMove;
    var getVisibleNodeIds = opts.getVisibleNodeIds || function () { return new Set(); };
    var hasActiveFilter = opts.hasActiveFilter || function () { return false; };
    var edgeHighlightsWithNode = opts.edgeHighlightsWithNode;
    var getChargeStrength = opts.getChargeStrength || function () { return -800; };
    var getMagneticConfig = opts.getMagneticConfig || function () { return null; };
    var resolveSourceNode = opts.resolveSourceNode || function (id) {
      return sourceNodes.find(function (n) { return n.id === id; });
    };

    var nodes3d = sourceNodes.map(cloneNodeFor3D);
    var sidebarNodeId = null;
    var sidebarDirect = new Set();
    var sidebarSecondary = new Set();
    var hoverNodeId = null;
    var nodeById = new Map();
    var lastPointer = { clientX: 0, clientY: 0 };
    var size = measureContainerSize(container);
    var bundledThree = null;
    var customMeshesInstalled = false;

    function rebuildNodeIndex() {
      nodeById = new Map(nodes3d.map(function (n) { return [n.id, n]; }));
    }
    rebuildNodeIndex();

    // 预建邻接表：edges 在视图生命周期内不变，构建一次即可。
    // 避免 isNeighborOf 在每次 hover 刷新时对每个节点都全量遍历边（O(N×E)）。
    var adjacency = new Map();
    edges.forEach(function (e) {
      var s = edgeEndpointId(e.source);
      var t = edgeEndpointId(e.target);
      if (!adjacency.has(s)) adjacency.set(s, new Set());
      if (!adjacency.has(t)) adjacency.set(t, new Set());
      adjacency.get(s).add(t);
      adjacency.get(t).add(s);
    });

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
        if (!filtered) return hoverNodeId === d.id || isNeighborOf(hoverNodeId, d.id) ? 1 : 0.15;
        if (!ok) return 0.08;
        return hoverNodeId === d.id || isNeighborOf(hoverNodeId, d.id) ? 1 : 0.15;
      }
      if (!filtered) return 1;
      return ok ? 1 : 0.08;
    }

    function isNeighborOf(nodeId, otherId) {
      var set = adjacency.get(nodeId);
      return set ? set.has(otherId) : false;
    }

    function linkOpacityFor(l) {
      var s = edgeEndpointId(l.source);
      var t = edgeEndpointId(l.target);
      var visible = getVisibleNodeIds();
      var filtered = hasActiveFilter();
      if (sidebarNodeId && edgeHighlightsWithNode) {
        var ref = l._ref;
        if (!ref) return 0.08;
        if (!filtered) return edgeHighlightsWithNode(ref, sidebarNodeId) ? 0.55 : 0.08;
        if (!visible.has(s) || !visible.has(t)) return 0.08;
        return edgeHighlightsWithNode(ref, sidebarNodeId) ? 0.55 : 0.08;
      }
      if (hoverNodeId) {
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode) return edgeHighlightsWithNode(eRef, hoverNodeId) ? 0.45 : 0.1;
      }
      if (!filtered) return 0.06;
      if (!visible.has(s) || !visible.has(t)) return 0.03;
      return 0.06;
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
      var base = getLinkWidth() * 0.3;
      if (sidebarNodeId) {
        var ref = l._ref;
        if (ref && edgeHighlightsWithNode && edgeHighlightsWithNode(ref, sidebarNodeId)) return base * 1.5;
      }
      if (hoverNodeId) {
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode && edgeHighlightsWithNode(eRef, hoverNodeId)) return base * 1.8;
      }
      return base;
    }

    function sphereRadiusFor(d) {
      var r = getNodeRadius(d);
      return Math.max(11, r * 1.65);
    }

    function nodeValFor(d) {
      // 自定义 mesh 安装前（最初约 700ms）由 3d-force-graph 默认球体渲染，
      // 其半径 = cbrt(nodeVal) × nodeRelSize。配合 nodeRelSize(1) 取 radius³，
      // 使默认球体半径恰为 sphereRadiusFor，避免进入 3D 瞬间“大球→骤缩”的跳变。
      var radius = sphereRadiusFor(d);
      return radius * radius * radius;
    }

    function createNodeMesh(d) {
      if (!bundledThree) return null;
      var radius = sphereRadiusFor(d);
      var opacity = nodeOpacityFor(d);
      var geometry = new bundledThree.SphereGeometry(radius, 18, 14);
      var MaterialCtor = bundledThree.MeshLambertMaterial;
      var material = new MaterialCtor({
        color: getNodeColor(d),
        transparent: opacity < 0.999,
        opacity: opacity,
      });
      var mesh = new bundledThree.Mesh(geometry, material);
      mesh.userData.nodeId = d.id;
      return mesh;
    }

    function updateNodeMeshes() {
      if (!graph || !bundledThree) return;
      var scene = graph.scene && graph.scene();
      if (!scene) return;
      scene.traverse(function (obj) {
        if (!obj.isMesh || !obj.userData || !obj.userData.nodeId || !obj.material) return;
        var d = nodeById.get(obj.userData.nodeId);
        if (!d) return;
        var opacity = nodeOpacityFor(d);
        obj.material.color.set(getNodeColor(d));
        obj.material.opacity = opacity;
        obj.material.transparent = opacity < 0.999;
        obj.visible = opacity > 0.02;
      });
    }

    function refreshAppearance() {
      if (!customMeshesInstalled) {
        graph
          .nodeColor(function (d) { return getNodeColor(d); })
          .nodeVal(nodeValFor)
          .nodeOpacity(function (d) { return nodeOpacityFor(d); });
      }
      graph
        .linkColor(function (l) { return linkColorFor(l); })
        .linkWidth(function (l) { return linkWidthFor(l); })
        .linkOpacity(function (l) { return linkOpacityFor(l); });
      updateNodeMeshes();
    }

    function installCustomNodeMeshes() {
      if (customMeshesInstalled || !graph) return false;
      bundledThree = captureBundledThree(graph);
      if (!bundledThree) return false;
      graph
        .nodeThreeObject(function (d) { return createNodeMesh(d); })
        .nodeThreeObjectExtend(false);
      graph.graphData({ nodes: nodes3d, links: buildLinks() });
      ensureSceneLights(graph.scene(), bundledThree);
      customMeshesInstalled = true;
      refreshAppearance();
      return true;
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
      graph.d3Force('x', window.d3.forceX(function (d) {
        var key = getKey(d);
        return centers[key] ? centers[key].x : 0;
      }).strength(0.18));
      graph.d3Force('y', window.d3.forceY(function (d) {
        var key = getKey(d);
        return centers[key] ? centers[key].y : 0;
      }).strength(0.18));
      graph.d3Force('z', createForceZ(function (d) {
        var key = getKey(d);
        return centers[key] ? centers[key].z : 0;
      }, 0.18));
    }

    function syncPositionsFromSource() {
      sourceNodes.forEach(function (src) {
        var n3 = nodeById.get(src.id);
        if (!n3) return;
        if (src.x != null) n3.x = src.x;
        if (src.y != null) n3.y = src.y;
      });
    }

    function syncViewport() {
      var next = measureContainerSize(container);
      graph.width(next.width);
      graph.height(next.height);
    }

    function onContainerPointerMove(ev) {
      lastPointer.clientX = ev.clientX;
      lastPointer.clientY = ev.clientY;
      if (onPointerMove) onPointerMove(ev);
    }
    container.addEventListener('mousemove', onContainerPointerMove);
    container.addEventListener('pointermove', onContainerPointerMove);

    var graph = window.ForceGraph3D()(container)
      .width(size.width)
      .height(size.height)
      .graphData({ nodes: nodes3d, links: buildLinks() })
      .backgroundColor(backgroundColor())
      .showNavInfo(false)
      .warmupTicks(40)
      .nodeId('id')
      .nodeRelSize(1)
      .nodeResolution(24)
      .nodeColor(function (d) { return getNodeColor(d); })
      .nodeVal(nodeValFor)
      .nodeOpacity(function (d) { return nodeOpacityFor(d); })
      .linkColor(function (l) { return linkColorFor(l); })
      .linkWidth(function (l) { return linkWidthFor(l); })
      .linkOpacity(function (l) { return linkOpacityFor(l); })
      .enableNodeDrag(true)
      .onNodeClick(function (node, ev) {
        var src = resolveSourceNode(node.id) || node;
        if (onNodeClick) onNodeClick(src, ev || lastPointer);
      })
      .onNodeHover(function (node) {
        hoverNodeId = node ? node.id : null;
        refreshAppearance();
        var src = node ? (resolveSourceNode(node.id) || node) : null;
        if (onNodeHover) onNodeHover(src, lastPointer);
      })
      .onBackgroundClick(function () {
        hoverNodeId = null;
        refreshAppearance();
        if (opts.onBackgroundClick) opts.onBackgroundClick();
      });

    var chargeForce = graph.d3Force('charge');
    if (chargeForce && chargeForce.strength) chargeForce.strength(getChargeStrength());

    return {
      show: function () {
        container.hidden = false;
        syncPositionsFromSource();
        syncViewport();
        graph.backgroundColor(backgroundColor());
        refreshAppearance();
        this.resumeSimulation();
        var self = this;
        window.requestAnimationFrame(function () { syncViewport(); });
        window.setTimeout(function () {
          if (!customMeshesInstalled) installCustomNodeMeshes();
          self.fitToScreen(800);
        }, 700);
        window.setTimeout(function () { self.fitToScreen(600); }, 2600);
      },

      hide: function () {
        this.pauseSimulation();
        container.hidden = true;
      },

      pauseSimulation: function () {
        if (!graph) return;
        if (typeof graph.d3AlphaTarget === 'function') graph.d3AlphaTarget(0);
      },

      resumeSimulation: function () {
        if (!graph) return;
        if (typeof graph.resumeAnimation === 'function') graph.resumeAnimation();
        if (typeof graph.d3AlphaTarget === 'function') graph.d3AlphaTarget(0.25);
      },

      resize: function () {
        if (container.hidden) return;
        syncViewport();
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
        nodes3d = sourceNodes.map(cloneNodeFor3D);
        rebuildNodeIndex();
        graph.graphData({ nodes: nodes3d, links: buildLinks() });
        this.updateForces();
      },

      fitToScreen: function (ms) {
        graph.zoomToFit(ms == null ? 650 : ms, 16);
      },

      fitToVisible: function (ms) {
        var visible = getVisibleNodeIds();
        graph.zoomToFit(ms == null ? 650 : ms, 64, function (n) { return visible.has(n.id); });
      },

      focusNode: function (node, ms) {
        var n3 = node && nodeById.get(node.id);
        if (!n3 || n3.x == null) return;
        var z = n3.z != null ? n3.z : 0;
        var dist = Math.max(180, sphereRadiusFor(n3) * 16);
        graph.cameraPosition(
          { x: n3.x, y: n3.y, z: z + dist },
          n3,
          ms == null ? 700 : ms
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
        container.removeEventListener('mousemove', onContainerPointerMove);
        container.removeEventListener('pointermove', onContainerPointerMove);
        if (graph._destructor) graph._destructor();
        container.replaceChildren();
      },
    };
  }

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
