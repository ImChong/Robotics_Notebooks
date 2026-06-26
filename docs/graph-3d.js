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

  function isSphereGeometryType(geometry) {
    if (!geometry || !geometry.type) return false;
    return /^Sphere(Buffer)?Geometry$/i.test(geometry.type);
  }

  function captureBundledThreeFromMesh(obj) {
    return {
      SphereGeometry: obj.geometry.constructor,
      MeshLambertMaterial: obj.material.constructor,
      Mesh: obj.constructor,
      AmbientLight: null,
      DirectionalLight: null,
    };
  }

  function validateBundledThree(T) {
    if (!T || !T.SphereGeometry || !T.MeshLambertMaterial || !T.Mesh) return false;
    try {
      var probe = new T.SphereGeometry(1, 8, 6);
      if (probe && probe.dispose) probe.dispose();
      return true;
    } catch (_e) {
      return false;
    }
  }

  /**
   * 只从 three-forcegraph 的默认节点球体抓取 THREE 构造器。
   * 若误抓连线 TubeGeometry，后续 new SphereGeometry(...) 会失败，导致 3D 节点全空。
   */
  function captureBundledThree(graph) {
    if (!graph || !graph.scene) return null;
    var scene = graph.scene();
    if (!scene) return null;
    var captured = null;
    scene.traverse(function (obj) {
      if (!obj.isMesh || !obj.geometry || !obj.material) return;
      if (obj.__graphObjType !== 'node') return;
      if (!isSphereGeometryType(obj.geometry)) return;
      if (!captured || obj.__graphDefaultObj) {
        captured = captureBundledThreeFromMesh(obj);
      }
    });
    if (!captured) return null;
    scene.traverse(function (obj) {
      if (!captured || !obj.isLight) return;
      if (obj.type === 'AmbientLight') captured.AmbientLight = obj.constructor;
      if (obj.type === 'DirectionalLight') captured.DirectionalLight = obj.constructor;
    });
    return validateBundledThree(captured) ? captured : null;
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
    var sharedSphereGeometry = null;
    var customMeshesInstalled = false;
    var meshInstallScheduled = false;

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
      // 自定义 mesh 安装前由 3d-force-graph 默认球体渲染；nodeResolution(8) 仅作短暂占位。
      var radius = sphereRadiusFor(d);
      return radius * radius * radius;
    }

    function ensureSharedSphereGeometry(T) {
      if (!sharedSphereGeometry) {
        // 单位球 + mesh.scale：1336 节点共享一份几何体，避免逐节点 new SphereGeometry 阻塞主线程。
        sharedSphereGeometry = new T.SphereGeometry(1, 12, 10);
      }
      return sharedSphereGeometry;
    }

    function createNodeMesh(d) {
      if (!bundledThree) return null;
      var radius = sphereRadiusFor(d);
      var opacity = nodeOpacityFor(d);
      var geometry = ensureSharedSphereGeometry(bundledThree);
      var MaterialCtor = bundledThree.MeshLambertMaterial;
      var material = new MaterialCtor({
        color: getNodeColor(d),
        transparent: opacity < 0.999,
        opacity: opacity,
      });
      var mesh = new bundledThree.Mesh(geometry, material);
      mesh.scale.set(radius, radius, radius);
      mesh.userData.nodeId = d.id;
      mesh.userData.baseRadius = Math.max(11, getNodeRadius(d) * 1.65);
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
        var radius = sphereRadiusFor(d);
        var opacity = nodeOpacityFor(d);
        obj.scale.set(radius, radius, radius);
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
      ensureSharedSphereGeometry(bundledThree);
      graph
        .nodeThreeObject(function (d) { return createNodeMesh(d); })
        .nodeThreeObjectExtend(false);
      graph.graphData({ nodes: nodes3d, links: buildLinks() });
      ensureSceneLights(graph.scene(), bundledThree);
      customMeshesInstalled = true;
      refreshAppearance();
      return true;
    }

    function scheduleCustomMeshInstall(onReady) {
      if (customMeshesInstalled) {
        if (onReady) onReady();
        return;
      }
      if (meshInstallScheduled) return;
      meshInstallScheduled = true;
      var attempts = 0;
      var maxAttempts = 90;
      var onReadyCalled = false;
      function finish(onReadyFn) {
        meshInstallScheduled = false;
        if (onReadyFn && !onReadyCalled) {
          onReadyCalled = true;
          onReadyFn();
        }
      }
      function tryInstall() {
        attempts += 1;
        if (installCustomNodeMeshes()) {
          finish(onReady);
          return;
        }
        if (attempts < maxAttempts) {
          window.requestAnimationFrame(tryInstall);
        } else {
          finish(onReady);
        }
      }
      window.requestAnimationFrame(tryInstall);
    }

    function pauseRenderLoop() {
      if (!graph) return;
      if (typeof graph.pauseAnimation === 'function') graph.pauseAnimation();
      if (typeof graph.d3AlphaTarget === 'function') graph.d3AlphaTarget(0);
    }

    function resumeRenderLoop() {
      if (!graph) return;
      if (typeof graph.resumeAnimation === 'function') graph.resumeAnimation();
      if (typeof graph.d3AlphaTarget === 'function') graph.d3AlphaTarget(0.25);
    }

  /** graphData 更新完成后再 reheat / resume，避免 layout 未就绪时 tick 抛错中断 rAF。 */
    function reheatAfterGraphData() {
      if (!graph) return;
      if (typeof graph.d3AlphaTarget === 'function') graph.d3AlphaTarget(0.25);
      if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
      resumeRenderLoop();
    }

    function resetNodePositionsInPlace() {
      sourceNodes.forEach(function (src) {
        var n3 = nodeById.get(src.id);
        if (!n3) return;
        n3.x = src.x != null ? src.x : (Math.random() - 0.5) * 80;
        n3.y = src.y != null ? src.y : (Math.random() - 0.5) * 80;
        n3.z = src.z != null ? src.z : (Math.random() - 0.5) * 80;
        n3.vx = 0;
        n3.vy = 0;
        n3.vz = 0;
        delete n3.fx;
        delete n3.fy;
        delete n3.fz;
      });
    }

    function clearMagneticForces() {
      ['x', 'y', 'z'].forEach(function (name) {
        if (graph.d3Force(name)) graph.d3Force(name, null);
      });
    }

    function applyMagneticForces() {
      var magneticConfig = getMagneticConfig();
      if (!magneticConfig || !magneticConfig.enabled) {
        clearMagneticForces();
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

    function bboxFromNodeData(nodeFilter) {
      var filterFn = nodeFilter || function () { return true; };
      var minX = Infinity;
      var maxX = -Infinity;
      var minY = Infinity;
      var maxY = -Infinity;
      var minZ = Infinity;
      var maxZ = -Infinity;
      var count = 0;
      nodes3d.forEach(function (n) {
        if (!filterFn(n) || n.x == null || n.y == null || n.z == null) return;
        var r = sphereRadiusFor(n) + 8;
        count += 1;
        minX = Math.min(minX, n.x - r);
        maxX = Math.max(maxX, n.x + r);
        minY = Math.min(minY, n.y - r);
        maxY = Math.max(maxY, n.y + r);
        minZ = Math.min(minZ, n.z - r);
        maxZ = Math.max(maxZ, n.z + r);
      });
      if (!count || !isFinite(minX)) return null;
      return { x: [minX, maxX], y: [minY, maxY], z: [minZ, maxZ] };
    }

    function bboxSpan(bbox) {
      if (!bbox) return 0;
      var halfX = (bbox.x[1] - bbox.x[0]) / 2;
      var halfY = (bbox.y[1] - bbox.y[0]) / 2;
      var halfZ = (bbox.z[1] - bbox.z[0]) / 2;
      return Math.sqrt(halfX * halfX + halfY * halfY + halfZ * halfZ);
    }

    function inflateBbox(bbox, nodeFilter) {
      var padR = 0;
      nodes3d.forEach(function (n) {
        if (nodeFilter && !nodeFilter(n)) return;
        if (n.x == null || n.y == null || n.z == null) return;
        padR = Math.max(padR, sphereRadiusFor(n) + 8);
      });
      return {
        x: [bbox.x[0] - padR, bbox.x[1] + padR],
        y: [bbox.y[0] - padR, bbox.y[1] + padR],
        z: [bbox.z[0] - padR, bbox.z[1] + padR],
      };
    }

    function fitCameraToBbox(bbox, duration) {
      if (!graph || !bbox) return false;
      var cx = (bbox.x[0] + bbox.x[1]) / 2;
      var cy = (bbox.y[0] + bbox.y[1]) / 2;
      var cz = (bbox.z[0] + bbox.z[1]) / 2;
      var halfX = Math.max((bbox.x[1] - bbox.x[0]) / 2, 8);
      var halfY = Math.max((bbox.y[1] - bbox.y[0]) / 2, 8);
      var halfZ = Math.max((bbox.z[1] - bbox.z[0]) / 2, 8);
      var radius = Math.sqrt(halfX * halfX + halfY * halfY + halfZ * halfZ);

      var view = measureContainerSize(container);
      var cam3d = graph.camera();
      var fovDeg = (cam3d && cam3d.fov) || 50;
      var fov = fovDeg * Math.PI / 180;
      var aspect = view.width / Math.max(view.height, 1);
      var distV = radius / Math.sin(fov / 2);
      var hFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);
      var distH = radius / Math.sin(hFov / 2);
      var dist = Math.max(distV, distH, 48);
      dist *= 0.78;

      var lookAt = { x: cx, y: cy, z: cz };
      var cur = graph.cameraPosition();
      var dx = cur.x - cur.lookAt.x;
      var dy = cur.y - cur.lookAt.y;
      var dz = cur.z - cur.lookAt.z;
      var len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-3) {
        dx = 0.35;
        dy = 0.55;
        dz = 1.0;
        len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      graph.cameraPosition(
        { x: cx + (dx / len) * dist, y: cy + (dy / len) * dist, z: cz + (dz / len) * dist },
        lookAt,
        duration == null ? 650 : duration
      );
      return true;
    }

    function zoomFitToNodes(duration, nodeFilter) {
      if (!graph) return;
      var filterFn = nodeFilter || function () { return true; };
      var sceneBbox = graph.getGraphBbox(filterFn);
      var dataBbox = bboxFromNodeData(filterFn);
      var bbox = dataBbox || sceneBbox;
      if (sceneBbox && dataBbox) {
        var sceneSpan = bboxSpan(sceneBbox);
        var dataSpan = bboxSpan(dataBbox);
        // 力模拟首帧前三维对象常堆在原点；数据坐标已展开时优先信 scene bbox。
        if (sceneSpan >= dataSpan * 0.25) bbox = sceneBbox;
        else bbox = dataBbox;
      }
      if (bbox) bbox = inflateBbox(bbox, filterFn);
      if (!fitCameraToBbox(bbox, duration)) {
        graph.zoomToFit(duration, 80, filterFn);
      }
    }

    function scheduleInitialFit(ms) {
      var duration = ms == null ? 450 : ms;
      function runFit() {
        if (!graph) return;
        if (typeof graph.zoomToFit === 'function') {
          graph.zoomToFit(duration, 100);
          return;
        }
        zoomFitToNodes(duration);
      }
      if (typeof graph.onEngineStop === 'function') {
        graph.onEngineStop(function () { runFit(); });
      }
      window.setTimeout(runFit, 2000);
    }

    function onContainerPointerMove(ev) {
      lastPointer.clientX = ev.clientX;
      lastPointer.clientY = ev.clientY;
      if (onPointerMove) onPointerMove(ev);
    }
    container.addEventListener('mousemove', onContainerPointerMove);
    container.addEventListener('pointermove', onContainerPointerMove);

    var graph = null;

    function afterGraphDataApplied(fn) {
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(fn);
      });
    }

    function bindGraphInstance() {
      graph
        .width(size.width)
        .height(size.height)
        .backgroundColor(backgroundColor())
        .showNavInfo(false)
        .warmupTicks(8)
        .cooldownTicks(24)
        .nodeId('id')
        .nodeRelSize(1)
        .nodeResolution(8)
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
    }

    function ensureGraph() {
      if (graph) return graph;
      size = measureContainerSize(container);
      // init 末尾会同步启动 rAF；先 pause，待 graphData onFinishUpdate 后再 resume。
      graph = window.ForceGraph3D()(container);
      graph.pauseAnimation();
      bindGraphInstance();
      return graph;
    }

    return {
      show: function () {
        container.hidden = false;
        syncPositionsFromSource();
        ensureGraph();
        syncViewport();
        graph.backgroundColor(backgroundColor());
        refreshAppearance();
        pauseRenderLoop();
        graph.graphData({ nodes: nodes3d, links: buildLinks() });
        afterGraphDataApplied(function () {
          if (!graph) return;
          reheatAfterGraphData();
          syncViewport();
          if (typeof graph.zoomToFit === 'function') graph.zoomToFit(0, 90);
          scheduleInitialFit(650);
        });
      },

      hide: function () {
        pauseRenderLoop();
        container.hidden = true;
      },

      pauseSimulation: function () {
        pauseRenderLoop();
      },

      resumeSimulation: function () {
        resumeRenderLoop();
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
        if (!graph) return;
        var charge = graph.d3Force('charge');
        if (charge && charge.strength) charge.strength(getChargeStrength());
        applyMagneticForces();
        afterGraphDataApplied(function () {
          if (!graph) return;
          if (typeof graph.d3AlphaTarget === 'function') graph.d3AlphaTarget(0.22);
          else if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
        });
      },

      restartSimulation: function () {
        if (customMeshesInstalled) {
          resetNodePositionsInPlace();
          this.updateForces();
          return;
        }
        nodes3d = sourceNodes.map(cloneNodeFor3D);
        rebuildNodeIndex();
        graph.graphData({ nodes: nodes3d, links: buildLinks() });
        this.updateForces();
      },

      fitToScreen: function (ms) {
        zoomFitToNodes(ms == null ? 650 : ms);
      },

      fitToVisible: function (ms) {
        var visible = getVisibleNodeIds();
        zoomFitToNodes(ms == null ? 650 : ms, function (n) { return visible.has(n.id); });
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
        if (graph && graph._destructor) graph._destructor();
        graph = null;
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
