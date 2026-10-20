(function () {
  'use strict';

  /**
   * 知识图谱 3D 视图（3d-force-graph）。
   * 使用独立节点副本，避免与 2D d3 力模拟争用节点对象。
   */
  function isAvailable() {
    return typeof window.ForceGraph3D === 'function';
  }

  /**
   * 3D 社区胶囊按画布短边连续缩放（相对约 800px 的桌面参考短边）。
   * 取代原先「≤768 / 粗指针 → 字号×0.55 且 zoom×0.55」的二元收紧：
   * 那套在手机/平板上有效字号可落到 ~3–5px（难读），大屏又完全不放大。
   */
  var COMMUNITY_LABEL_REF_SHORT_PX = 800;
  var COMMUNITY_LABEL_VP_SCALE_MIN = 0.78;
  var COMMUNITY_LABEL_VP_SCALE_MAX = 1.28;
  var COMMUNITY_LABEL_FONT_ABS_MIN_PX = 7;
  var COMMUNITY_LABEL_FONT_ABS_MAX_PX = 22;
  var COMMUNITY_LABEL_COMPACT_SHORT_PX = 520;
  var COMMUNITY_LABEL_PAD_Y = 5;
  var COMMUNITY_LABEL_PAD_X = 11;
  var COMMUNITY_LABEL_COMPACT_PAD_Y = 3.5;
  var COMMUNITY_LABEL_COMPACT_PAD_X = 7.5;

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

  function isCylinderGeometryType(geometry) {
    if (!geometry || !geometry.type) return false;
    return /^Cylinder(Buffer)?Geometry$/i.test(geometry.type);
  }

  /** three-forcegraph 连线可能是 Group 包裹的 Cylinder mesh，缩放必须落在圆柱体上。 */
  function resolveLinkCylinderMesh(linkObj) {
    if (!linkObj) return null;
    if (linkObj.isMesh && linkObj.geometry && isCylinderGeometryType(linkObj.geometry)) return linkObj;
    if (linkObj.children && linkObj.children.length) {
      for (var i = 0; i < linkObj.children.length; i++) {
        var ch = linkObj.children[i];
        if (ch.isMesh && ch.geometry && isCylinderGeometryType(ch.geometry)) return ch;
      }
    }
    return linkObj.isMesh ? linkObj : null;
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
    // 更新明度渐变系数（1 = 全亮）：由 graph.html 按节点 recency 注入，关闭时恒为 1。
    var getNodeDim = opts.getNodeDim || function () { return 1; };
    var getNodeRadius = opts.getNodeRadius || function () { return 6; };
    var getLinkColor = opts.getLinkColor || function () { return 'rgba(255,255,255,0.07)'; };
    var getLinkWidth = opts.getLinkWidth || function () { return 1; };
    var isDark = opts.isDark || function () { return true; };
    // 低端设备（触屏 / ≤4 核 / ≤4 GB）：由 graph.html 的 LOW_END_DEVICE 注入，用于冷却节奏与渲染档位
    var isLowEnd = opts.isLowEnd || function () { return false; };
    var onNodeClick = opts.onNodeClick;
    var onNodeHover = opts.onNodeHover;
    var onPointerMove = opts.onPointerMove;
    // WebGL 上下文丢失且未能恢复时通知宿主（graph.html 据此回退 2D）
    var onContextLost = opts.onContextLost;
    var getVisibleNodeIds = opts.getVisibleNodeIds || function () { return new Set(); };
    var hasActiveFilter = opts.hasActiveFilter || function () { return false; };
    var edgeHighlightsWithNode = opts.edgeHighlightsWithNode;
    var getChargeStrength = opts.getChargeStrength || function () { return -800; };
    var getMagneticConfig = opts.getMagneticConfig || function () { return null; };
    var resolveSourceNode = opts.resolveSourceNode || function (id) {
      return sourceNodes.find(function (n) { return n.id === id; });
    };
    var areEdgesVisible = opts.areEdgesVisible || function () { return true; };
    var areNodeLabelsVisible = opts.areNodeLabelsVisible || function () { return true; };
    var getNodeLabelText = opts.getNodeLabelText || function (d) {
      return d.label || d.id;
    };
    var getNodeLabelFontSize = opts.getNodeLabelFontSize || function () { return 10; };
    var getNodeLabelColor = opts.getNodeLabelColor || function () {
      return isDark() ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.5)';
    };
    var getNodeLabelOpacity = opts.getNodeLabelOpacity || function () { return 0.85; };
    var isNodeLabelHub = opts.isNodeLabelHub || function () { return false; };
    // 社区聚类中心胶囊标签（HTML overlay，投影自 3D 质心）
    var areCommunityLabelsVisible = opts.areCommunityLabelsVisible || function () { return false; };
    var getCommunityLabelDescriptors = opts.getCommunityLabelDescriptors || function () { return []; };

    function getCommunityLabelCanvasShortSide() {
      var w = (container && container.clientWidth) || window.innerWidth || 1440;
      var h = (container && container.clientHeight) || window.innerHeight || 900;
      return Math.min(w, h);
    }

    /** 相对桌面参考短边的连续视口比例（sqrt 缓两端：手机抬底、大屏限顶） */
    function communityLabelViewportScale() {
      var shortSide = getCommunityLabelCanvasShortSide();
      var raw = Math.sqrt(Math.max(1e-3, shortSide / COMMUNITY_LABEL_REF_SHORT_PX));
      return Math.max(
        COMMUNITY_LABEL_VP_SCALE_MIN,
        Math.min(COMMUNITY_LABEL_VP_SCALE_MAX, raw)
      );
    }

    function resolveCommunityLabelFontSize(baseFontSize) {
      var base = (baseFontSize != null && isFinite(baseFontSize)) ? baseFontSize : 13;
      var fontSize = base * communityLabelViewportScale();
      return Math.max(
        COMMUNITY_LABEL_FONT_ABS_MIN_PX,
        Math.min(COMMUNITY_LABEL_FONT_ABS_MAX_PX, fontSize)
      );
    }

    /** 相机 dolly 后的 scale 钳制：窄屏上限略紧，避免胶囊盖住节点 */
    function communityLabelZoomClamps() {
      var shortSide = getCommunityLabelCanvasShortSide();
      if (shortSide < COMMUNITY_LABEL_COMPACT_SHORT_PX) {
        return { min: 0.55, max: 1.25 };
      }
      if (shortSide < 900) {
        return { min: 0.45, max: 1.55 };
      }
      return { min: 0.4, max: 1.75 };
    }

    var labelLayer = null;
    var labelEls = new Map();
    var communityLabelEls = new Map();
    var labelTickBound = false;
    var labelsLayoutReady = false;
    var nodeLabelsRendered = false;   // 已把节点标签渲染成可见状态（用于「由显到隐」只熄灭一次）
    var baselineCameraDist = null;
    var cameraZoomInteracted = false;
    // 程序化取景（适配屏幕）期间锁定胶囊 scale，避免飞行动画中途
    // |cam−lookAt| 瞬时偏离基线导致标签忽大忽小，结束再猛跳回 1。
    var fitLabelZoomLocked = false;
    var fitLabelZoomUnlockTimer = 0;
    // 社区质心世界坐标缓存：相机旋转时节点坐标不变，避免每帧 O(N) 重算。
    var communityCentroidsCache = null;
    var communityCentroidsDirty = true;
    var cameraLabelSyncRaf = 0;
    // 屏幕坐标写入阈值（px）：亚像素抖动不写 DOM，减轻 layout/compositor 压力。
    var LABEL_POS_EPS = 0.35;

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
    var pendingFirstShowKick = false;
    var initialFitDone = false;
    // 引擎 tick 计数（每次 graphData / reheat 后由库从 alpha=1 重新衰减，这里只用于首次取景与标签就绪）
    var engineTicks = 0;
    var INITIAL_FIT_TICKS = 60;
    var initialFitRunner = null;
    // 拖拽手势期间是否已 reheat：见 bindGraphInstance 里 onNodeDrag 注释。
    var nodeDragReheated = false;
    // 时序激活动画：null = 关闭（正常渲染）；Set = 「已激活」节点集合（= 进入力模拟的子集）。
    // 仿 2D：每激活一个节点就把它种在已激活邻居附近、加入力模拟子集并重排，整图随时间生长。
    var timelineRevealedIds = null;
    var timelineOrder = [];          // 按时间排序后的 node3d 列表（timelineBegin 传入）
    var timelineCount = 0;           // 当前已激活数量（= 力模拟子集大小）
    var timelineSnapshot = null;     // 进入前全图位置快照（退出时还原）
    var timelinePopIn = new Map();   // nodeId -> 激活起始时间戳（pop-in 放大动画）
    var timelinePopRaf = null;
    var TIMELINE_POP_MS = 280;

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
      return isDark() ? '#191919' : '#f7f6f3';
    }

    function getCameraDistance() {
      if (!graph) return null;
      var px;
      var py;
      var pz;
      var tx;
      var ty;
      var tz;
      var cam3d = (typeof graph.camera === 'function') ? graph.camera() : null;
      if (cam3d && cam3d.position) {
        px = cam3d.position.x;
        py = cam3d.position.y;
        pz = cam3d.position.z;
      } else if (typeof graph.cameraPosition === 'function') {
        var cam = graph.cameraPosition();
        if (!cam) return null;
        px = cam.x;
        py = cam.y;
        pz = cam.z;
      } else {
        return null;
      }
      // 滚轮缩放时 TrackballControls.target 比 cameraPosition().lookAt 更可靠
      var controls = (typeof graph.controls === 'function') ? graph.controls() : null;
      if (controls && controls.target) {
        tx = controls.target.x;
        ty = controls.target.y;
        tz = controls.target.z;
      } else if (typeof graph.cameraPosition === 'function') {
        var camLook = graph.cameraPosition();
        if (!camLook || !camLook.lookAt) return null;
        tx = camLook.lookAt.x;
        ty = camLook.lookAt.y;
        tz = camLook.lookAt.z;
      } else {
        return null;
      }
      var dx = px - tx;
      var dy = py - ty;
      var dz = pz - tz;
      var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      return (isFinite(dist) && dist > 1e-3) ? dist : null;
    }

    function getLabelZoomFactor() {
      var dist = getCameraDistance();
      if (!dist || !baselineCameraDist) return 1;
      return baselineCameraDist / dist;
    }

    function captureBaselineCameraDist() {
      var dist = getCameraDistance();
      if (dist && isFinite(dist)) baselineCameraDist = dist;
    }

    function ensureLabelLayer() {
      if (labelLayer) return labelLayer;
      labelLayer = document.createElement('div');
      labelLayer.className = 'graph-3d-labels';
      labelLayer.setAttribute('aria-hidden', 'true');
      container.appendChild(labelLayer);
      return labelLayer;
    }

    // 节点标签默认关闭（showNodeLabels=false，浮窗开关亦已隐藏），此前每次进入 3D 都会
    // 预建 N 个 div、且每次 syncLabelStyles 无差别写一遍 textContent/color/fontSize。
    // 改为「按需创建」：只有真正要显示的标签才建元素，未显示的节点不产生任何 DOM。
    function buildLabelEls() {
      var layer = ensureLabelLayer();
      if (!layer) return;
      labelEls.clear();
      communityLabelEls.clear();   // innerHTML='' 会一并清掉社区胶囊，重建时重新创建
      invalidateCommunityCentroids3D();
      nodeLabelsRendered = false;
      layer.innerHTML = '';
      syncLabelStyles();
    }

    // 惰性创建单个节点标签元素，并写入与 syncLabelStyles 一致的静态样式。
    function ensureNodeLabelEl(id) {
      var el = labelEls.get(id);
      if (el) return el;
      var layer = ensureLabelLayer();
      if (!layer) return null;
      var src = resolveSourceNode(id) || nodeById.get(id);
      if (!src) return null;
      el = document.createElement('div');
      el.className = 'graph-3d-label';
      applyLabelStaticStyle(el, src, getNodeLabelColor());
      layer.appendChild(el);
      labelEls.set(id, el);
      return el;
    }

    function applyLabelStaticStyle(el, src, color) {
      el.textContent = getNodeLabelText(src);
      el.style.color = color;
      el.style.fontSize = getNodeLabelFontSize(src) + 'px';
      el.classList.toggle('is-depth-hub', !!isNodeLabelHub(src));
    }

    // 标签整体不可见时只需把已建元素熄灭一次，之后每帧直接返回。
    function hideAllNodeLabels() {
      if (!nodeLabelsRendered) return;
      nodeLabelsRendered = false;
      labelEls.forEach(function (el) { setLabelVisible(el, false, '0'); });
    }

    function nodeLabelsSuppressed() {
      return !areNodeLabelsVisible() || container.hidden || timelineActive() || !labelsLayoutReady;
    }

    function effectiveLabelOpacity(d) {
      if (!areNodeLabelsVisible() || container.hidden || timelineActive() || nodeHiddenByFilter(d)) {
        return 0;
      }
      if (!labelsLayoutReady) return 0;
      var src = resolveSourceNode(d.id) || d;
      var base = getNodeLabelOpacity(src, getLabelZoomFactor(), labelsLayoutReady, cameraZoomInteracted);
      if (base <= 0) return 0;
      return base * nodeOpacityFor(d);
    }

    function syncLabelStyles() {
      if (!labelLayer) return;
      if (nodeLabelsSuppressed()) { hideAllNodeLabels(); return; }
      nodeLabelsRendered = true;
      var color = getNodeLabelColor();
      labelEls.forEach(function (el, id) {
        var d = nodeById.get(id);
        if (!d) return;
        applyLabelStaticStyle(el, resolveSourceNode(id) || d, color);
        var opacity = effectiveLabelOpacity(d);
        var on = opacity > 0.02;
        setLabelVisible(el, on, on ? String(opacity) : '0');
      });
    }

    function syncLabelPositions() {
      if (!graph || container.hidden || !labelLayer) return;
      if (nodeLabelsSuppressed()) { hideAllNodeLabels(); return; }
      nodeLabelsRendered = true;
      var dist = getCameraDistance();
      if (baselineCameraDist && dist && Math.abs(dist - baselineCameraDist) > baselineCameraDist * 0.04) {
        cameraZoomInteracted = true;
      }
      var graphNodes = graph.graphData().nodes || [];
      graphNodes.forEach(function (d) {
        if (d.x == null || d.y == null) return;
        // 先判可见再投影：不可见节点不做两次 graph2ScreenCoords，也不建元素。
        var opacity = effectiveLabelOpacity(d);
        if (opacity <= 0.02) {
          var stale = labelEls.get(d.id);
          if (stale) setLabelVisible(stale, false, '0');
          return;
        }
        var z = d.z != null ? d.z : 0;
        var center = graph.graph2ScreenCoords(d.x, d.y, z);
        if (!center || !isFinite(center.x) || !isFinite(center.y)) {
          var off = labelEls.get(d.id);
          if (off) setLabelVisible(off, false, '0');
          return;
        }
        var el = ensureNodeLabelEl(d.id);
        if (!el) return;
        var src = resolveSourceNode(d.id) || d;
        var rimY = getNodeRadius(src) + 13;
        var bottom = graph.graph2ScreenCoords(d.x, d.y + rimY, z);
        var topY = bottom && isFinite(bottom.y) ? bottom.y : center.y + 12;
        setLabelScreenPos(el, center.x, topY, 'translate3d(' + center.x + 'px,' + topY + 'px,0) translate(-50%,0)');
        setLabelVisible(el, true, String(opacity));
      });
    }

    function setLabelScreenPos(el, x, y, transformValue, scaleKey) {
      var sk = (scaleKey != null && isFinite(scaleKey)) ? scaleKey : 1;
      if (el._lx != null
          && Math.abs(el._lx - x) < LABEL_POS_EPS
          && Math.abs(el._ly - y) < LABEL_POS_EPS
          && el._lz != null
          && Math.abs(el._lz - sk) < 0.002) {
        return;
      }
      el._lx = x;
      el._ly = y;
      el._lz = sk;
      el.style.transform = transformValue;
    }

    function setLabelVisible(el, on, opacityStr) {
      var nextOpacity = on ? (opacityStr != null ? opacityStr : '1') : '0';
      if (el._on === on && el._op === nextOpacity) return;
      el._on = on;
      el._op = nextOpacity;
      el.style.opacity = nextOpacity;
      el.style.visibility = on ? 'visible' : 'hidden';
    }

    function bindLabelTick() {
      if (labelTickBound || !graph) return;
      labelTickBound = true;
      graph.onEngineTick(function () {
        engineTicks += 1;
        engineRunning = true;
        if (idlePaused) wakeRender();
        // 慢设备上引擎每帧只跑 1 tick，等 alpha 自然衰减到阈值可能要一两分钟；
        // 跑够 INITIAL_FIT_TICKS 时布局骨架已成型，先取景，不等停机。
        if (!initialFitDone && engineTicks >= INITIAL_FIT_TICKS && initialFitRunner) initialFitRunner();
        // 力引擎推进时节点世界坐标在变 → 质心缓存失效
        invalidateCommunityCentroids3D();
        syncCommunityLabelPositions();
        if (!areNodeLabelsVisible()) return;
        maybeMarkLabelsLayoutReady();
        syncLabelPositions();
        syncLabelStyles();
      });
      bindCameraLabelSync();
    }

    var cameraLabelSyncBound = false;
    // 引擎收敛（alpha < d3AlphaMin）后 onEngineTick 停发；相机旋转/缩放/飞行动画只走渲染环，
    // 不重投影则标签冻结在旧屏幕坐标。TrackballControls 的 'change' 事件覆盖一切相机位移
    // （手动拖拽、滚轮缩放、cameraPosition 飞行动画），在此同步标签。
    // 用 rAF 合并同帧多次 change，且只做「缓存质心 → 屏幕投影」，避免每帧 O(N) 重算质心。
    // controls 与 renderer 同在库 init digest 创建，未就绪时下一帧重试（有上限兜底）。
    function bindCameraLabelSync(attempt) {
      if (cameraLabelSyncBound || !graph) return;
      var controls = (typeof graph.controls === 'function') ? graph.controls() : null;
      if (!controls || typeof controls.addEventListener !== 'function') {
        var next = (attempt || 0) + 1;
        if (next < 240) window.requestAnimationFrame(function () { bindCameraLabelSync(next); });
        return;
      }
      cameraLabelSyncBound = true;
      controls.addEventListener('change', function () {
        if (container.hidden) return;
        var cam = (typeof graph.camera === 'function') ? graph.camera() : null;
        var moved = true;
        if (cam && cam.position) {
          var pp = cam.position;
          if (lastChangeCamPos) {
            var ddx = pp.x - lastChangeCamPos.x;
            var ddy = pp.y - lastChangeCamPos.y;
            var ddz = pp.z - lastChangeCamPos.z;
            moved = (ddx * ddx + ddy * ddy + ddz * ddz) > CAMERA_CHANGE_WAKE_DIST2;
            lastChangeCamPos.x = pp.x; lastChangeCamPos.y = pp.y; lastChangeCamPos.z = pp.z;
          } else {
            lastChangeCamPos = { x: pp.x, y: pp.y, z: pp.z };
          }
        }
        if (moved || performance.now() - lastInputAt < INPUT_RECENT_MS) wakeRender(250);
        if (cameraLabelSyncRaf) return;
        cameraLabelSyncRaf = window.requestAnimationFrame(function () {
          cameraLabelSyncRaf = 0;
          if (!graph || container.hidden) return;
          syncCommunityLabelPositions();
          if (areNodeLabelsVisible()) syncLabelPositions();
        });
      });
    }

    function resetLabelsLayoutState() {
      labelsLayoutReady = false;
      baselineCameraDist = null;
      cameraZoomInteracted = false;
      fitLabelZoomLocked = false;
      if (fitLabelZoomUnlockTimer) {
        window.clearTimeout(fitLabelZoomUnlockTimer);
        fitLabelZoomUnlockTimer = 0;
      }
      syncLabelStyles();
    }

    function markLabelsLayoutReady() {
      labelsLayoutReady = true;
      captureBaselineCameraDist();
      syncLabelPositions();
      syncLabelStyles();
    }

    function maybeMarkLabelsLayoutReady() {
      if (labelsLayoutReady || !graph) return;
      // ForceGraph3D 1.80.0 不透出 d3Alpha()，此前的守卫是死代码（第一 tick 就标就绪）；
      // 改用引擎 tick 计数：跑够初始取景所需 tick 数或引擎已停机才算布局就绪。
      if (!initialFitDone && engineTicks < INITIAL_FIT_TICKS) return;
      markLabelsLayoutReady();
    }

    function showLabelLayer() {
      if (labelLayer) labelLayer.style.display = '';
    }

    function hideLabelLayer() {
      if (labelLayer) labelLayer.style.display = 'none';
    }

    /* ── 社区聚类中心胶囊标签 ── */
    function invalidateCommunityCentroids3D() {
      communityCentroidsDirty = true;
      communityCentroidsCache = null;
    }

    function ensureCommunityLabelEls() {
      var layer = ensureLabelLayer();
      if (!layer) return;
      getCommunityLabelDescriptors().forEach(function (desc) {
        var el = communityLabelEls.get(desc.id);
        if (!el) {
          el = document.createElement('div');
          el.className = 'graph-3d-community-label';
          layer.appendChild(el);
          communityLabelEls.set(desc.id, el);
        }
        el.textContent = desc.label;
        el.style.background = desc.color;
        el.style.color = desc.textColor;
        // 字号：graph.html 按社区节点数给出 3D 基准 8–16px，再乘视口短边比例并绝对钳制
        var fontSize = resolveCommunityLabelFontSize(desc.fontSize);
        var scale = fontSize / 13;
        var compact = getCommunityLabelCanvasShortSide() < COMMUNITY_LABEL_COMPACT_SHORT_PX;
        var padY = (compact ? COMMUNITY_LABEL_COMPACT_PAD_Y : COMMUNITY_LABEL_PAD_Y) * scale;
        var padX = (compact ? COMMUNITY_LABEL_COMPACT_PAD_X : COMMUNITY_LABEL_PAD_X) * scale;
        el.style.fontSize = fontSize + 'px';
        el.style.padding = padY.toFixed(2) + 'px ' + padX.toFixed(2) + 'px';
      });
    }

    // 质心只统计「有效可见」节点（语义与 2D 一致：无筛选全部；有筛选取 getVisibleNodeIds）
    function computeCommunityCentroids3D() {
      var acc = new Map();
      if (!graph) return [];
      var filtered = hasActiveFilter();
      var visible = filtered ? getVisibleNodeIds() : null;
      (graph.graphData().nodes || []).forEach(function (d) {
        if (!d.community || d.x == null || d.y == null) return;
        if (filtered && visible && !visible.has(d.id)) return;
        var a = acc.get(d.community);
        if (!a) { a = { sx: 0, sy: 0, sz: 0, n: 0 }; acc.set(d.community, a); }
        a.sx += d.x; a.sy += d.y; a.sz += (d.z != null ? d.z : 0); a.n += 1;
      });
      var out = [];
      acc.forEach(function (a, id) {
        out.push({ id: id, x: a.sx / a.n, y: a.sy / a.n, z: a.sz / a.n });
      });
      return out;
    }

    function getCommunityCentroids3D() {
      if (!communityCentroidsDirty && communityCentroidsCache) return communityCentroidsCache;
      communityCentroidsCache = computeCommunityCentroids3D();
      communityCentroidsDirty = false;
      return communityCentroidsCache;
    }

    /** 取景落稳后的胶囊 scale（zf=1 时的钳制结果；字号已按视口缩放，基线恒为 1） */
    function communityLabelZoomAtBaseline() {
      var clamps = communityLabelZoomClamps();
      return Math.max(clamps.min, Math.min(clamps.max, 1));
    }

    function communityLabelZoomScale() {
      // 取景飞行中锁定为落稳尺寸，不跟中途错误距离漂移
      if (fitLabelZoomLocked) return communityLabelZoomAtBaseline();
      // 首次适配前不懒捕获（避免把「已缩放后的距离」误记为基线）
      if (!baselineCameraDist) return 1;
      var zf = getLabelZoomFactor();
      if (!isFinite(zf) || zf <= 0) return 1;
      // 与节点一起放大缩小；钳制随画布短边略调，避免滚轮极限下过大/过小
      var clamps = communityLabelZoomClamps();
      return Math.max(clamps.min, Math.min(clamps.max, zf));
    }

    function invalidateCommunityLabelScaleCache() {
      communityLabelEls.forEach(function (el) { el._lz = null; });
    }

    /**
     * 程序化取景：起飞前把缩放基线设为目标距离，并在动画期间锁定胶囊为落稳尺寸。
     * 这样飞行动画中 controls.change 重投影不会把标签拉成忽大忽小，结束也无需猛跳。
     */
    function beginFitLabelZoomLock(targetDist, durationMs) {
      if (targetDist && isFinite(targetDist) && targetDist > 1e-3) {
        baselineCameraDist = targetDist;
        labelsLayoutReady = true;
      }
      if (fitLabelZoomUnlockTimer) {
        window.clearTimeout(fitLabelZoomUnlockTimer);
        fitLabelZoomUnlockTimer = 0;
      }
      var duration = Math.max(0, durationMs == null ? 0 : durationMs);
      if (duration <= 0) {
        // 瞬时取景：保留上方写入的目标基线，不在此处 capture（相机可能尚未落到目标）
        fitLabelZoomLocked = false;
        cameraZoomInteracted = false;
        invalidateCommunityLabelScaleCache();
        syncCommunityLabelPositions();
        return;
      }
      fitLabelZoomLocked = true;
      invalidateCommunityLabelScaleCache();
      syncCommunityLabelPositions();
      fitLabelZoomUnlockTimer = window.setTimeout(function () {
        fitLabelZoomUnlockTimer = 0;
        fitLabelZoomLocked = false;
        captureBaselineCameraDist();
        cameraZoomInteracted = false;
        invalidateCommunityLabelScaleCache();
        syncCommunityLabelPositions();
        if (areNodeLabelsVisible()) {
          syncLabelPositions();
          syncLabelStyles();
        }
      }, duration + 60);
    }

    // 沿视线推进/拉远相机（factor>1 放大），供验证与程序化缩放
    function dollyCameraByFactor(factor, durationMs) {
      if (!graph || !isFinite(factor) || factor <= 0) return false;
      var dist = getCameraDistance();
      var controls = (typeof graph.controls === 'function') ? graph.controls() : null;
      var cam3d = (typeof graph.camera === 'function') ? graph.camera() : null;
      if (!dist || !controls || !controls.target || !cam3d || !cam3d.position) return false;
      var tx = controls.target.x;
      var ty = controls.target.y;
      var tz = controls.target.z;
      var s = 1 / factor;
      var next = {
        x: tx + (cam3d.position.x - tx) * s,
        y: ty + (cam3d.position.y - ty) * s,
        z: tz + (cam3d.position.z - tz) * s,
      };
      wakeRender((durationMs == null ? 0 : durationMs) + 200);
      graph.cameraPosition(next, { x: tx, y: ty, z: tz }, durationMs == null ? 0 : durationMs);
      return true;
    }

    function syncCommunityLabelPositions() {
      if (!labelLayer || communityLabelEls.size === 0) return;
      var hide = !areCommunityLabelsVisible() || container.hidden || timelineActive() || !graph;
      var seen = new Set();
      if (!hide) {
        var zf = communityLabelZoomScale();
        // 相机旋转路径只读缓存；引擎 tick / syncLabels 会先 invalidate。
        getCommunityCentroids3D().forEach(function (c) {
          var el = communityLabelEls.get(c.id);
          if (!el) return;
          var p = graph.graph2ScreenCoords(c.x, c.y, c.z);
          if (!p || !isFinite(p.x) || !isFinite(p.y)) return;
          seen.add(c.id);
          setLabelScreenPos(
            el, p.x, p.y,
            'translate3d(' + p.x + 'px,' + p.y + 'px,0) translate(-50%,-50%) scale(' + zf + ')',
            zf
          );
        });
      }
      communityLabelEls.forEach(function (el, id) {
        setLabelVisible(el, !hide && seen.has(id));
      });
    }

    function syncLabels() {
      ensureCommunityLabelEls();
      // 全量同步（筛选/外观/数据变更）时质心可能变，强制重算一次。
      invalidateCommunityCentroids3D();
      syncLabelPositions();
      syncLabelStyles();
      syncCommunityLabelPositions();
    }

    // 连线记录只建一次并在每次 graphData 复用：库的数据映射按对象身份 diff，
    // 每次传新对象会把 53k 连线对象全部拆掉重建（首次进入曾因此重建 3 次）。
    var linkRecords = null;
    function getLinkRecords() {
      if (!linkRecords) linkRecords = buildLinks();
      return linkRecords;
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

    // 仅保留两端都已激活的连线（供时序生长时的力模拟子集使用）。
    function buildTimelineLinks() {
      return edges
        .map(function (e) {
          var sId = edgeEndpointId(e.source);
          var tId = edgeEndpointId(e.target);
          if (!timelineRevealedIds.has(sId) || !timelineRevealedIds.has(tId)) return null;
          var source = nodeById.get(sId);
          var target = nodeById.get(tId);
          if (!source || !target) return null;
          return { source: source, target: target, _ref: e };
        })
        .filter(Boolean);
    }

    // 新激活节点的出生位置：落在已激活邻居的质心附近（+ 抖动）；无已激活邻居则落在原点附近。
    // 之后由力模拟把它推到平衡位置，从而实现「冒一个、重排一次」的生长。
    function seedTimelineNodePos(d) {
      var sx = 0, sy = 0, sz = 0, cnt = 0;
      var neighbors = adjacency.get(d.id);
      if (neighbors) {
        neighbors.forEach(function (otherId) {
          if (!timelineRevealedIds.has(otherId)) return;
          var o = nodeById.get(otherId);
          if (!o || o.x == null) return;
          sx += o.x; sy += o.y; sz += (o.z != null ? o.z : 0); cnt += 1;
        });
      }
      if (cnt > 0) {
        d.x = sx / cnt + (Math.random() - 0.5) * 24;
        d.y = sy / cnt + (Math.random() - 0.5) * 24;
        d.z = sz / cnt + (Math.random() - 0.5) * 24;
      } else {
        d.x = (Math.random() - 0.5) * 40;
        d.y = (Math.random() - 0.5) * 40;
        d.z = (Math.random() - 0.5) * 40;
      }
      d.vx = 0; d.vy = 0; d.vz = 0;
      delete d.fx; delete d.fy; delete d.fz;
    }

    function nodeOpacityFor(d) {
      // 时序激活模式下子集内节点全部点亮（不参与 hover/侧栏淡化），未激活节点根本不在图里。
      if (timelineActive()) return 1;
      // 更新明度渐变：按节点 recency 注入的明暗系数，关闭时恒为 1 不影响既有淡化逻辑。
      var dim = getNodeDim(d);
      var visible = getVisibleNodeIds();
      var filtered = hasActiveFilter();
      var ok = visible.has(d.id);
      if (sidebarNodeId) {
        if (d.id === sidebarNodeId || sidebarDirect.has(d.id)) return dim;
        if (sidebarSecondary.has(d.id)) return 0.4 * dim;
        return 0.05 * dim;
      }
      if (hoverNodeId) {
        if (!filtered) return (hoverNodeId === d.id || isNeighborOf(hoverNodeId, d.id) ? 1 : 0.15) * dim;
        if (!ok) return 0.08 * dim;
        return (hoverNodeId === d.id || isNeighborOf(hoverNodeId, d.id) ? 1 : 0.15) * dim;
      }
      if (!filtered) return dim;
      return (ok ? 1 : 0.08) * dim;
    }

    function isNeighborOf(nodeId, otherId) {
      var set = adjacency.get(nodeId);
      return set ? set.has(otherId) : false;
    }

    // 3D 下「筛选 = 直接隐藏」：命中筛选集合之外的节点/连线整体不渲染，而非淡化。
    // 2D 视图仍沿用淡化逻辑（见 graph.html applyFilters），这两个判定只作用于 3D。
    function nodeHiddenByFilter(d) {
      return hasActiveFilter() && !getVisibleNodeIds().has(d.id);
    }

    function timelineActive() {
      return timelineRevealedIds !== null;
    }

    // 节点刚被激活时的放大系数：0.2 → 1，easeOutCubic 收敛，营造「点亮」感。
    function timelinePopFactor(id) {
      if (!timelinePopIn.has(id)) return 1;
      var t = (performance.now() - timelinePopIn.get(id)) / TIMELINE_POP_MS;
      if (t >= 1) { timelinePopIn.delete(id); return 1; }
      var e = 1 - Math.pow(1 - t, 3);
      return 0.2 + 0.8 * e;
    }

    function linkVisibleFor(l) {
      if (!areEdgesVisible()) {
        // 关闭「显示连线」后仍留下选中 / 悬停节点的连线，其余全部隐藏。
        var focusId = sidebarNodeId || hoverNodeId;
        if (!focusId || !edgeHighlightsWithNode) return false;
        return !!(l._ref && edgeHighlightsWithNode(l._ref, focusId));
      }
      if (!hasActiveFilter()) return true;
      var visible = getVisibleNodeIds();
      return visible.has(edgeEndpointId(l.source)) && visible.has(edgeEndpointId(l.target));
    }

    function linkOpacityFor(l) {
      var s = edgeEndpointId(l.source);
      var t = edgeEndpointId(l.target);
      // 时序模式下可见连线给一个稳定的较高不透明度（已由 linkVisibility 过滤掉未激活端点）。
      if (timelineActive()) return 0.25;
      var visible = getVisibleNodeIds();
      var filtered = hasActiveFilter();
      if (sidebarNodeId && edgeHighlightsWithNode) {
        var ref = l._ref;
        if (!ref) return 0.08;
        if (!filtered) return edgeHighlightsWithNode(ref, sidebarNodeId) ? 0.88 : 0.06;
        if (!visible.has(s) || !visible.has(t)) return 0.08;
        return edgeHighlightsWithNode(ref, sidebarNodeId) ? 0.88 : 0.06;
      }
      if (hoverNodeId) {
        // 非邻接连线保持常态值（不再整体从 0.06 变 0.1）：hover 只需改邻接连线，O(度) 而非 O(E)
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode && edgeHighlightsWithNode(eRef, hoverNodeId)) return 0.68;
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

    function linkBaseWidth() {
      // 3D 世界坐标下节点球体半径约 11–40；基准连线需与节点同量级才可见。
      return getLinkWidth() * 0.5;
    }

    // 期望连线宽度（高亮连线除染色外再加粗）。实际粗细不靠 linkWidth 访问器生效——
    // three-forcegraph 把圆柱半径烘进几何体，事后改访问器不会重建几何（连线只会被染色、
    // 不会变粗），所以这里返回的宽度交由 linkScaleUpdate（linkPositionUpdate 钩子）以 mesh 半径轴缩放来落地。
    function linkWidthFor(l) {
      var base = linkBaseWidth();
      if (sidebarNodeId) {
        var ref = l._ref;
        if (ref && edgeHighlightsWithNode && edgeHighlightsWithNode(ref, sidebarNodeId)) return base * 12;
      }
      if (hoverNodeId) {
        var eRef = l._ref;
        if (eRef && edgeHighlightsWithNode && edgeHighlightsWithNode(eRef, hoverNodeId)) return base * 8;
      }
      return base;
    }

    /* ── 连线材质 ──
       3d-force-graph 1.80.0 的 linkOpacity 是标量：此前传函数，库内 `linkOpacity * colorAlpha`
       算成 NaN → 全部连线 transparent:false / opacity:NaN，开「显示连线」后整图一片白线。
       另外任何访问器 setter（linkColor / linkOpacity / linkVisibility / linkWidth）都会触发整图
       digest，linkWidth 还会拆掉并重建全部 53k 连线对象（hover 一次 3 s 以上）。
       改法：把「颜色 × 不透明度」预先混进不透明的场景底色（背景是纯色，视觉上与 alpha 叠加
       等价，且不进透明队列、不逐帧排序），材质按颜色串共享；hover / 侧栏只给颜色变了的连线
       换材质，不再碰任何 setter。 */
    var LINK_BASE_OPACITY = 0.06;   // linkOpacityFor 的常态值；rgba 底色的 alpha 以此为基准缩放
    var linkMaterialCache = new Map();
    var linkMaterialCtor = null;
    // 「颜色串|不透明度」→ 混合后 hex 的备忘：全图 53k 连线只会出现几种组合，避免逐线正则解析
    var linkDisplayColorMemo = new Map();

    function parseCssColor(str) {
      var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(str || '');
      if (m) {
        var h = m[1].length === 3 ? m[1].replace(/./g, function (c) { return c + c; }) : m[1];
        var n = parseInt(h, 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1];
      }
      m = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$/i.exec(str || '');
      if (m) return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]];
      return [255, 255, 255, 1];
    }

    function toHex(r, g, b) {
      var n = (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
      return '#' + ('000000' + n.toString(16)).slice(-6);
    }

    /** 连线最终显示色：底色（可带 alpha）按 linkOpacityFor 缩放后与场景背景混合为不透明色。 */
    function linkDisplayColor(l) {
      var colorStr = linkColorFor(l);
      var op = linkOpacityFor(l);
      var key = colorStr + '|' + op;
      var hex = linkDisplayColorMemo.get(key);
      if (hex) return hex;
      var c = parseCssColor(colorStr);
      // 高亮色为不透明 hex 时直接用 op；rgba 底色自带的 alpha 按 op/常态值 等比缩放
      var alpha = c[3] < 1 ? c[3] * (op / LINK_BASE_OPACITY) : op;
      alpha = Math.max(0, Math.min(1, alpha));
      var bg = parseCssColor(backgroundColor());
      hex = toHex(bg[0] + (c[0] - bg[0]) * alpha, bg[1] + (c[1] - bg[1]) * alpha, bg[2] + (c[2] - bg[2]) * alpha);
      linkDisplayColorMemo.set(key, hex);
      return hex;
    }

    function linkMaterialFor(colorHex, sampleMaterial) {
      var mat = linkMaterialCache.get(colorHex);
      if (mat) return mat;
      if (!linkMaterialCtor) linkMaterialCtor = sampleMaterial && sampleMaterial.constructor;
      if (!linkMaterialCtor) return null;
      mat = new linkMaterialCtor({ color: colorHex });
      linkMaterialCache.set(colorHex, mat);
      return mat;
    }

    // 按端点索引当前 graphData 的连线（随 links 数组身份重建），供 hover 增量换材质
    var linkIndexSource = null;
    var linkIndexMap = null;
    function linksTouching(nodeId) {
      if (!graph || !nodeId) return [];
      var links = (graph.graphData() || {}).links || [];
      if (linkIndexSource !== links) {
        linkIndexSource = links;
        linkIndexMap = new Map();
        for (var i = 0; i < links.length; i++) {
          var l = links[i];
          var a = edgeEndpointId(l.source);
          var b = edgeEndpointId(l.target);
          if (!linkIndexMap.has(a)) linkIndexMap.set(a, []);
          if (!linkIndexMap.has(b)) linkIndexMap.set(b, []);
          linkIndexMap.get(a).push(l);
          if (b !== a) linkIndexMap.get(b).push(l);
        }
      }
      return linkIndexMap.get(nodeId) || [];
    }

    // 逐连线换材质：只对颜色变了的连线写 material，常态下 O(E) 比较、几乎不分配。
    function updateLinkMaterials(subset) {
      if (!graph) return;
      var links = subset || ((graph.graphData() || {}).links || []);
      for (var i = 0; i < links.length; i++) {
        var l = links[i];
        var mesh = resolveLinkCylinderMesh(l.__lineObj);
        if (!mesh || !mesh.material) continue;
        var want = linkDisplayColor(l);
        if (mesh.userData.rnColor === want) continue;
        var mat = linkMaterialFor(want, mesh.material);
        if (!mat) continue;
        mesh.material = mat;
        mesh.userData.rnColor = want;
      }
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
        // 取景后节点球在屏上只有 2–8 px：低端用 8×6（80 三角）代替 12×10（216 三角）看不出差别
        sharedSphereGeometry = isLowEnd() ? new T.SphereGeometry(1, 8, 6) : new T.SphereGeometry(1, 12, 10);
      }
      return sharedSphereGeometry;
    }

    // 节点材质按「颜色|不透明度（取两位小数）」共享：此前 6,216 个节点各持一份 MeshLambertMaterial，
    // 每个 draw call 都要重传材质 uniform，hover / 侧栏淡化时还逐节点 color.set 重新解析颜色串。
    // 全图只有社区色 × 几档淡化系数这几十种组合。
    var nodeMaterialCache = new Map();
    function nodeMaterialFor(colorStr, opacity) {
      var q = Math.round(opacity * 100) / 100;
      var key = colorStr + '|' + q;
      var mat = nodeMaterialCache.get(key);
      if (mat) return mat;
      mat = new bundledThree.MeshLambertMaterial({ color: colorStr, transparent: q < 0.999, opacity: q });
      nodeMaterialCache.set(key, mat);
      return mat;
    }

    function createNodeMesh(d) {
      if (!bundledThree) return null;
      // 应用 pop-in 系数，避免新激活节点出现的首帧以全尺寸闪一下。
      var radius = sphereRadiusFor(d) * timelinePopFactor(d.id);
      var geometry = ensureSharedSphereGeometry(bundledThree);
      var material = nodeMaterialFor(getNodeColor(d), nodeOpacityFor(d));
      var mesh = new bundledThree.Mesh(geometry, material);
      mesh.scale.set(radius, radius, radius);
      mesh.userData.nodeId = d.id;
      mesh.userData.baseRadius = Math.max(11, getNodeRadius(d) * 1.65);
      return mesh;
    }

    // 逐节点外观。此前用 scene.traverse 遍历整棵场景树（节点 + 每条边各一个对象，
    // 全图约 3.8 万个），其中绝大多数是连线 mesh，白跑一遍判断。改为直接遍历数据侧的
    // nodes3d 并取库绑定的 __threeObj，工作量降到节点数量级。
    function updateNodeMeshes() {
      if (!graph || !bundledThree) return;
      var data = graph.graphData();
      var list = (data && data.nodes) || nodes3d;
      for (var i = 0; i < list.length; i++) {
        var d = list[i];
        var obj = d && d.__threeObj;
        if (!obj || !obj.isMesh || !obj.material) continue;
        var radius = sphereRadiusFor(d) * timelinePopFactor(d.id);
        var opacity = nodeOpacityFor(d);
        obj.scale.set(radius, radius, radius);
        var mat = nodeMaterialFor(getNodeColor(d), opacity);
        if (obj.material !== mat) obj.material = mat;
        obj.visible = !nodeHiddenByFilter(d) && opacity > 0.02;
      }
    }

    // 连线加粗：linkWidth 访问器只在「创建连线几何」时把半径烘进圆柱，事后改它不会重建已渲染
    // 连线；而高亮(改 linkColor 等)又会让库整体重建连线对象，抹掉任何对其 scale/geometry 的事后改动。
    // 故改用 linkPositionUpdate —— 库每帧/每次重建连线时都会回调它来摆放连线，正是稳定的落点：
    // 在这里按「期望宽度 / 基准宽度」设置 mesh 半径轴(x/y)缩放；返回 falsy 让库继续做默认定位
    // （只写 scale.z 长度 + lookAt 朝向），两者互不干扰，加粗即可稳定保留、且无需重建几何或重排布局。
    function linkScaleUpdate(obj, _coords, link) {
      var mesh = resolveLinkCylinderMesh(obj);
      if (!mesh) return false;
      if (!obj.userData.rnLayerSet) markLinkObjectLayer(obj);
      var mult = 1;
      if (sidebarNodeId || hoverNodeId) {
        var base = linkBaseWidth();
        if (base > 0) mult = linkWidthFor(link) / base;
      }
      mesh.scale.x = mult;
      mesh.scale.y = mult;
      return false;
    }

    // 外观刷新（hover / 侧栏 / 主题）：只直接改已有对象，不调用任何库 setter。
    function refreshAppearance() {
      if (!customMeshesInstalled) {
        // nodeOpacity 是标量配置：传函数会算出 NaN 不透明度让默认球体全透明。
        // 自定义 mesh 装上前用常量 1 保底可见，逐节点淡化交给 updateNodeMeshes。
        graph
          .nodeColor(function (d) { return getNodeColor(d); })
          .nodeVal(nodeValFor)
          .nodeOpacity(1);
      }
      updateNodeMeshes();
      updateLinkMaterials();
      syncLabels();
      wakeRender();
    }

    // hover 专用：节点全量（O(N)，逐节点淡化本就需要）+ 只换前后两个焦点节点的邻接连线材质
    function setHoverNode(nextId) {
      var prevId = hoverNodeId;
      hoverNodeId = nextId || null;
      if (prevId === hoverNodeId) return;
      updateNodeMeshes();
      if (prevId) updateLinkMaterials(linksTouching(prevId));
      if (hoverNodeId) updateLinkMaterials(linksTouching(hoverNodeId));
      syncLabels();
      wakeRender();
      // 关闭「显示连线」时只画焦点连线：焦点变化即可见集合变化
      if (!areEdgesVisible()) refreshVisibility();
    }

    // 可见集合变化（筛选 / 显示连线开关 / 关连线时的焦点连线）才需要库 digest 增删对象；
    // 访问器本身读闭包状态，重新 set 只是为了触发 digest。刻意不碰 linkWidth。
    function refreshVisibility() {
      if (!graph) return;
      graph
        .nodeVisibility(function (d) { return !nodeHiddenByFilter(d); })
        .linkVisibility(function (l) { return linkVisibleFor(l); });
      afterGraphDataApplied(function () {});   // digest 落地后给新建的连线对象标层
      wakeRender();
    }

    // pop-in 期间逐帧刷新节点 mesh 缩放，让「激活」放大过程平滑（由 3d-force-graph 的渲染环负责出图）。
    function timelinePopTick() {
      timelinePopRaf = null;
      if (!timelineActive()) { timelinePopIn.clear(); return; }
      updateNodeMeshes();
      if (timelinePopIn.size > 0) timelinePopRaf = window.requestAnimationFrame(timelinePopTick);
    }

    function ensureTimelinePopRunning() {
      if (timelinePopRaf == null && timelinePopIn.size > 0) {
        timelinePopRaf = window.requestAnimationFrame(timelinePopTick);
      }
    }

    function installCustomNodeMeshes() {
      if (customMeshesInstalled || !graph) return false;
      bundledThree = captureBundledThree(graph);
      if (!bundledThree) return false;
      ensureSharedSphereGeometry(bundledThree);
      // nodeThreeObject setter 本身即触发节点对象重建（库内 nodeDataMapper.clear()），
      // 无需再 graphData（那会连带重建全部连线并把 alpha 重置为 1）。
      graph
        .nodeThreeObject(function (d) { return createNodeMesh(d); })
        .nodeThreeObjectExtend(false);
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

    /* ── WebGL 上下文丢失 ──
       手机（尤其 iOS Safari）在内存压力下会丢掉上下文；three 自身只在 restored 时重建 GL 状态，
       若浏览器不再恢复，库的渲染环会对着死上下文空转、画面黑屏且无提示。这里：丢失即暂停渲染环，
       CONTEXT_RESTORE_GRACE_MS 内恢复则刷新外观继续，否则通知宿主回退 2D。 */
    var CONTEXT_RESTORE_GRACE_MS = 3000;
    var contextLostTimer = 0;
    var contextLost = false;
    function bindContextLossHandlers(canvasEl) {
      canvasEl.addEventListener('webglcontextlost', function (ev) {
        ev.preventDefault();   // 允许浏览器随后派发 webglcontextrestored
        contextLost = true;
        pauseRenderLoop();
        if (contextLostTimer) window.clearTimeout(contextLostTimer);
        contextLostTimer = window.setTimeout(function () {
          contextLostTimer = 0;
          if (!contextLost) return;
          if (onContextLost) onContextLost();
        }, CONTEXT_RESTORE_GRACE_MS);
      });
      canvasEl.addEventListener('webglcontextrestored', function () {
        contextLost = false;
        if (contextLostTimer) { window.clearTimeout(contextLostTimer); contextLostTimer = 0; }
        if (!graph || container.hidden) return;
        refreshAppearance();
        resumeRenderLoop();
        forceRendererResize();
      });
    }

    /* ── 拾取范围与节奏 ──
       库在渲染环里每 50 ms 对整个场景递归 raycast（开连线时 6 万对象，指针不动也跑），
       而本页没有任何连线的 hover / click 行为。连线对象放到 LINK_LAYER：库的 Raycaster 只查
       默认层（0），相机另开该层所以照常渲染。再把库的指针拾取只在「最近有输入」时打开。 */
    var LINK_LAYER = 1;
    function markLinkObjectLayer(obj) {
      if (!obj || obj.userData.rnLayerSet) return;
      if (typeof obj.traverse === 'function') obj.traverse(function (o) { if (o.layers) o.layers.set(LINK_LAYER); });
      else if (obj.layers) obj.layers.set(LINK_LAYER);
      obj.userData.rnLayerSet = true;
    }
    // 全量补标：连线对象由库在防抖 digest 里异步创建，时机不可靠，digest 后延迟扫两次兜底；
    // 常规路径由 linkScaleUpdate（库每次摆放连线都会回调）顺手标层，零额外遍历。
    function markLinkLayers() {
      if (!graph) return;
      var links = (graph.graphData() || {}).links || [];
      for (var i = 0; i < links.length; i++) markLinkObjectLayer(links[i].__lineObj);
    }
    function scheduleLinkLayerMarking() {
      window.setTimeout(markLinkLayers, 300);
      window.setTimeout(markLinkLayers, 2000);
    }

    var POINTER_IDLE_DISABLE_MS = 1000;
    var pointerInteractionTimer = 0;
    /** 最近有输入时才让库做 hover raycast；holdMs 后自动关闭（触屏给更长的保持期以保证轻触能命中） */
    function armPointerInteraction(holdMs) {
      if (!graph || typeof graph.enablePointerInteraction !== 'function') return;
      if (!graph.enablePointerInteraction()) graph.enablePointerInteraction(true);
      if (pointerInteractionTimer) window.clearTimeout(pointerInteractionTimer);
      pointerInteractionTimer = window.setTimeout(function () {
        pointerInteractionTimer = 0;
        if (graph && typeof graph.enablePointerInteraction === 'function') graph.enablePointerInteraction(false);
      }, holdMs || POINTER_IDLE_DISABLE_MS);
    }

    /* ── 渲染环空闲暂停 ──
       3d-force-graph 的 _animationCycle 无条件 rAF：引擎停机、相机静止时每帧仍重算 6 万对象矩阵、
       做视锥剔除、提交 6k–60k 次 draw call，并每 50 ms raycast 整个场景。这里在
       「引擎已停 + 无相机变化 / 指针 / 飞行动画」持续 IDLE_PAUSE_MS 后 pauseAnimation()，
       任何输入、reheat、外观变化或相机飞行都先 wakeRender() 再生效。 */
    var IDLE_PAUSE_MS = 600;
    var idlePauseTimer = 0;
    var idlePaused = false;
    var engineRunning = false;
    var renderHoldUntil = 0;
    var engineStopListeners = [];
    var lastInputAt = 0;
    var lastChangeCamPos = null;
    // TrackballControls 的阻尼尾巴会以亚像素位移持续派发 change；只有位移可见或刚有输入才唤醒
    var CAMERA_CHANGE_WAKE_DIST2 = 0.01;
    var INPUT_RECENT_MS = 1500;

    function scheduleIdlePause() {
      if (idlePauseTimer) window.clearTimeout(idlePauseTimer);
      var wait = Math.max(IDLE_PAUSE_MS, renderHoldUntil - performance.now() + 50);
      idlePauseTimer = window.setTimeout(tryIdlePause, wait);
    }

    function tryIdlePause() {
      idlePauseTimer = 0;
      if (!graph || container.hidden || idlePaused) return;
      if (engineRunning || performance.now() < renderHoldUntil || timelinePopRaf != null) {
        scheduleIdlePause();
        return;
      }
      idlePaused = true;
      if (typeof graph.pauseAnimation === 'function') graph.pauseAnimation();
    }

    /** 唤醒渲染环；holdMs > 0 表示至少持续渲染这么久（相机飞行等）。 */
    function wakeRender(holdMs) {
      if (!graph || container.hidden) return;
      if (holdMs) renderHoldUntil = Math.max(renderHoldUntil, performance.now() + holdMs);
      if (idlePaused) {
        idlePaused = false;
        resumeRenderLoop();
      }
      scheduleIdlePause();
    }

    function pauseRenderLoop() {
      if (!graph) return;
      if (idlePauseTimer) { window.clearTimeout(idlePauseTimer); idlePauseTimer = 0; }
      if (typeof graph.pauseAnimation === 'function') graph.pauseAnimation();
    }

    var resumeDeferrals = 0;
    function resumeRenderLoop() {
      // show() 排队的回调（afterGraphDataApplied / 自定义 mesh 安装）可能在用户已切回 2D 后才到达，
      // 此前没有 hidden 守卫 → 隐藏的 3D 继续跑整轮力模拟并每帧渲染 6 万对象
      if (!graph || container.hidden) return;
      // 3d-force-graph 在 kapsule 的防抖 digest 里才创建 WebGL renderer 并由 init 自启渲染环。
      // 主线程繁忙时本函数（经 afterGraphDataApplied 调度）可能早于该 digest 跑完，
      // 此刻 renderer 尚未就绪；若仍 resumeAnimation，会让库内部 _animationCycle 在对象未就绪时
      // 同步抛错——且抛错发生在 requestAnimationFrame 重排之前 → 渲染环被永久中断、整屏黑屏。
      // 故 renderer 未就绪时推迟到下一帧重试（有上限兜底）；待库 init digest 建好 renderer 并自启
      // 渲染环后，这里再 resumeAnimation 也只是无害的 no-op。
      if (typeof graph.renderer === 'function' && !graph.renderer()) {
        if (resumeDeferrals++ < 180) window.requestAnimationFrame(resumeRenderLoop);
        return;
      }
      // renderer 已就绪但 graphData digest 可能尚未把 layout 赋给 forceGraph；此时
      // _animationCycle → forceGraph.tickFrame → layout.tick() 仍会同步抛错并永久中断 rAF。
      // 捕获后推迟重试，直到 layout 建好（首次 2D→3D 切换、主线程繁忙时常见）。
      try {
        if (typeof graph.resumeAnimation === 'function') graph.resumeAnimation();
        resumeDeferrals = 0;
        idlePaused = false;
        scheduleIdlePause();
      } catch (_resumeErr) {
        if (resumeDeferrals++ < 180) window.requestAnimationFrame(resumeRenderLoop);
      }
    }

    // 渲染环自愈：万一 3d-force-graph 的 _animationCycle 仍在内部对象（renderObjs / forceGraph）
    // 未就绪时被唤醒并同步抛错，rAF 链会断开（抛错在 requestAnimationFrame 重排之前），表现为整屏
    // 黑屏且不再自行恢复。此处监听 window error，命中库自身的该类崩溃后下一帧重新点火渲染环；
    // 因为抛错时 animationFrameRequestId 仍为 null，resumeRenderLoop 在 renderer 就绪后即可重启。
    var renderLoopHealAttempts = 0;
    function reviveRenderLoopOnError(ev) {
      if (!graph || container.hidden) return;
      var fname = (ev && ev.filename) || '';
      if (fname.indexOf('3d-force-graph') === -1) return;
      var msg = (ev && ev.message) || (ev && ev.error && ev.error.message) || '';
      if (msg.indexOf('tick') === -1 && msg.indexOf('renderObjs') === -1 && msg.indexOf('forceGraph') === -1) return;
      if (renderLoopHealAttempts >= 60) return;
      renderLoopHealAttempts += 1;
      window.requestAnimationFrame(function () { resumeRenderLoop(); });
    }

  /** graphData 更新完成后再 reheat / resume，避免 layout 未就绪时 tick 抛错中断 rAF。 */
    function reheatAfterGraphData() {
      if (!graph || container.hidden) return;
      if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
      resumeRenderLoop();
    }

    function resetNodePositionsInPlace() {
      // 「刷新布局」时每个节点都重新随机散布在中心附近（与 2D randomizeNodePositions 对齐）。
      // sourceNodes 与 2D 力模拟共用同一批对象，src.x/src.y 早被 2D 布局写满；若沿用 src 坐标，
      // 刷新只会重随机 z，x/y 仍钉在上次布局——观感上几乎不变。这里三轴一律重随机，得到全新布局。
      sourceNodes.forEach(function (src) {
        var n3 = nodeById.get(src.id);
        if (!n3) return;
        n3.x = (Math.random() - 0.5) * 80;
        n3.y = (Math.random() - 0.5) * 80;
        n3.z = (Math.random() - 0.5) * 80;
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

    // 强制 three.js 重建 WebGL drawing buffer 并重绘：先把宽度收窄 1px（必产生一次
    // renderer.setSize），维持几帧让渲染器在新 buffer 上真正绘制、合成层重新拿到画面，
    // 再还原真实尺寸。3d-force-graph 的 drawing buffer 完全由 .width()/.height() 决定，
    // 重跑 graphData 不会触发 setSize；而尺寸变化会重建 drawing buffer + 重排 canvas +
    // 重绘合成层——正是「切回 2D 再切 3D」能恢复整屏空白的本质。收窄 1px 露出的是
    // #graph-canvas-3d 自身的同色暗背景，且仅维持几帧，肉眼不可见。
    function forceRendererResize() {
      if (!graph) return;
      var s = measureContainerSize(container);
      graph.width(Math.max(1, s.width - 1)).height(s.height);
      var frames = 0;
      function restore() {
        if (!graph) return;
        if (++frames < 3) { window.requestAnimationFrame(restore); return; }
        graph.width(s.width).height(s.height);
      }
      window.requestAnimationFrame(restore);
    }

    function graphFitPadPx() {
      var view = measureContainerSize(container);
      return Math.min(view.width, view.height) < 520 ? 48 : 120;
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
        var r = sphereRadiusFor(n) + 4;
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

    function inflateBbox(bbox) {
      // bbox 已含节点半径；仅留极小呼吸空间，避免与 fitCameraToBbox 的边距重复叠加。
      var pad = 8;
      return {
        x: [bbox.x[0] - pad, bbox.x[1] + pad],
        y: [bbox.y[0] - pad, bbox.y[1] + pad],
        z: [bbox.z[0] - pad, bbox.z[1] + pad],
      };
    }

    function fitCameraToBbox(bbox, duration) {
      if (!graph || !bbox) return false;
      var cx = (bbox.x[0] + bbox.x[1]) / 2;
      var cy = (bbox.y[0] + bbox.y[1]) / 2;
      var cz = (bbox.z[0] + bbox.z[1]) / 2;
      var halfX = Math.max((bbox.x[1] - bbox.x[0]) / 2, 8);
      var halfY = Math.max((bbox.y[1] - bbox.y[0]) / 2, 8);

      var view = measureContainerSize(container);
      var cam3d = graph.camera();
      var fovDeg = (cam3d && cam3d.fov) || 50;
      var fov = fovDeg * Math.PI / 180;
      var aspect = view.width / Math.max(view.height, 1);
      // 与 2D computeGraphFitTransform 对齐：按轴向半宽/半高分别取景，再取较远相机距离。
      var pad = graphFitPadPx() / 2;
      var distV = (halfY + pad) / Math.tan(fov / 2);
      var hFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);
      var distH = (halfX + pad) / Math.tan(hFov / 2);
      var dist = Math.max(distV, distH, 32);

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
      var durationMs = duration == null ? 650 : duration;
      // 起飞前锁定胶囊为落稳尺寸，并把基线设为目标距离（避免飞行中途 scale 闪烁）
      beginFitLabelZoomLock(dist, durationMs);
      wakeRender(durationMs + 200);
      graph.cameraPosition(
        { x: cx + (dx / len) * dist, y: cy + (dy / len) * dist, z: cz + (dz / len) * dist },
        lookAt,
        durationMs
      );
      // 瞬时取景：相机已落到目标后再用实测距离微调基线
      if (durationMs <= 0) {
        captureBaselineCameraDist();
        invalidateCommunityLabelScaleCache();
        syncCommunityLabelPositions();
      }
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
      if (bbox) bbox = inflateBbox(bbox);
      if (!fitCameraToBbox(bbox, duration)) {
        var durationMs = duration == null ? 650 : duration;
        beginFitLabelZoomLock(null, durationMs);
        graph.zoomToFit(durationMs, graphFitPadPx() / 2, filterFn);
        if (durationMs <= 0) {
          captureBaselineCameraDist();
          invalidateCommunityLabelScaleCache();
          syncCommunityLabelPositions();
        }
      }
    }

    function scheduleInitialFit(ms) {
      // 只在「首次力布局结束」时自动适配一次；适配后注销 onEngineStop，
      // 之后任何交互（hover / 点击 / 拖拽引发的 reheat）都不再自动适配屏幕。
      if (initialFitDone) return;
      var duration = ms == null ? 450 : ms;
      function runFit() {
        if (!graph || initialFitDone) return;
        initialFitDone = true;
        // 用基于节点数据坐标的安全 fit：库自带 zoomToFit 在场景对象尚未生成时
        // （getGraphBbox 返回 null）会静默 no-op，首次切换易停在未取景的相机上。
        zoomFitToNodes(duration);
        // 等相机飞行动画结束再记基线，否则社区标签 scale 会相对「半路距离」漂移
        window.setTimeout(function () {
          markLabelsLayoutReady();
          syncCommunityLabelPositions();
        }, Math.max(0, duration) + 60);
      }
      initialFitRunner = runFit;
      engineStopListeners.push(runFit);
      // 主触发：onEngineTick 计数到 INITIAL_FIT_TICKS 或 onEngineStop；兜底只防引擎根本没跑起来。
      window.setTimeout(runFit, 60000);
    }

    // 最近一次输入的指针类型：触屏抬指后库仍会按最后位置 raycast 并发 hover，须忽略
    var lastPointerType = 'mouse';
    function onContainerPointerMove(ev) {
      lastPointer.clientX = ev.clientX;
      lastPointer.clientY = ev.clientY;
      if (ev.pointerType) lastPointerType = ev.pointerType;
      lastInputAt = performance.now();
      wakeRender();   // hover 检测靠渲染环里的 raycast，指针一动就得先把环唤醒
      armPointerInteraction(ev && ev.pointerType === 'touch' ? 1200 : POINTER_IDLE_DISABLE_MS);
      if (onPointerMove) onPointerMove(ev);
    }
    function onContainerInput(ev) {
      if (ev && ev.type === 'touchstart') lastPointerType = 'touch';
      else if (ev && ev.pointerType) lastPointerType = ev.pointerType;
      lastInputAt = performance.now();
      wakeRender();
      armPointerInteraction(ev && (ev.type === 'touchstart' || ev.pointerType === 'touch') ? 1200 : POINTER_IDLE_DISABLE_MS);
    }
    container.addEventListener('mousemove', onContainerPointerMove);
    container.addEventListener('pointermove', onContainerPointerMove);
    container.addEventListener('pointerdown', onContainerInput, { passive: true });
    container.addEventListener('wheel', onContainerInput, { passive: true });
    container.addEventListener('touchstart', onContainerInput, { passive: true });
    window.addEventListener('error', reviveRenderLoopOnError);

    var graph = null;

    function afterGraphDataApplied(fn) {
      // graphData 经 kapsule 防抖 digest 异步落地；2×rAF 在 2D 力模拟刚停、1500+ 节点
      // 首次进 3D 时不足以等到 renderer/layout 就绪。轮询 renderer 后再回调，配合
      // resumeRenderLoop 的 try/catch 重试，避免 layout.tick 抢跑导致渲染环永久中断。
      var attempts = 0;
      function wait() {
        attempts += 1;
        if (!graph) {
          if (attempts < 240) window.requestAnimationFrame(wait);
          else fn();
          return;
        }
        if (typeof graph.renderer === 'function' && !graph.renderer()) {
          if (attempts < 240) window.requestAnimationFrame(wait);
          else fn();
          return;
        }
        markLinkLayers();
        scheduleLinkLayerMarking();
        fn();
      }
      window.requestAnimationFrame(wait);
    }

    function bindGraphInstance() {
      graph
        .width(size.width)
        .height(size.height)
        .backgroundColor(backgroundColor())
        .showNavInfo(false)
        // 像 2D 那样自然震荡收敛：不再 24 tick 硬停，而是让 alpha 从 1 自然衰减到
        // alphaMin 再停（≈270 tick）。三方库默认 cooldownTicks=∞ 且 d3AlphaMin=0
        // （永不因 alpha 停，只会跑满 cooldownTime），故显式开启 alpha 收敛阈值并对齐
        // 2D 的 alphaDecay(0.025)；warmup 设 0 以便从第一帧就能看到力模拟过程。
        //
        // 注意：three-forcegraph 的 tickFrame 在 layout.tick() 之前就判断
        // alpha < d3AlphaMin；引擎冷却后库内拖拽只做 d3AlphaTarget(0.3).resetCountdown()，
        // 无法把 alpha 抬过阈值，下一帧会立刻再次停机 → 松手后节点钉在落点不回弹。
        // 因此下方 onNodeDrag / onNodeDragEnd 需在手势期间 d3ReheatSimulation()。
        .warmupTicks(0)
        .cooldownTicks(Infinity)
        // 库默认 cooldownTime=15 s：慢设备每帧 1 tick，15 s 只够几十 tick，alpha≈0.3 就停机、
        // 相机 fit 到未收敛布局。改为只由 alpha 阈值停机，冷却节奏按设备档位缩放
        // （低端 decay 0.05 / alphaMin 0.01 ≈ 92 tick；桌面 0.025 / 0.005 ≈ 212 tick）。
        .cooldownTime(Infinity)
        .d3AlphaDecay(isLowEnd() ? 0.05 : 0.025)
        .d3AlphaMin(isLowEnd() ? 0.01 : 0.005)
        .d3VelocityDecay(0.4)
        .nodeId('id')
        .nodeRelSize(1)
        .nodeResolution(8)
        // 连线圆柱默认 6 棱 + 封顶 = 24 三角 / 条，投影后不足 1 px；3 棱柱 12 三角视觉一致，
        // 全图 53,740 条少约 64 万三角形
        .linkResolution(3)
        .nodeColor(function (d) { return getNodeColor(d); })
        .nodeVal(nodeValFor)
        .nodeOpacity(1)
        .nodeVisibility(function (d) { return !nodeHiddenByFilter(d); })
        // 颜色已混入底色（见 linkDisplayColor），库按颜色串缓存共享材质；linkOpacity 为标量
        .linkColor(linkDisplayColor)
        .linkOpacity(1)
        // 几何半径恒为基准宽度；逐连线加粗交给 linkPositionUpdate 里的 mesh 半径轴缩放。
        .linkWidth(linkBaseWidth)
        .linkPositionUpdate(linkScaleUpdate)
        .linkVisibility(function (l) { return linkVisibleFor(l); })
        .enableNodeDrag(true)
        .onNodeDrag(function () {
          // 每个拖拽手势只 reheat 一次，避免拖动过程中反复 alpha=1 导致整图狂抖。
          if (nodeDragReheated) return;
          nodeDragReheated = true;
          if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
        })
        .onNodeDragEnd(function () {
          nodeDragReheated = false;
          // 松手后再抬一次 alpha：确保回弹力模拟能跨过 alphaMin 判停门槛。
          // （库在 onNodeDragEnd 回调之后才会 alphaTarget(0).resetCountdown()，
          // 因此这里 reheat 后仍会以 target=0 自然收敛。）
          if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
        })
        .onNodeClick(function (node, ev) {
          var src = resolveSourceNode(node.id) || node;
          if (onNodeClick) onNodeClick(src, ev || lastPointer);
        })
        .onNodeHover(function (node) {
          // 侧栏/触屏选中态由 sidebarNodeId 驱动；旋转视角时 hover 会短暂落空，不能覆盖持久高亮。
          // 触屏没有 hover：旋转 / 平移抬指后落在节点上会被库当成悬停，整图淡化直到下次点击
          if (!sidebarNodeId && lastPointerType !== 'touch') setHoverNode(node ? node.id : null);
          var src = node ? (resolveSourceNode(node.id) || node) : null;
          if (onNodeHover) onNodeHover(src, lastPointer);
        })
        .onBackgroundClick(function () {
          setHoverNode(null);
          if (opts.onBackgroundClick) opts.onBackgroundClick();
        });
      // 唯一的 onEngineStop 处理器（kapsule 属性只能挂一个）：先跑监听者，再开始空闲计时
      graph.onEngineStop(function () {
        engineRunning = false;
        engineStopListeners.forEach(function (fn) { fn(); });
        scheduleIdlePause();
      });
      var cam = (typeof graph.camera === 'function') ? graph.camera() : null;
      if (cam && cam.layers) cam.layers.enable(LINK_LAYER);
      var chargeForce = graph.d3Force('charge');
      if (chargeForce && chargeForce.strength) chargeForce.strength(getChargeStrength());
    }

    function ensureGraph() {
      if (graph) return graph;
      size = measureContainerSize(container);
      // init 末尾会同步启动 rAF；先 pause，待 graphData onFinishUpdate 后再 resume。
      // rendererConfig 透传给内部 three.js WebGLRenderer：powerPreference='high-performance'
      // 提示浏览器/驱动在双显卡设备（典型为带集显+独显的笔记本）上优先选用独立 GPU。
      // 仅为性能提示——单 GPU 设备无效果，浏览器亦可忽略，不影响渲染正确性。
      // 低端：关 MSAA（填充率是移动 GPU 最弱项）、不透明上下文（背景本就是纯色，省一次页面合成混合）
      graph = window.ForceGraph3D({
        rendererConfig: {
          powerPreference: 'high-performance',
          antialias: !isLowEnd(),
          alpha: false,
        },
      })(container);
      // 库默认 setPixelRatio(min(2, DPR))：DPR 3 手机按 2× 渲染 = 4 倍像素。
      // 按档位封顶：低端 1×，其余 1.5×（2.6M 三角形的细圆柱在 1.5× 已无可见差别）。
      try {
        var renderer = typeof graph.renderer === 'function' ? graph.renderer() : null;
        if (renderer && typeof renderer.setPixelRatio === 'function') {
          renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isLowEnd() ? 1 : 1.5));
        }
        if (renderer && renderer.domElement) bindContextLossHandlers(renderer.domElement);
      } catch (_e) { /* 渲染器未就绪时沿用库默认 */ }
      graph.pauseAnimation();
      bindGraphInstance();
      pendingFirstShowKick = true;
      return graph;
    }

    // Chrome 下 2D 第一次切到 3D 偶发整屏空白（新建 WebGL 渲染器后首帧渲染管线未完成），
    // 现象上只有「切回 2D 再切 3D」能恢复。仅重跑 graphData 不会触发 renderer.setSize，
    // 无法恢复空白；恢复的本质是 drawing buffer 重建 + 合成层重绘，由 forceRendererResize
    // 显式抖动尺寸来强制完成。这里在首次 show 后幂等地复刻该恢复动作，避免用户手动来回切。
    function firstShowKick() {
      if (!pendingFirstShowKick) return;
      pendingFirstShowKick = false;
      window.setTimeout(function () {
        if (!graph || container.hidden) return;
        syncViewport();
        // 只做尺寸抖动：恢复空白的本质是 drawing buffer 重建 + 合成层重绘，
        // 此前这里还重跑一次 graphData（第三次全量重建连线），对恢复无帮助。
        afterGraphDataApplied(function () {
          if (!graph || container.hidden) return;
          forceRendererResize();
        });
      }, 250);
    }

    return {
      show: function () {
        container.hidden = false;
        showLabelLayer();
        syncPositionsFromSource();
        ensureGraph();
        armPointerInteraction();   // 需在 ensureGraph 之后：无输入时保持期一过即关闭库的拾取
        buildLabelEls();
        bindLabelTick();
        syncViewport();
        graph.backgroundColor(backgroundColor());
        refreshAppearance();
        pauseRenderLoop();
        graph.graphData({ nodes: nodes3d, links: getLinkRecords() });
        afterGraphDataApplied(function () {
          if (!graph) return;
          reheatAfterGraphData();
          syncViewport();
          // 首次进入时先框住初始布局，便于观看力模拟震荡收敛全过程；收敛完成（onEngineStop）
          // 后由 scheduleInitialFit 再适配一次最终结果。仅首次，re-entry/交互不触发。
          if (!initialFitDone) zoomFitToNodes(0);
          // 首帧取景后立刻记下基线，社区标签即可随后续滚轮缩放（不等 onEngineStop）
          captureBaselineCameraDist();
          // 默认渲染把 nodeOpacity 当标量，传入函数会得到 NaN 不透明度（节点全透明不可见）。
          // 装上自定义 mesh 后由 createNodeMesh / updateNodeMeshes 写入逐节点数值不透明度。
          scheduleCustomMeshInstall(function () {
            if (!graph) return;
            reheatAfterGraphData();
            scheduleInitialFit(350);
          });
          scheduleInitialFit(650);
          firstShowKick();
          syncLabels();
        });
      },

      hide: function () {
        pauseRenderLoop();
        hideLabelLayer();
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
        wakeRender();
        syncViewport();
        syncLabels();
      },

      syncFilters: function () {
        refreshAppearance();
        refreshVisibility();
      },

      syncLabels: function () {
        if (areNodeLabelsVisible()) maybeMarkLabelsLayoutReady();
        syncLabels();
      },

      // ── 时序激活动画（生长式，仿 2D）──
      // timelineBegin：进入时序模式，按时间顺序的 node id 列表初始化；先清空力模拟子集（0 个节点）。
      timelineBegin: function (orderedIds) {
        if (!graph) return;
        resetLabelsLayoutState();
        timelineSnapshot = nodes3d.map(function (n) {
          return { n: n, x: n.x, y: n.y, z: n.z };
        });
        timelineOrder = (orderedIds || [])
          .map(function (id) { return nodeById.get(id); })
          .filter(Boolean);
        timelineCount = 0;
        timelineRevealedIds = new Set();
        timelinePopIn.clear();
        graph.graphData({ nodes: [], links: [] });
      },

      // timelineSeek：把已激活数量推进/回退到 count。新激活节点种在已激活邻居附近后加入力模拟
      // 子集并重排（alpha 经 graphData 重置为 1，已平衡节点净力≈0 故几乎不动，新节点被推入位）。
      timelineSeek: function (count) {
        if (!graph || timelineRevealedIds === null) return;
        count = Math.max(0, Math.min(timelineOrder.length, Math.round(count)));
        if (count === timelineCount) return;
        var now = performance.now();
        if (count > timelineCount) {
          for (var i = timelineCount; i < count; i++) {
            var d = timelineOrder[i];
            timelineRevealedIds.add(d.id);   // 先入集合，便于后续节点把它当已激活邻居
            seedTimelineNodePos(d);
            timelinePopIn.set(d.id, now);
          }
        } else {
          for (var j = count; j < timelineCount; j++) {
            var rd = timelineOrder[j];
            timelineRevealedIds.delete(rd.id);
            timelinePopIn.delete(rd.id);
          }
        }
        timelineCount = count;
        graph.graphData({ nodes: timelineOrder.slice(0, count), links: buildTimelineLinks() });
        ensureTimelinePopRunning();
        wakeRender();
      },

      // 相机适配到「已激活」子集（生长时随之缩放，类似 2D fitToTimelineNodes）。
      timelineFit: function (ms) {
        if (timelineRevealedIds === null) return;
        // beginFitLabelZoomLock 在 fitCameraToBbox 内处理基线与飞行期锁定
        zoomFitToNodes(ms == null ? 500 : ms, function (n) { return timelineRevealedIds.has(n.id); });
        syncLabels();
      },

      // timelineEnd：退出时序模式 —— 还原进入前全图位置，恢复完整 graphData 并重排收敛。
      timelineEnd: function () {
        if (timelinePopRaf != null) { window.cancelAnimationFrame(timelinePopRaf); timelinePopRaf = null; }
        timelinePopIn.clear();
        timelineRevealedIds = null;
        timelineOrder = [];
        timelineCount = 0;
        if (timelineSnapshot) {
          timelineSnapshot.forEach(function (s) {
            s.n.x = s.x; s.n.y = s.y; s.n.z = s.z;
            s.n.vx = 0; s.n.vy = 0; s.n.vz = 0;
            delete s.n.fx; delete s.n.fy; delete s.n.fz;
          });
          timelineSnapshot = null;
        }
        if (!graph) return;
        graph.graphData({ nodes: nodes3d, links: getLinkRecords() });
        refreshAppearance();
        reheatAfterGraphData();
        markLabelsLayoutReady();
      },

      refreshColors: function () {
        graph.backgroundColor(backgroundColor());
        linkMaterialCache.clear();   // 混色依赖背景色，换主题后全部重算
        linkDisplayColorMemo.clear();
        refreshAppearance();
        syncLabels();
      },

      updateForces: function () {
        if (!graph) return;
        var charge = graph.d3Force('charge');
        if (charge && charge.strength) charge.strength(getChargeStrength());
        applyMagneticForces();
        afterGraphDataApplied(function () {
          if (!graph) return;
          // 力参数变化后重跑一轮自然收敛（alpha=1→0），而非把 alphaTarget 钉在 0.22。
          if (typeof graph.d3ReheatSimulation === 'function') graph.d3ReheatSimulation();
          wakeRender();
        });
      },

      restartSimulation: function () {
        resetLabelsLayoutState();
        if (customMeshesInstalled) {
          resetNodePositionsInPlace();
          this.updateForces();
          return;
        }
        nodes3d = sourceNodes.map(cloneNodeFor3D);
        rebuildNodeIndex();
        linkRecords = null;   // 节点对象换了，连线记录里被库解析成对象的 source/target 也要重建
        graph.graphData({ nodes: nodes3d, links: getLinkRecords() });
        this.updateForces();
      },

      fitToScreen: function (ms) {
        // 飞行期胶囊缩放锁定与基线写入由 fitCameraToBbox → beginFitLabelZoomLock 负责
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
        wakeRender((ms == null ? 700 : ms) + 200);
        graph.cameraPosition(
          { x: n3.x, y: n3.y, z: z + dist },
          n3,
          ms == null ? 700 : ms
        );
        window.setTimeout(function () {
          cameraZoomInteracted = true;
          syncLabels();
        }, (ms == null ? 700 : ms) + 40);
      },

      // 程序化推进相机并刷新社区标签 scale（验证 / 调试）
      dollyZoom: function (factor, ms) {
        if (!dollyCameraByFactor(factor, ms == null ? 0 : ms)) return false;
        cameraZoomInteracted = true;
        var delay = ms == null ? 0 : ms;
        if (delay <= 0) {
          syncCommunityLabelPositions();
          if (areNodeLabelsVisible()) syncLabelPositions();
          return true;
        }
        window.setTimeout(function () {
          syncCommunityLabelPositions();
          if (areNodeLabelsVisible()) syncLabelPositions();
        }, delay + 40);
        return true;
      },

      getCommunityLabelZoomScale: function () {
        return communityLabelZoomScale();
      },

      getCommunityLabelViewportScale: function () {
        return communityLabelViewportScale();
      },

      getCameraDistance: function () {
        return getCameraDistance();
      },

      getBaselineCameraDist: function () {
        return baselineCameraDist;
      },

      applySidebarHighlight: function (d, direct, secondary) {
        sidebarNodeId = d.id;
        sidebarDirect = direct || new Set();
        sidebarSecondary = secondary || new Set();
        hoverNodeId = null;
        refreshAppearance();
        if (!areEdgesVisible()) refreshVisibility();
      },

      clearSidebarHighlight: function () {
        sidebarNodeId = null;
        sidebarDirect = new Set();
        sidebarSecondary = new Set();
        refreshAppearance();
        if (!areEdgesVisible()) refreshVisibility();
      },

      // G2 基线测量：分离悬停刷新 / 标签同步 / 场景对象数（非业务 UI）
      measureHover: function (nodeId) {
        var t = performance.now();
        setHoverNode(nodeId || null);
        return +(performance.now() - t).toFixed(2);
      },

      measureSyncLabels: function () {
        var t = performance.now();
        syncLabels();
        return +(performance.now() - t).toFixed(2);
      },

      getGraphDataCounts: function () {
        if (!graph || typeof graph.graphData !== 'function') return { nodes: 0, links: 0 };
        var gd = graph.graphData() || {};
        return {
          nodes: Array.isArray(gd.nodes) ? gd.nodes.length : 0,
          links: Array.isArray(gd.links) ? gd.links.length : 0,
        };
      },

      getSceneStats: function () {
        var scene = graph && typeof graph.scene === 'function' ? graph.scene() : null;
        if (!scene) return null;
        var meshes = 0, lines = 0, labels = 0;
        var nodeMeshes = 0, dimmed = 0, bright = 0;
        var nodeMaterials = new Set();
        scene.traverse(function (obj) {
          if (obj.isMesh) {
            meshes += 1;
            if (obj.userData && obj.userData.nodeId && obj.material) {
              nodeMeshes += 1;
              nodeMaterials.add(obj.material);
              if (obj.material.opacity < 0.5) dimmed += 1; else bright += 1;
            }
          } else if (obj.isLine || obj.isLineSegments) lines += 1;
        });
        if (labelLayer) labels = labelLayer.querySelectorAll('.graph-3d-label').length;
        return {
          scene_meshes: meshes, scene_lines: lines, label_elements: labels,
          node_meshes: nodeMeshes, node_dimmed: dimmed, node_bright: bright,
          node_materials: nodeMaterials.size,
        };
      },

      // 验证用：模拟 WebGL 上下文丢失（restoreAfterMs > 0 则随后恢复）
      simulateContextLoss: function (restoreAfterMs) {
        var r = graph && typeof graph.renderer === 'function' ? graph.renderer() : null;
        var gl = r && typeof r.getContext === 'function' ? r.getContext() : null;
        var ext = gl && gl.getExtension('WEBGL_lose_context');
        if (!ext) return false;
        ext.loseContext();
        if (restoreAfterMs > 0) window.setTimeout(function () { ext.restoreContext(); }, restoreAfterMs);
        return true;
      },
      // 验证用：节点屏幕坐标与库指针拾取开关状态（非业务 UI）
      getNodeScreenPos: function (id) {
        var n = nodeById.get(id);
        if (!graph || !n || n.x == null) return null;
        var c = graph.graph2ScreenCoords(n.x, n.y, n.z != null ? n.z : 0);
        var rect = container.getBoundingClientRect();
        return c ? { x: c.x + rect.left, y: c.y + rect.top } : null;
      },
      isPointerInteractionEnabled: function () {
        return !!(graph && typeof graph.enablePointerInteraction === 'function' && graph.enablePointerInteraction());
      },

      // 连线材质体检（非业务 UI）：此前 linkOpacity 传函数导致全部材质 opacity 为 NaN
      getLinkMaterialStats: function () {
        if (!graph || typeof graph.graphData !== 'function') return null;
        var links = (graph.graphData() || {}).links || [];
        var total = 0, nan = 0, transparent = 0, colors = {}, sample = null, onLinkLayer = 0;
        for (var i = 0; i < links.length; i++) {
          var mesh = resolveLinkCylinderMesh(links[i].__lineObj);
          if (!mesh || !mesh.material) continue;
          total += 1;
          if (!isFinite(mesh.material.opacity)) nan += 1;
          if (mesh.material.transparent) transparent += 1;
          var hex = mesh.material.color ? mesh.material.color.getHexString() : '';
          colors[hex] = (colors[hex] || 0) + 1;
          if (mesh.layers && mesh.layers.mask === (1 << LINK_LAYER)) onLinkLayer += 1;
          if (!sample) sample = { color: hex, opacity: mesh.material.opacity, transparent: mesh.material.transparent };
        }
        return { total: total, nan_opacity: nan, transparent: transparent, colors: colors, sample: sample, on_link_layer: onLinkLayer };
      },

      // G0 基线测量：读取 three.js 当帧 draw calls / 三角形数（非业务 UI）
      getRendererInfo: function () {
        var r = graph && typeof graph.renderer === 'function' ? graph.renderer() : null;
        if (!r || !r.info) return null;
        return {
          frame: r.info.render.frame,
          draw_calls: r.info.render.calls,
          triangles: r.info.render.triangles,
          lines: r.info.render.lines,
          points: r.info.render.points,
          geometries: r.info.memory.geometries,
          textures: r.info.memory.textures,
          programs: r.info.programs ? r.info.programs.length : null,
          pixel_ratio: typeof r.getPixelRatio === 'function' ? r.getPixelRatio() : null,
          context: (typeof r.getContextAttributes === 'function') ? r.getContextAttributes() : null,
          drawing_buffer: r.domElement ? [r.domElement.width, r.domElement.height] : null,
        };
      },

      destroy: function () {
        container.removeEventListener('mousemove', onContainerPointerMove);
        container.removeEventListener('pointermove', onContainerPointerMove);
        container.removeEventListener('pointerdown', onContainerInput);
        container.removeEventListener('wheel', onContainerInput);
        container.removeEventListener('touchstart', onContainerInput);
        if (idlePauseTimer) { window.clearTimeout(idlePauseTimer); idlePauseTimer = 0; }
        if (pointerInteractionTimer) { window.clearTimeout(pointerInteractionTimer); pointerInteractionTimer = 0; }
        window.removeEventListener('error', reviveRenderLoopOnError);
        if (fitLabelZoomUnlockTimer) {
          window.clearTimeout(fitLabelZoomUnlockTimer);
          fitLabelZoomUnlockTimer = 0;
        }
        fitLabelZoomLocked = false;
        if (cameraLabelSyncRaf) {
          window.cancelAnimationFrame(cameraLabelSyncRaf);
          cameraLabelSyncRaf = 0;
        }
        if (graph && graph._destructor) graph._destructor();
        graph = null;
        labelLayer = null;
        labelEls.clear();
        communityLabelEls.clear();
        invalidateCommunityCentroids3D();
        labelTickBound = false;
        cameraLabelSyncBound = false;
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
