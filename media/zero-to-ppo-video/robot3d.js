// 舞台里的 URDF 机器人。模型取自 Robot Description Gallery Online（见 robots.json，python3 fetch_robots.py 下载到 assets/）。
// 全页只开一个离屏 WebGL 画布：每个视图是分镜 svg 里的一块 foreignObject + 2D canvas，渲染完把那一块拷过去，
// 几十个视图也只占一个 WebGL 上下文；姿态和高亮没变就不重画。
// 相机是正交的：世界坐标（米，z 朝上）线性映射到分镜坐标。视图给定 origin（世界原点在分镜里的位置）、scale（分镜单位 / 米）
// 和观察方向 az / el（度）：az 0 = 从机器人右侧看、它朝分镜 +x；az 90 = 从正前方看；el > 0 = 从上往下看。
// 分镜里的叠加标注（关节点、重心、力箭头……）用 view.project() / view.at() 算位置，和模型对得上。
import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import { XacroParser } from 'xacro-parser';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { mergeVertices, toCreasedNormals } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { MeshoptSimplifier } from 'meshoptimizer/meshopt_simplifier.module.js';

// 离屏画布每次缩到当前视图的大小再画：拷回 2D canvas 时浏览器要把整块 WebGL 缓冲读回来，2048² 的缓冲一次就要 ~80 ms
const gl = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
gl.setPixelRatio(1);
gl.setClearColor(0x000000, 0);
gl.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const hemi = new THREE.HemisphereLight(0xdfe6f2, 0x30343c, 1.6);
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
const rimLight = new THREE.DirectionalLight(0x9fc3ff, 1.4);
scene.add(hemi, keyLight, keyLight.target, rimLight, rimLight.target);
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);

/* 各台机器人的部位 → 连杆（tint / alpha 按部位上色、变淡）。左右指机器人自己的左右 */
const PARTS = {
  g1: {
    pelvis: ['pelvis', 'pelvis_contour_link'],
    torso: ['waist_yaw_link', 'waist_roll_link', 'torso_link', 'logo_link'],
    head: ['head_link', 'd435_link', 'mid360_link'],
    leftThigh: ['left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link'], leftShin: ['left_knee_link'],
    leftFoot: ['left_ankle_pitch_link', 'left_ankle_roll_link'],
    rightThigh: ['right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link'], rightShin: ['right_knee_link'],
    rightFoot: ['right_ankle_pitch_link', 'right_ankle_roll_link'],
    leftArm: ['left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
      'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand'],
    rightArm: ['right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',
      'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand']
  }
};
(function () {
  const p = PARTS.g1;
  p.leftLeg = p.leftThigh.concat(p.leftShin, p.leftFoot);
  p.rightLeg = p.rightThigh.concat(p.rightShin, p.rightFoot);
  p.body = p.pelvis.concat(p.torso, p.head);
})();

/* CSS 颜色（含 var(--x)、color-mix）→ THREE.Color：先让浏览器算出 computed color，再交给 2D context 规整成 #rrggbb */
const colorCache = new Map();
const probe = document.createElement('span');
const ctx2d = document.createElement('canvas').getContext('2d');
function cssColor(c) {
  if (colorCache.has(c)) return colorCache.get(c);
  if (!probe.isConnected) document.body.appendChild(probe);
  probe.style.color = '';
  probe.style.color = c;
  const computed = getComputedStyle(probe).color;
  ctx2d.fillStyle = '#000';
  ctx2d.fillStyle = computed;
  ctx2d.fillRect(0, 0, 1, 1);
  const d = ctx2d.getImageData(0, 0, 1, 1).data;
  const col = new THREE.Color().setRGB(d[0] / 255, d[1] / 255, d[2] / 255, THREE.SRGBColorSpace);
  colorCache.set(c, col);
  return col;
}

/* ── 网格减面 ──
   G1、Go2 各约 40 万个三角形，软件渲染（容器里没有 GPU，Chromium 走 SwiftShader）一帧要 0.3 s。加载后用 meshoptimizer
   把每个网格减到约 15%（误差上限为网格尺寸的 1%），再按 30° 折角重算法线：机械件的棱角保留，曲面仍然是光滑的。
   带多个材质分组的网格（Collada 里一个网格多种颜色）按组分别减面，再按原顺序拼回去，分组不乱。 */
const KEEP = 0.15;
function simplify(mesh) {
  let g = mesh.geometry;
  const tris = (g.index ? g.index.count : g.attributes.position.count) / 3;
  if (tris < 2000) return;
  g = g.clone();
  g.deleteAttribute('normal');
  g = mergeVertices(g, 1e-5);
  const all = g.index.array instanceof Uint32Array ? g.index.array : new Uint32Array(g.index.array);
  const groups = g.groups.length ? g.groups : [{ start: 0, count: all.length, materialIndex: 0 }];
  const parts = groups.map((gr) => {
    const sub = all.slice(gr.start, gr.start + gr.count);
    if (sub.length < 6000) return sub;
    return MeshoptSimplifier.simplify(sub, g.attributes.position.array, 3, Math.max(4500, Math.floor(sub.length * KEEP / 3) * 3), 0.01, ['LockBorder'])[0];
  });
  const idx = new Uint32Array(parts.reduce((n, p) => n + p.length, 0));
  g.clearGroups();
  let at = 0;
  parts.forEach((p, k) => { idx.set(p, at); if (groups.length > 1) g.addGroup(at, p.length, groups[k].materialIndex); at += p.length; });
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  mesh.geometry = toCreasedNormals(g, Math.PI / 6);
}

/* ── 加载 ── */
// URDF / xacro 里写的路径 → 仓库内相对路径：package://pkg/… 按 packages 映射，其余已是相对仓库的（xacro 的 workingPath 给的是仓库内目录）
function repoPath(path, packages) {
  if (path.startsWith('package://')) {
    const [pkg, ...rest] = path.slice('package://'.length).split('/');
    if (!(pkg in packages)) throw new Error('xacro: unknown package ' + pkg);
    path = packages[pkg] + '/' + rest.join('/');
  }
  const out = [];
  for (const seg of path.split('/')) { if (seg === '..') out.pop(); else if (seg && seg !== '.') out.push(seg); }
  return out.join('/');
}
// 上游只发布 xacro 的（Create 2、Jackal）照 Gallery 的 web/js/xacro.js 展开：ROS Jade 语义、<xacro:arg> 取默认值，
// $(optenv NAME 默认值) 取默认值，表达式里 Python 的 ** 换成 expr-eval 的 ^，xacro_stubs 里的包 include 进来是空 <robot/>；
// 去掉 gazebo / transmission 块，以及条件没成立、只剩空 <geometry> 的 visual / collision
async function expandXacro(text, m, root) {
  const parser = new XacroParser();
  parser.inOrder = true; parser.requirePrefix = true; parser.localProperties = true;
  parser.workingPath = m.urdf.replace(/[^/]+$/, '');
  parser.arguments = {};
  for (const [, k, v] of text.matchAll(/<(?:xacro:)?arg\s+name=["']([^"']+)["'](?:\s+default=["']([^"']*)["'])?/g)) {
    parser.arguments[k] = (v === undefined ? '' : v).replace(/^\$\(optenv\s+\S+\s*([^)]*)\)$/, '$1');
  }
  parser.rospackCommands = { find: (pkg) => `package://${pkg}`, optenv: (name, ...fallback) => fallback.join(' ') };
  const pythonPower = (src) => src.replace(/\$\{[^}]*\}/g, (expr) => expr.replace(/\*\*/g, '^'));
  const stubs = m.xacro_stubs || [];
  parser.getFileContents = async (path) => stubs.some((pkg) => path.startsWith(`package://${pkg}/`)) ? '<robot/>'
    : pythonPower(await (await fetch(root + repoPath(path, m.packages))).text());
  const doc = await parser.parse(pythonPower(text));
  for (const n of doc.querySelectorAll('gazebo, transmission')) n.remove();
  for (const g of doc.querySelectorAll('geometry')) {
    const holder = g.parentNode;
    if (!g.children.length && (holder.nodeName === 'visual' || holder.nodeName === 'collision')) holder.remove();
  }
  return new XMLSerializer().serializeToString(doc).replace(/\sxmlns=""/g, '');
}

const models = {};
async function loadRobot(id, manifest) {
  const m = manifest.robots[id];
  const root = `assets/robots/${id}/`;
  const manager = new THREE.LoadingManager();
  const done = new Promise((resolve, reject) => { manager.onLoad = resolve; manager.onError = (url) => reject(new Error('load failed: ' + url)); });
  const loader = new URDFLoader(manager);
  loader.parseCollision = false;
  // urdf-loader 自带的只认 STL / Collada，Prius 的网格是 OBJ
  const objLoader = new OBJLoader(manager), defaultLoad = loader.loadMeshCb.bind(loader);
  loader.loadMeshCb = (path, mgr, onComplete) => {
    if (/\.obj$/i.test(path)) objLoader.load(path, (obj) => onComplete(obj), undefined, (err) => onComplete(null, err));
    else defaultLoad(path, mgr, onComplete);
  };
  loader.packages = Object.fromEntries(Object.entries(m.packages).map(([k, v]) => [k, (root + v).replace(/\/+$/, '')]));
  loader.workingPath = root + m.urdf.replace(/[^/]+$/, '');
  let text = await (await fetch(root + m.urdf)).text();
  if (m.urdf.endsWith('.xacro')) text = await expandXacro(text, m, root);
  manager.itemStart('urdf');
  const robot = loader.parse(text);
  manager.itemEnd('urdf');
  await done;
  await MeshoptSimplifier.ready;
  // Collada 里自带的灯光、相机（Create 2 的网格就有）会照乱统一的打光，去掉
  const extras = [];
  robot.traverse((o) => { if (o.isLight || o.isCamera) extras.push(o); });
  extras.forEach((o) => o.removeFromParent());
  robot.traverse((o) => { if (o.isMesh) simplify(o); });
  // 统一换成 PBR 材质；原色太暗的（G1、Go2 的黑件）提亮到深灰，深色底上也看得见轮廓
  robot.traverse((o) => {
    if (!o.isMesh) return;
    const old = Array.isArray(o.material) ? o.material[0] : o.material;
    const hsl = { h: 0, s: 0, l: 0.7 };
    if (old && old.color) old.color.getHSL(hsl);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15, map: old && old.map ? old.map : null });
    mat.color.setHSL(hsl.h, hsl.s * 0.6, 0.3 + 0.62 * hsl.l);
    mat.userData.base = mat.color.clone();
    o.material = mat;
  });
  robot.updateMatrixWorld(true);
  models[id] = robot;
}

/* ── 视图 ── */
const XHTML = 'http://www.w3.org/1999/xhtml';
let viewN = 0;
function view(parent, o) {
  const template = models[o.robot];
  if (!template) throw new Error('robot not loaded: ' + o.robot);
  const robot = template.clone(true);
  robot.traverse((m) => { if (m.isMesh) { m.material = m.material.clone(); m.material.userData.base = m.material.color.clone(); } });
  const parts = PARTS[o.robot] || {};
  const st = { origin: o.origin.slice(), scale: o.scale, az: o.az || 0, el: o.el || 0, box: o.box.slice(), res: o.res || 2 };
  const NS = 'http://www.w3.org/2000/svg';
  // 不可见的框：stage.html 的 fitStory 按 rect/path… 量分镜范围，量不到 canvas，这个框跟着模型的投影外框走
  const hit = document.createElementNS(NS, 'rect');
  hit.setAttribute('fill', 'none'); hit.setAttribute('stroke', 'none');
  const fo = document.createElementNS(NS, 'foreignObject');
  const canvas = document.createElementNS(XHTML, 'canvas');
  canvas.style.display = 'block';
  fo.appendChild(canvas);
  parent.appendChild(hit);
  parent.appendChild(fo);
  const ctx = canvas.getContext('2d');
  const id = ++viewN;
  let lastKey = '';
  const tints = new Map();     // 连杆名 → [颜色, 强度, 不透明度]
  const basis = { d: new THREE.Vector3(), r: new THREE.Vector3(), u: new THREE.Vector3() };

  function linksOf(names) {
    if (names === 'all') return Object.keys(robot.links);
    return [].concat(...[].concat(names).map((n) => parts[n] || [n]));
  }
  function meshesOf(link) {
    const out = [];
    (function walk(n) { n.children.forEach((c) => { if (c.isURDFJoint || c.isURDFLink) return; if (c.isMesh) out.push(c); walk(c); }); })(robot.links[link]);
    return out;
  }
  function updateBasis() {
    const az = st.az * Math.PI / 180, el = st.el * Math.PI / 180;
    basis.d.set(-Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), -Math.sin(el));
    basis.r.crossVectors(basis.d, new THREE.Vector3(0, 0, 1)).normalize();
    basis.u.crossVectors(basis.r, basis.d).normalize();
  }
  updateBasis();
  function project(p) {
    const v = p.isVector3 ? p : new THREE.Vector3(p[0], p[1], p[2]);
    return [st.origin[0] + st.scale * v.dot(basis.r), st.origin[1] - st.scale * v.dot(basis.u)];
  }
  function layout() {
    const [bx, by, bw, bh] = st.box;
    fo.setAttribute('x', bx); fo.setAttribute('y', by); fo.setAttribute('width', bw); fo.setAttribute('height', bh);
    canvas.style.width = bw + 'px'; canvas.style.height = bh + 'px';
    const W = Math.round(bw * st.res), H = Math.round(bh * st.res);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  }
  layout();

  const api = {
    robot, fo, canvas, hit, parts,
    joints(values) { for (const k in values) if (robot.joints[k]) robot.joints[k].setJointValue(values[k]); return api; },
    jointValue(name) { return robot.joints[name] ? robot.joints[name].angle : 0; },
    // 机器人根（URDF 的根连杆）在世界里的位置（米）与 rpy（弧度）
    base(pos, rpy) {
      robot.position.set(pos[0], pos[1], pos[2]);
      if (rpy) robot.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');
      return api;
    },
    camera(c) { Object.assign(st, c); if (c.az !== undefined || c.el !== undefined) updateBasis(); if (c.box) layout(); return api; },
    project,
    // 连杆 name 上一点（连杆坐标系里的 local，米）的世界坐标
    world(name, local) {
      robot.updateMatrixWorld(true);
      const l = robot.links[name] || robot.frames && robot.frames[name];
      const v = new THREE.Vector3(...(local || [0, 0, 0]));
      return l ? v.applyMatrix4(l.matrixWorld) : v;
    },
    at(name, local) { return project(api.world(name, local)); },
    // 按部位上色：color 为 CSS 颜色（null 复原），amount 0…1；opacity 让这部分变淡
    tint(names, color, amount, opacity) {
      linksOf(names).forEach((n) => {
        if (!color && (opacity === undefined || opacity >= 1)) tints.delete(n);
        else tints.set(n, [color, amount === undefined ? 1 : amount, opacity === undefined ? 1 : opacity]);
      });
      return api;
    },
    clearTint() { tints.clear(); return api; },
    // 整台在世界里的外框（米）
    box3() { robot.updateMatrixWorld(true); return new THREE.Box3().setFromObject(robot); },
    render() {
      robot.updateMatrixWorld(true);
      const key = JSON.stringify([st, [...tints], robot.matrixWorld.elements, Object.values(robot.joints).map((j) => j.angle)]);
      if (key === lastKey) return api;
      lastKey = key;
      // 材质
      robot.traverse((m) => {
        if (!m.isMesh) return;
        m.material.color.copy(m.material.userData.base);
        m.material.emissive.setRGB(0, 0, 0);
        m.material.opacity = 1; m.material.transparent = false; m.material.depthWrite = true; m.visible = true;
      });
      tints.forEach(([color, amount, opacity], link) => {
        if (!robot.links[link]) return;
        meshesOf(link).forEach((m) => {
          if (color) {
            const c = cssColor(color);
            m.material.color.lerp(c, 0.75 * amount);
            m.material.emissive.copy(c).multiplyScalar(0.45 * amount);
          }
          if (opacity <= 0.001) m.visible = false;
          else if (opacity < 1) { m.material.opacity = opacity; m.material.transparent = true; m.material.depthWrite = false; }
        });
      });
      // 相机：正交，视景框就是 st.box 在相机平面上的范围
      const [bx, by, bw, bh] = st.box, S = st.scale, [ox, oy] = st.origin;
      cam.left = (bx - ox) / S; cam.right = (bx + bw - ox) / S;
      cam.top = (oy - by) / S; cam.bottom = (oy - by - bh) / S;
      cam.position.copy(basis.d).multiplyScalar(-20);
      cam.up.copy(basis.u);
      cam.lookAt(0, 0, 0);
      cam.updateProjectionMatrix();
      // 主光从相机左上前方打过来，轮廓光从背后右上
      keyLight.position.copy(cam.position).addScaledVector(basis.r, -12).addScaledVector(basis.u, 14);
      rimLight.position.copy(basis.d).multiplyScalar(15).addScaledVector(basis.r, 8).addScaledVector(basis.u, 10);
      const W = canvas.width, H = canvas.height;
      if (gl.domElement.width !== W || gl.domElement.height !== H) gl.setSize(W, H, false);
      scene.add(robot);
      gl.render(scene, cam);
      scene.remove(robot);
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(gl.domElement, 0, 0);
      // 不可见外框：模型外框八个角投影后的范围
      const b = new THREE.Box3().setFromObject(robot);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i < 8; i++) {
        const p = project(new THREE.Vector3(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z));
        x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
      }
      x0 = Math.max(x0, bx); y0 = Math.max(y0, by); x1 = Math.min(x1, bx + bw); y1 = Math.min(y1, by + bh);
      hit.setAttribute('x', x0.toFixed(1)); hit.setAttribute('y', y0.toFixed(1));
      hit.setAttribute('width', Math.max(0, x1 - x0).toFixed(1)); hit.setAttribute('height', Math.max(0, y1 - y0).toFixed(1));
      return api;
    }
  };
  api.id = id;
  live.add(api);
  return api;
}

// 机器人在零位、站直时的几何量（米）：脚底离根的高度等，各集按需取
function measure(id) {
  const r = models[id];
  r.updateMatrixWorld(true);
  return new THREE.Box3().setFromObject(r);
}

/* ══ 分镜用的机器人 ══
   和各集 00-common.js 里手画版同一套接口（E.robot 侧视人形、E.front 正面人形、E.arm 机械臂、E.quadruped 四足、E.humanoid 小图标；E.vacuum 扫地机、E.car 汽车、E.wheeled 轮式底盘），
   各集只把这几个函数换成下面的版本，分镜里的动作、叠加标注照旧。
   手画版里被分镜改样式的部件（躯干、大腿、脚……）在这里是不显示的 <g>「把手」：分镜照样设它的 stroke / fill / filter / opacity，
   画之前按这些样式给对应连杆上色（发光 = 上色更重）、变淡；关节圆点、摄像头、IMU 这类标注仍是 svg，位置按正运动学算。 */
const SVGNS = 'http://www.w3.org/2000/svg', DEG = Math.PI / 180;
function sel(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
// 「没上色」的样式：空、正文色、次要色、静音色（手画版的默认色）
const PLAIN = new Set(['', 'var(--text)', 'var(--text-secondary)', 'var(--demo-muted)']);
function handleColor(h) { const c = (h.style.stroke || h.style.fill || '').trim(); return PLAIN.has(c) ? null : c; }
function handleOpacity(h) { return h.style.opacity === '' ? 1 : +h.style.opacity; }
function applyHandle(v, h, parts) {
  const c = handleColor(h), o = handleOpacity(h);
  v.tint(parts, c, h.style.filter ? 1 : 0.8, o);
}
// 叠加标注：没上色时不显示（手画版里它们是火柴人自己的一部分；3D 模型上只在被点名时出现）
function showIfColored(n) { n.style.visibility = handleColor(n) ? 'visible' : 'hidden'; }

/* 全身重心（世界坐标）：各连杆 <inertial> 的质量与质心 */
function centerOfMass(robot) {
  robot.updateMatrixWorld(true);
  const sum = new THREE.Vector3(), p = new THREE.Vector3();
  let m = 0;
  Object.values(robot.links).forEach((l) => {
    const n = l.urdfNode && l.urdfNode.querySelector(':scope > inertial');
    if (!n) return;
    const mass = +n.querySelector('mass').getAttribute('value');
    const o = n.querySelector('origin');
    const xyz = o && o.getAttribute('xyz') ? o.getAttribute('xyz').trim().split(/\s+/).map(Number) : [0, 0, 0];
    p.set(xyz[0], xyz[1], xyz[2]).applyMatrix4(l.matrixWorld);
    sum.addScaledVector(p, mass);
    m += mass;
  });
  return sum.multiplyScalar(1 / m);
}

/* G1 零位几何：脚底高度、右腿髋 / 膝 / 踝（矢状面里大腿、小腿两段向量）、头顶高度 */
let g1G = null;
function g1Geom() {
  if (g1G) return g1G;
  const r = models.g1;
  r.updateMatrixWorld(true);
  const pos = (n) => new THREE.Vector3().setFromMatrixPosition(r.links[n].matrixWorld);
  const box = new THREE.Box3().setFromObject(r);
  const hip = pos('right_hip_pitch_link'), knee = pos('right_knee_link'), ankle = pos('right_ankle_pitch_link');
  const foot = new THREE.Box3().setFromObject(r.links.right_ankle_roll_link);
  g1G = { sole: box.min.z, top: box.max.z, hip, T: [knee.x - hip.x, knee.z - hip.z], Sv: [ankle.x - knee.x, ankle.z - knee.z],
    ankleH: ankle.z - box.min.z, foot };
  return g1G;
}
// 绕 y 轴转 q（urdf 的俯仰）作用在矢状面向量 (x, z) 上
function rotY(v, q) { return [v[0] * Math.cos(q) + v[1] * Math.sin(q), -v[0] * Math.sin(q) + v[1] * Math.cos(q)]; }

/* ── 侧视 G1（替换 E.robot）：本地坐标与手画版一致——朝 +x、脚底 y = 388、右髋在 (0, 302)，比例 = 86 / 髋高（≈ 125 / 米），
   膝盖正好落在 (0, 344)。近侧是机器人的右侧。legs / arm 的角度沿用手画版的定义（度，0 = 竖直向下，正值朝 +x）：
   髋俯仰 = −大腿角，膝 = 大腿角 − 小腿角，踝俯仰让脚掌放平；legIK 用 G1 真实的大腿 / 小腿长度反解。 ── */
// opts.box / opts.res：只画一块（本地坐标）、按放大倍数提高分辨率——001 第 4 幕的腿部特写用；opts.fadeTop：上缘渐隐，特写不硬切；
// opts.hideArms：不画两臂（特写里垂下来的手会压在髋上）
function g1Side(parent, X, opts) {
  const o = opts || {}, G = g1Geom(), S = 86 / (G.hip.z - G.sole);
  const outer = sel('g', {}, parent), body = sel('g', {}, outer);
  const v = view(body, { robot: 'g1', origin: [0, 388 + G.sole * S], scale: S, az: 0, box: o.box || [-100, 140, 220, 262], res: o.res || 2 });
  if (o.fadeTop) v.canvas.style.maskImage = 'linear-gradient(to bottom, transparent 0%, #000 28%)';
  const handle = () => sel('g', {}, body);
  const torso = handle(), head = handle(), neck = handle(), thigh = handle(), shin = handle(), foot = handle(), upper = handle(), fore = handle();
  const pelvis = handle();   // 只有骨盆（G1 的根连杆 = 浮动基）；torso 把手是骨盆 + 腰 + 胸腔
  const far = handle();
  const ov = sel('g', {}, body);
  const cam = sel('rect', { width: 12, height: 8, rx: 2 }, ov);
  const imu = sel('rect', { width: 10, height: 10, rx: 2 }, ov);
  const dot = () => sel('circle', { r: 5 }, ov);
  const hipDot = dot(), kneeDot = dot(), ankleDot = dot();
  [cam, imu, hipDot, kneeDot, ankleDot].forEach((n) => { n.style.fill = 'var(--text)'; });
  let legA = [4, 0], legB = [-4, 0], armA = [8, 4], tiltA = 0, tiltB = 0;
  const state = { x: X || 0, rot: 0, pivot: [0, 388], scale: 1, dy: 0 };
  function apply() {
    const leg = (side, a, tilt) => {
      const q1 = -a[0] * DEG, q2 = Math.max(-0.08, (a[0] - a[1]) * DEG);
      v.joints({ [side + '_hip_pitch_joint']: q1, [side + '_knee_joint']: q2, [side + '_ankle_pitch_joint']: -(q1 + q2) + (tilt || 0) * DEG });
    };
    leg('right', legA, tiltA); leg('left', legB, tiltB);
    const arm = (side, s, e) => v.joints({ [side + '_shoulder_pitch_joint']: -s * DEG, [side + '_elbow_joint']: Math.min(2.09, (90 - (e - s)) * DEG) });
    arm('right', armA[0], armA[1]);
    arm('left', 16 - armA[0], 16 - armA[0] + (armA[1] - armA[0]));   // 远侧手臂和近侧对着摆（绕站姿 8° 镜像）
    v.joints({ left_shoulder_roll_joint: 0.12, right_shoulder_roll_joint: -0.12 });
  }
  function local(name, off) { return v.at(name, off); }
  const P = {};
  function pts() {
    P.hip = local('right_hip_pitch_link'); P.knee = local('right_knee_link'); P.ankle = local('right_ankle_pitch_link');
    P.cam = local('d435_link'); P.imu = local('imu_in_pelvis'); P.shoulder = local('right_shoulder_pitch_link');
    P.pelvis = local('pelvis');
    return P;
  }
  function place(o) {
    for (const k in o) state[k] = o[k];
    outer.setAttribute('transform', 'translate(' + state.x.toFixed(2) + ' ' + state.dy.toFixed(2) + ') translate(0 388) scale(' + state.scale.toFixed(4) + ') translate(0 -388)');
    body.setAttribute('transform', state.rot ? 'rotate(' + state.rot.toFixed(2) + ' ' + state.pivot[0] + ' ' + state.pivot[1] + ')' : '');
  }
  v.pre = () => {
    pts();
    [[cam, P.cam, 6, 4], [imu, P.imu, 5, 5]].forEach(([n, p, hx, hy]) => { n.setAttribute('x', (p[0] - hx).toFixed(1)); n.setAttribute('y', (p[1] - hy).toFixed(1)); showIfColored(n); });
    [[hipDot, P.hip], [kneeDot, P.knee], [ankleDot, P.ankle]].forEach(([n, p]) => { n.setAttribute('cx', p[0].toFixed(1)); n.setAttribute('cy', p[1].toFixed(1)); showIfColored(n); });
    v.clearTint();
    applyHandle(v, torso, ['pelvis', 'torso']); applyHandle(v, head, 'head');
    if (handleColor(pelvis) || handleOpacity(pelvis) < 1) applyHandle(v, pelvis, 'pelvis');   // 没上色时别把 torso 给骨盆的颜色抹掉
    applyHandle(v, thigh, 'rightThigh'); applyHandle(v, shin, 'rightShin'); applyHandle(v, foot, 'rightFoot');
    applyHandle(v, upper, 'rightArm'); applyHandle(v, fore, 'rightArm');
    v.tint('leftLeg', handleColor(far), 0.8, handleOpacity(far));
    if (o.hideArms) v.tint(['leftArm', 'rightArm'], null, 1, 0);
  };
  apply(); place({});
  return {
    g: outer, body, view: v, head, cam, imu, torso, pelvis, neck, thigh, shin, foot, far, upper, fore, hipDot, kneeDot, ankleDot,
    nearLeg: [thigh, shin, foot, hipDot, kneeDot, ankleDot],
    place, state, scale: S, ankleY: 388 - G.ankleH * S,
    // tilt / tiltFar：近侧 / 远侧脚掌在放平之外再转多少度（正值脚尖朝下），踩到不平、抬脚跟、落地翘脚尖时用
    legs(near, farA, tilt, tiltFar) { legA = near; legB = farA || legB; tiltA = tilt || 0; tiltB = tiltFar || 0; apply(); },
    arm(a) { armA = a; apply(); },
    stand() { legA = [4, 0]; legB = [-4, 0]; armA = [8, 4]; tiltA = 0; tiltB = 0; apply(); },
    // 走路时近侧手臂的 [大臂角, 小臂角]：绕站姿的 8° 前后各摆 17.6°，和近腿反相。远侧手臂在 apply() 里绕 8° 镜像，
    // 所以两只手摆幅一样；手画版 poseWalk 的手臂绕 −8° 摆，直接喂进来会一只手往前多摆、一只往后多摆
    walkArm(ph) { const a = 8 - 17.6 * Math.sin(ph * 2 * Math.PI); return [a, a + 19.2]; },
    walk(ph) { const w = K().poseWalk(ph); legA = w.legA; legB = w.legB; armA = this.walkArm(ph); apply(); },
    pts,
    // 给髋、踝的本地坐标，返回 [大腿角, 小腿角]（度），喂给 legs() 正好让踝落在那里
    legIK(hip, ankle) {
      const D = [(ankle[0] - 0) / S, -(ankle[1] - hip[1]) / S], d = Math.hypot(D[0], D[1]);
      const len = (q2) => { const s2 = rotY(G.Sv, q2); return Math.hypot(G.T[0] + s2[0], G.T[1] + s2[1]); };
      let lo = 0, hi = 2.4;
      if (len(lo) <= d) hi = lo;
      for (let i = 0; i < 40 && hi > lo; i++) { const m = (lo + hi) / 2; if (len(m) > d) lo = m; else hi = m; }
      const q2 = (lo + hi) / 2, s2 = rotY(G.Sv, q2), V = [G.T[0] + s2[0], G.T[1] + s2[1]];
      const q1 = Math.atan2(V[1], V[0]) - Math.atan2(D[1], D[0]);
      const a = -q1 / DEG;
      return [a, a - q2 / DEG];
    },
    legStyle(c, dash, opacity) {
      const o = opacity === undefined ? 1 : opacity;
      [thigh, shin, foot].forEach((h) => { h.style.stroke = c || ''; h.style.opacity = o; });
      [hipDot, kneeDot, ankleDot].forEach((d) => { d.style.fill = c || 'var(--text)'; d.style.opacity = o; });
      far.style.opacity = o;
    }
  };
}

/* ── 正面 G1（替换 E.front）：本地坐标——两脚中间 x = 0，脚底 y = 388，头顶 y = 92（比例 ≈ 224 / 米）。画面右边是机器人的左边。
   pose({ shift, lean, lift: [画面左脚, 画面右脚], arm: [[外张角, 小臂角], [..]], tilt, pivot })：
   shift 用两腿髋 / 踝侧摆（平行四边形）把骨盆横移、两脚不动；lift 屈髋屈膝抬脚、脚掌放平；lean 是腰的侧倾；
   外张角 ≤ 100° 用肩侧摆，更大（举过头顶）改用肩俯仰；tilt 整个人绕 pivot 转（svg 旋转，正面看就是绕前后轴的侧倒）。
   joints：29 个转动关节的圆点（svg，opts.dots 才显示），按腿 / 腰 / 胳膊分组。com() 用各连杆的质量与质心算真实重心。
   opts.az / opts.el 换成斜视（和分镜里斜着画的桌子、地面网格同一个相机）：这时 tilt 是真的三维滚转——绕 pivot[0]（本地 x，
   即机器人 +y 方向离两脚中间多远）那条贴地、沿机器人前后方向的轴转，svg 不再旋转。斜视时 proj([x, y, z]) 把机器人坐标系里的
   三维点（米，地面 z = ground）投到分镜坐标，com3() 是三维重心，sole(side) 是一只脚掌贴地的四个角。 ── */
function g1Front(parent, opts) {
  const o = opts || {}, G = g1Geom(), S = 296 / (G.top - G.sole);
  const az = o.az === undefined ? 90 : o.az, el = o.el || 0, oblique = az !== 90 || el !== 0;
  const outer = sel('g', {}, parent), body = sel('g', {}, outer);
  // 原点放在让「两脚中间的地面点」投到本地 (0, 388)：斜视时地面高度要乘 cos(el)
  const v = view(body, { robot: 'g1', origin: [0, 388 + G.sole * S * Math.cos(el * DEG)], scale: S, az, el,
    box: o.box || [-150, 0, 300, 402], res: o.res || 2 });
  const ov = sel('g', {}, body);
  const names = Object.keys(v.robot.joints).filter((n) => v.robot.joints[n].jointType === 'revolute');
  const groupOf = (n) => (/hip|knee|ankle/.test(n) ? 'legs' : /waist/.test(n) ? 'waist' : 'arms');
  const joints = names.map((n) => {
    const c = sel('circle', { r: 6, 'stroke-width': 2 }, ov);
    c.style.fill = 'var(--text)'; c.style.stroke = 'var(--demo-surface)';
    return { name: n, group: groupOf(n), el: c };
  });
  const groups = { legs: [], waist: [], arms: [] };
  joints.forEach((j) => groups[j.group].push(j.el));
  const all = groups.legs.concat(groups.waist, groups.arms);
  ov.style.display = o.dots ? '' : 'none';
  const st = { shift: 0, lean: 0, lift: [0, 0], arm: [[8, 4], [8, 4]], tilt: 0, pivot: [0, 388] };
  let X = 0, SC = 1, GY = 388;                           // GY：两脚中间的地面点放到画面里的 y
  const L = -G.T[1] - G.Sv[1];                           // 髋到踝的竖直距离（米）
  function apply() {
    const dy = st.shift / S, th = Math.asin(Math.max(-0.25, Math.min(0.25, dy / L)));
    const j = {};
    // 正值髋侧摆让脚往 +y（机器人左边）去：骨盆往 +y 挪 = 脚相对髋往 −y，两腿同一角，踝反过来摆，脚掌仍平
    ['right', 'left'].forEach((side, k) => {
      const lift = st.lift[k] / S, ph = Math.acos(Math.max(0.35, 1 - lift / L));
      j[side + '_hip_roll_joint'] = -th; j[side + '_ankle_roll_joint'] = th;
      j[side + '_hip_pitch_joint'] = -ph; j[side + '_knee_joint'] = 2 * ph; j[side + '_ankle_pitch_joint'] = -ph;
      const a = st.arm[k], sg = side === 'right' ? -1 : 1;
      if (a[0] <= 100) {
        j[side + '_shoulder_roll_joint'] = sg * Math.min(a[0] + 0.5 * Math.max(0, a[1] - a[0]), 120) * DEG;
        j[side + '_shoulder_pitch_joint'] = 0; j[side + '_elbow_joint'] = 1.45;
      } else {
        j[side + '_shoulder_roll_joint'] = sg * 10 * DEG; j[side + '_shoulder_pitch_joint'] = -a[0] * DEG; j[side + '_elbow_joint'] = 1.45;
      }
    });
    j.waist_roll_joint = st.lean * DEG;
    v.joints(j);
    if (oblique) {
      // 往画面右边倒 = 往机器人左边（+y）倒：绕 +x 轴转负角，根连杆的位置也绕支点那条轴转过去
      const a = -st.tilt * DEG, P = new THREE.Vector3(0, st.pivot[0] / S, G.sole);
      const q = new THREE.Vector3(0, dy, -L * (1 - Math.cos(th))).sub(P).applyAxisAngle(new THREE.Vector3(1, 0, 0), a).add(P);
      v.base([q.x, q.y, q.z], [a, 0, 0]);
      return;
    }
    v.base([0, dy, -L * (1 - Math.cos(th))]);
    body.setAttribute('transform', st.tilt ? 'rotate(' + st.tilt.toFixed(2) + ' ' + st.pivot[0] + ' ' + st.pivot[1] + ')' : '');
  }
  function world(p) {
    if (oblique) return [X + p[0] * SC, GY + (p[1] - 388) * SC];
    const a = st.tilt * DEG, px = p[0] - st.pivot[0], py = p[1] - st.pivot[1];
    const q = [st.pivot[0] + px * Math.cos(a) - py * Math.sin(a), st.pivot[1] + px * Math.sin(a) + py * Math.cos(a)];
    return [X + q[0] * SC, GY + (q[1] - 388) * SC];   // 和 place() 的变换一致：两脚中间的地面点在 (X, GY)
  }
  v.pre = () => {
    joints.forEach((jt) => {
      const p = v.at(jt.name.replace(/_joint$/, '_link'));
      jt.el.setAttribute('cx', p[0].toFixed(1)); jt.el.setAttribute('cy', p[1].toFixed(1));
    });
  };
  // 脚：画面里的左右（本地 x）范围
  const fb = G.foot;
  const footSpan = [[S * fb.min.y, S * fb.max.y], [-S * fb.max.y, -S * fb.min.y]];   // [画面左脚（机器人右脚）, 画面右脚]
  apply();
  return {
    g: outer, body, view: v, joints, groups, all, footSpan, scale: S,
    // place(x, s, gy)：两脚中间的地面点放到 (x, gy)（gy 默认 388），以它为心缩放 s
    place(x, s, gy) {
      X = x; SC = s || 1; GY = gy === undefined ? 388 : gy;
      outer.setAttribute('transform', 'translate(' + X + ' ' + (GY - 388) + ') translate(0 388) scale(' + SC + ') translate(0 -388)');
    },
    pose(p) { for (const k in p) st[k] = p[k]; apply(); },
    reset() { Object.assign(st, { shift: 0, lean: 0, lift: [0, 0], arm: [[8, 4], [8, 4]], tilt: 0, pivot: [0, 388] }); apply(); },
    com() { return v.project(centerOfMass(v.robot)); },
    com3() { return centerOfMass(v.robot); },
    proj(p) { return world(v.project(p)); },
    ground: G.sole,
    // 一只脚掌贴地的四个角（机器人坐标系，米）：'right' / 'left'，零位站立时的脚掌外框
    sole(side) {
      const sg = side === 'right' ? 1 : -1, y0 = sg * fb.min.y, y1 = sg * fb.max.y;
      return [[fb.max.x, y0, G.sole], [fb.max.x, y1, G.sole], [fb.min.x, y1, G.sole], [fb.min.x, y0, G.sole]];
    },
    world,
    locate(el) { return world([+el.getAttribute('cx'), +el.getAttribute('cy')]); },
    tint(part, color, amount, opacity) { v.tint(part, color, amount, opacity); },
    showDots(on) { ov.style.display = on ? '' : 'none'; }
  };
}

/* ── UR5（替换 E.arm）：底座底面中心在 (x, y)，s = 1 时大臂 + 小臂长 118（同手画版 64 + 54）。
   set(a1, a2)：a1 大臂离竖直的角、正值朝 +x，a2 小臂相对大臂再往 +x 弯的角（度）；手腕跟着转，让工具头一直朝前下方。
   返回工具头的画面坐标。joints() 返回 6 个转动关节在画面里的位置（底座转台、肩、肘、腕 1–3）。 ── */
function ur5(parent, x, y, s, o) {
  const opt = o || {}, S = 118 * (s || 1) / 0.817;
  const g = sel('g', {}, parent);
  const v = view(g, { robot: 'ur5', origin: [x, y], scale: S, az: opt.az === undefined ? 22 : opt.az, el: opt.el === undefined ? 10 : opt.el,
    box: [x - 0.42 * S, y - 1.08 * S, 1.45 * S, 1.2 * S], res: opt.res || 2.4 });
  const LINKS = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'];
  function set(a1, a2) {
    const lift = (a1 - 90) * DEG, elbow = a2 * DEG;
    v.joints({ shoulder_pan_joint: 0, shoulder_lift_joint: lift, elbow_joint: elbow, wrist_1_joint: -(lift + elbow) - Math.PI / 2, wrist_2_joint: -Math.PI / 2, wrist_3_joint: 0 });
    return v.at('tool0');
  }
  set(25, 60);
  return { g, view: v, set, joints: () => LINKS.map((l) => v.at(l)), tool: () => v.at('tool0') };
}

/* ── Go2（替换 E.quadruped）：站姿，脚底在 y、身长 ≈ 110·s；从右前方斜看。 ── */
function go2(parent, x, y, s, o) {
  const opt = o || {}, S = 160 * (s || 1);
  const g = sel('g', {}, parent);
  const v = view(g, { robot: 'go2', origin: [x, y], scale: S, az: opt.az === undefined ? 18 : opt.az, el: opt.el === undefined ? 10 : opt.el,
    box: [x - 0.5 * S, y - 0.62 * S, 1.0 * S, 0.7 * S], res: opt.res || 2.4 });
  const stand = {};
  ['FL', 'FR', 'RL', 'RR'].forEach((l) => { stand[l + '_thigh_joint'] = 0.8; stand[l + '_calf_joint'] = -1.5; });
  v.joints(stand);
  const b = v.box3();
  v.base([-(b.min.x + b.max.x) / 2, 0, -b.min.z]);
  return { g, view: v };
}

/* ── 贴地的整台（Create 2 扫地机、Jackal 轮式底盘、Prius 汽车）：外框中心挪到原点、底面贴地，
   (x, y) 是它底下地面中心在画面里的位置，w 是外框投影后的画面宽度。top：从正上方看（扫地机俯视），车头朝 +x，(x, y) 为圆心。
   spin(圈数)：转轮子关节（只在车真的往前走时按走过的距离转）。 ── */
const WHEELS = {
  create2: ['left_wheel_joint', 'right_wheel_joint'],
  jackal: ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel'],
  prius: ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']
};
function grounded(robot, parent, x, y, w, o, az, el) {
  const opt = o || {};
  if (opt.az !== undefined) az = opt.az;
  if (opt.el !== undefined) el = opt.el;
  if (opt.top) { az = 0; el = 89.9; }
  const b = measure(robot), dx = b.max.x - b.min.x, dy = b.max.y - b.min.y, dz = b.max.z - b.min.z;
  const ca = Math.abs(Math.cos(az * DEG)), sa = Math.abs(Math.sin(az * DEG)), ce = Math.cos(el * DEG), se = Math.sin(el * DEG);
  const S = w / (dx * ca + dy * sa), half = (dx * sa + dy * ca) * se / 2;   // 地面外框投影后在原点上下各伸出 half
  const g = sel('g', {}, parent);
  const v = view(g, { robot, origin: [x, y], scale: S, az, el, res: opt.res || 2.4,
    box: [x - w / 2 - 4, y - (dz * ce + half) * S - 4, w + 8, (dz * ce + 2 * half) * S + 8] });
  v.base([-(b.min.x + b.max.x) / 2, -(b.min.y + b.max.y) / 2, -b.min.z]);
  function spin(turns) {
    const j = {};
    WHEELS[robot].forEach((n) => { j[n] = turns * 2 * Math.PI; });
    v.joints(j);
  }
  return { g, view: v, spin };
}
function create2(parent, x, y, s, o) { return grounded('create2', parent, x, y, 92 * (s || 1), o, 30, 25); }
function jackal(parent, x, y, s, o) { return grounded('jackal', parent, x, y, 104 * (s || 1), o, 25, 12); }
function prius(parent, x, y, s, o) { return grounded('prius', parent, x, y, 156 * (s || 1), o, 20, 10); }

/* ── 小人形图标（替换 E.humanoid）：G1 从右前方斜看，脚底在 (x, y)，高 ≈ 113·s（同手画版火柴人）。 ── */
function g1Icon(parent, x, y, s, o) {
  const opt = o || {}, G = g1Geom(), S = 113 * (s || 1) / (G.top - G.sole);
  const g = sel('g', {}, parent);
  const v = view(g, { robot: 'g1', origin: [x, y + G.sole * S], scale: S, az: opt.az === undefined ? 62 : opt.az, el: opt.el === undefined ? 6 : opt.el,
    box: [x - 0.45 * S, y - (G.top - G.sole) * S - 6, 0.9 * S, (G.top - G.sole) * S + 12], res: opt.res || 2.4 });
  function pose(w) {
    v.joints({ left_shoulder_roll_joint: 0.14, right_shoulder_roll_joint: -0.14, left_elbow_joint: 1.35, right_elbow_joint: 1.35,
      left_shoulder_pitch_joint: 0.1 + 0.12 * w, right_shoulder_pitch_joint: 0.1 - 0.12 * w, waist_yaw_joint: 0.06 * w });
  }
  pose(0);
  return {
    g, view: v,
    stand() { pose(0); },
    // 站着轻晃：两臂小幅前后摆、腰微转（不原地迈步）
    idle(t) { pose(Math.sin(2 * Math.PI * t / 3.2)); },
    walk(ph) { pose(Math.sin(2 * Math.PI * ph)); }
  };
}

function K() { return window.PaperDemoKit; }

/* 每帧画之前（stage.html 在分镜 draw 之后调）：只重画挂在页面上、姿态或上色变了的视图 */
const live = new Set();
function flush() {
  live.forEach((v) => {
    if (!v.fo.isConnected) return;
    if (v.pre) v.pre();
    v.render();
  });
}

window.Robot3D = { view, models, measure, flush, centerOfMass, g1Side, g1Front, ur5, go2, g1Icon, create2, jackal, prius, THREE };
// render.mjs / check_layout.mjs 打开舞台后先等这个：这一集要用的机器人（EpisodeVideo.robots）都加载完
window.robotsReady = (async () => {
  const want = (window.EpisodeVideo && window.EpisodeVideo.robots) || [];
  if (!want.length) return;
  const manifest = await (await fetch('robots.json')).json();
  await Promise.all(want.map((id) => loadRobot(id, manifest)));
})();
