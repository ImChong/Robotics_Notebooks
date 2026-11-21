#!/usr/bin/env node
/**
 * 导出期预计算 2D 图谱布局：用 vendor d3 + 与 graph.html 共用的力参数（docs/graph-force-params.js）
 * 把全图跑到 alphaMin，输出每个节点相对画布中心的整数坐标（按 nodes 顺序）。
 * 首屏直接用这份坐标落稳，不再在手机上跑几十秒的 warmup / 收敛。
 *
 * 用法：node scripts/precompute_graph_layout.cjs exports/link-graph.json  → stdout: JSON 数组 [[x,y],...]
 * 由 scripts/generate_link_graph.py 调用并写回 link-graph.json 的 layout 字段。
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const d3 = require(path.join(ROOT, 'docs', 'vendor', 'd3.min.js'));
const force = require(path.join(ROOT, 'docs', 'graph-force-params.js'));

// graph-node-size.js 只挂 window，用 vm 取出同一份半径标尺
const sizeCtx = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(ROOT, 'docs', 'graph-node-size.js'), 'utf8'), sizeCtx);
const nodeSize = sizeCtx.window.RNGraphNodeSize;

const inPath = process.argv[2] || path.join(ROOT, 'exports', 'link-graph.json');
const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
const ids = data.nodes.map(n => n.id);
// edge_format=index：[sIdx, tIdx]；旧格式对象边原样
const edges = (data.edges || []).map(e => Array.isArray(e)
  ? { source: ids[e[0]], target: ids[e[1]] }
  : { source: e.source, target: e.target }).filter(e => e.source !== undefined && e.target !== undefined);

const degreeMap = nodeSize.computeDegreeMap(edges);
const maxDegree = nodeSize.maxDegreeOf(degreeMap);
const nodes = data.nodes.map(n => ({ id: n.id, _degree: degreeMap[n.id] || 0 }));
const radiusOf = d => nodeSize.radiusForDegree(d._degree, maxDegree);   // nodeScale = 1

/**
 * 2D 节点防重叠解算（零分配空间哈希 + 对称阻尼 Jacobi 松弛），与 docs/graph.html 同参，
 * 在导出期一次性跑完并随 layout 写出，避免前端首屏再做全图多轮防重叠计算。
 */
function resolveNodeOverlaps2D(list, radii, opts) {
  const n = list.length;
  if (n < 2) return 0;
  const cfg = opts || {};
  const pad = cfg.padding != null ? cfg.padding : 4;
  const maxPasses = cfg.maxPasses != null ? cfg.maxPasses : 18;
  let maxR = 4;
  for (let i = 0; i < n; i++) {
    if (radii[i] > maxR) maxR = radii[i];
  }
  const cellSize = Math.max(16, maxR * 2 + pad * 1.5);
  const invCell = 1 / cellSize;
  let tableSize = 1024;
  while (tableSize < n * 4) tableSize <<= 1;
  const mask = tableSize - 1;
  const head = new Int32Array(tableSize);
  const next = new Int32Array(n);
  const cellX = new Int32Array(n);
  const cellY = new Int32Array(n);
  const accX = new Float32Array(n);
  const accY = new Float32Array(n);
  const accW = new Float32Array(n);
  let totalResolved = 0;

  for (let pass = 0; pass < maxPasses; pass++) {
    head.fill(-1);
    for (let i = 0; i < n; i++) {
      accX[i] = 0;
      accY[i] = 0;
      accW[i] = 0;
      const nd = list[i];
      if (nd.x == null || nd.y == null) {
        cellX[i] = 0x7fffffff;
        cellY[i] = 0x7fffffff;
        continue;
      }
      const gx = Math.floor(nd.x * invCell) | 0;
      const gy = Math.floor(nd.y * invCell) | 0;
      cellX[i] = gx;
      cellY[i] = gy;
      const h = (Math.imul(gx, 73856093) ^ Math.imul(gy, 19349663)) & mask;
      next[i] = head[h];
      head[h] = i;
    }

    let passOverlaps = 0;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      if (a.x == null || a.y == null) continue;
      const ra = radii[i];
      const gxa = cellX[i];
      const gya = cellY[i];
      const invMa = 1 / Math.max(1, ra * ra);

      for (let oy = -1; oy <= 1; oy++) {
        const qy = (gya + oy) | 0;
        const hy = Math.imul(qy, 19349663);
        for (let ox = -1; ox <= 1; ox++) {
          const qx = (gxa + ox) | 0;
          const qh = (Math.imul(qx, 73856093) ^ hy) & mask;
          for (let j = head[qh]; j !== -1; j = next[j]) {
            if (j <= i || cellX[j] !== qx || cellY[j] !== qy) continue;
            const b = list[j];
            const rb = radii[j];
            const minDist = ra + rb + pad;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const d2 = dx * dx + dy * dy;
            if (d2 >= minDist * minDist) continue;
            const dist = Math.sqrt(d2);
            let ux;
            let uy;
            if (dist < 1e-4) {
              const ang = (((i * 2654435761 + j * 1597334677) >>> 0) % 1024) * (Math.PI * 2 / 1024);
              ux = Math.cos(ang);
              uy = Math.sin(ang);
            } else {
              ux = dx / dist;
              uy = dy / dist;
            }
            const invMb = 1 / Math.max(1, rb * rb);
            const wa = invMa / (invMa + invMb);
            const wb = 1 - wa;
            const pen = minDist - dist;
            accX[i] -= ux * pen * wa;
            accY[i] -= uy * pen * wa;
            accW[i] += 1;
            accX[j] += ux * pen * wb;
            accY[j] += uy * pen * wb;
            accW[j] += 1;
            passOverlaps += 1;
            totalResolved += 1;
          }
        }
      }
    }
    if (passOverlaps === 0) break;
    let maxStep = 0;
    for (let i = 0; i < n; i++) {
      if (accW[i] <= 0) continue;
      const damp = 0.78 / Math.sqrt(accW[i]);
      const mx = accX[i] * damp;
      const my = accY[i] * damp;
      list[i].x += mx;
      list[i].y += my;
      const mv = Math.hypot(mx, my);
      if (mv > maxStep) maxStep = mv;
    }
    if (maxStep < 0.05) break;
  }
  return totalResolved;
}

force.seedPhyllotaxis(nodes, 0, 0);
// 桌面档参数（theta 1.2 / alphaMin 0.005）：预计算只跑一次，取更精确的一档
const sim = force.buildSimulation(d3, nodes, edges, { degreeMap, radiusOf, cx: 0, cy: 0, theta: 1.2, alphaMin: 0.005 });
sim.alpha(1);
let ticks = 0;
const t0 = Date.now();
while (sim.alpha() >= sim.alphaMin()) { sim.tick(); ticks += 1; }
const radii = new Float32Array(nodes.length);
for (let i = 0; i < nodes.length; i++) radii[i] = radiusOf(nodes[i]);
const overlapSteps = resolveNodeOverlaps2D(nodes, radii, { padding: 4.5, maxPasses: 18 });
process.stderr.write(`layout: ${nodes.length} nodes, ${edges.length} edges, ${ticks} ticks, ${overlapSteps} overlap pairs resolved, ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
process.stdout.write(JSON.stringify(nodes.map(n => [Math.round(n.x), Math.round(n.y)])));
