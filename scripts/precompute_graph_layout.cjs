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

force.seedPhyllotaxis(nodes, 0, 0);
// 桌面档参数（theta 1.2 / alphaMin 0.005）：预计算只跑一次，取更精确的一档
const sim = force.buildSimulation(d3, nodes, edges, { degreeMap, radiusOf, cx: 0, cy: 0, theta: 1.2, alphaMin: 0.005 });
sim.alpha(1);
let ticks = 0;
const t0 = Date.now();
while (sim.alpha() >= sim.alphaMin()) { sim.tick(); ticks += 1; }
process.stderr.write(`layout: ${nodes.length} nodes, ${edges.length} edges, ${ticks} ticks, ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
process.stdout.write(JSON.stringify(nodes.map(n => [Math.round(n.x), Math.round(n.y)])));
