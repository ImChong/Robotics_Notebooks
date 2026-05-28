#!/usr/bin/env node
/**
 * Offline force layout via d3-force (matches docs/graph.html Barnes-Hut charge).
 * stdin: JSON { nodes, edges, degree_map, width, height, seed, params }
 * stdout: layout object { version, width, height, params, positions }
 */
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from 'd3';

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function baseRadius(degree, nodeScale = 1) {
  return Math.max(5, Math.min(18, 4 + Math.sqrt(Math.max(degree, 0)) * 2.2)) * nodeScale;
}

const input = JSON.parse(await new Promise((resolve, reject) => {
  let data = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => { data += chunk; });
  process.stdin.on('end', () => resolve(data));
  process.stdin.on('error', reject);
}));

const {
  nodes = [],
  edges = [],
  degree_map: degreeMap = {},
  width = 1200,
  height = 800,
  seed = 42,
  params = {},
  max_ticks: maxTicks = 320,
} = input;

if (nodes.length === 0) {
  process.stdout.write(JSON.stringify({
    version: params.version ?? 1,
    width,
    height,
    params,
    positions: {},
  }));
  process.exit(0);
}

const rng = mulberry32(seed);
const simNodes = nodes.map(node => ({
  id: node.id,
  x: width / 2 + (rng() - 0.5) * 80,
  y: height / 2 + (rng() - 0.5) * 80,
}));

const charge = params.charge ?? -800;
const linkDistance = params.link_distance ?? 80;
const linkStrength = params.link_strength ?? 0.35;
const centerStrength = params.center_strength ?? 0.05;
const distanceMax = params.distance_max ?? 400;
const collisionStrength = params.collision_strength ?? 0.7;
const alphaDecay = params.alpha_decay ?? 0.025;
const nodeScale = params.node_scale ?? 1;

const simulation = forceSimulation(simNodes)
  .force('link', forceLink(edges).id(d => d.id).distance(linkDistance).strength(linkStrength))
  .force('charge', forceManyBody().strength(charge).distanceMax(distanceMax))
  .force('center', forceCenter(width / 2, height / 2).strength(centerStrength))
  .force(
    'collision',
    forceCollide()
      .radius(d => baseRadius(degreeMap[d.id] ?? 0, nodeScale) + 5)
      .strength(collisionStrength),
  )
  .alphaDecay(alphaDecay)
  .velocityDecay(0.6)
  .stop();

for (let i = 0; i < maxTicks; i += 1) {
  simulation.tick();
}

const positions = {};
for (const node of simNodes) {
  positions[node.id] = {
    x: Math.round(node.x * 100) / 100,
    y: Math.round(node.y * 100) / 100,
  };
}

process.stdout.write(JSON.stringify({
  version: params.version ?? 1,
  width,
  height,
  params,
  positions,
}));
