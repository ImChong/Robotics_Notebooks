/* Robotics Notebooks unified monochrome SVG icon registry (E3: zero UI emoji). */
(function (root) {
  'use strict';

  var STAR_PATH =
    '<path d="M8 1.8l1.85 3.76 4.15.6-3 2.93.71 4.13L8 11.27l-3.71 1.95.71-4.13-3-2.93 4.15-.6L8 1.8Z" fill="currentColor" stroke="currentColor" stroke-width="0.9" stroke-linejoin="round"/>';

  var ICONS = {
    // Global UI & navigation
    sun: '<circle cx="8" cy="8" r="2.6"/><path d="M8 2v1.6M8 12.4V14M2 8h1.6M12.4 8H14M3.8 3.8l1.1 1.1M11.1 11.1l1.1 1.1M12.2 3.8l-1.1 1.1M4.9 11.1l-1.1 1.1"/>',
    moon: '<path d="M12.8 9.8A5.3 5.3 0 1 1 6.2 3.2a4.3 4.3 0 0 0 6.6 6.6Z"/>',
    coffee: '<path d="M3 6.2h7.6v3.9a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V6.2Z"/><path d="M10.6 7.3h1.3a1.7 1.7 0 0 1 0 3.4h-1.3M5.3 3.2c0 .7.6 1 .6 1.6M8.3 3.2c0 .7.6 1 .6 1.6"/>',
    star: STAR_PATH,
    graph: '<circle cx="3.8" cy="12" r="1.6"/><circle cx="12.2" cy="4" r="1.6"/><circle cx="12.2" cy="12" r="1.6"/><circle cx="7.6" cy="7.6" r="1.8"/><path d="M5 10.8 6.4 9M9 6.3l1.9-1.4M9 8.9l2 2"/>',
    search: '<circle cx="7.1" cy="7.1" r="4.3"/><path d="M10.4 10.4 13.7 13.7"/>',
    sliders: '<path d="M2.5 5h11M2.5 11h11"/><circle cx="6" cy="5" r="1.6" fill="currentColor"/><circle cx="10.4" cy="11" r="1.6" fill="currentColor"/>',
    palette: '<circle cx="5.2" cy="5.2" r="1.7"/><circle cx="10.8" cy="5.2" r="1.7"/><circle cx="5.2" cy="10.8" r="1.7"/><circle cx="10.8" cy="10.8" r="1.7"/>',
    magnet: '<path d="M4.2 3.5v4.6a3.8 3.8 0 0 0 7.6 0V3.5M4.2 6h2.2M9.6 6h2.2"/>',
    link: '<path d="M6.7 9.3 9.3 6.7M5.7 11.2l-.9.9a2.4 2.4 0 0 1-3.4-3.4l.9-.9M10.3 4.8l.9-.9a2.4 2.4 0 0 1 3.4 3.4l-.9.9"/>',
    layers: '<path d="M8 2.5 13.5 5.6 8 8.7 2.5 5.6 8 2.5ZM2.5 8.6 8 11.7l5.5-3.1M2.5 11.4 8 14.3l5.5-2.9"/>',
    clock: '<circle cx="8" cy="8" r="5.5"/><path d="M8 4.8V8l2.3 1.4"/>',
    compass: '<circle cx="8" cy="8" r="5.6"/><polygon points="10.4,5.6 9.1,9.1 5.6,10.4 6.9,6.9" fill="currentColor" stroke="none"/>',
    building: '<path d="M2.6 6.4 8 3.2l5.4 3.2M4.3 6.4v6M8 6.4v6M11.7 6.4v6M2.5 12.6h11"/>',
    calendar: '<rect x="2.6" y="3.8" width="10.8" height="9.4" rx="1.6"/><path d="M2.6 7h10.8M5.4 2.3v2.5M10.6 2.3v2.5"/>',
    trend: '<path d="M2.5 12.6h11M3.4 10.2l3-3 2.1 2 4-4.6M10 4.6h2.5v2.5"/>',
    globe: '<circle cx="8" cy="8" r="5.5"/><ellipse cx="8" cy="8" rx="2.5" ry="5.5"/><path d="M2.7 8h10.6"/>',
    tag: '<path d="M2.8 8.3V3.4a.6.6 0 0 1 .6-.6h4.9l4.9 4.9a.9.9 0 0 1 0 1.3l-4.2 4.2a.9.9 0 0 1-1.3 0L2.8 8.3Z"/><circle cx="5.8" cy="5.8" r="1" fill="currentColor" stroke="none"/>',
    puzzle: '<path d="M3.2 5.4h2.5a1.3 1.3 0 1 1 2.6 0h2.5v2.5a1.3 1.3 0 1 1 0 2.6V13H8.3a1.3 1.3 0 1 0-2.6 0H3.2v-2.5a1.3 1.3 0 1 0 0-2.6V5.4Z"/>',
    'heart-pulse': '<path d="M2.5 8.2h2.3l1.3-3 2 5.6 1.5-3.4h3.9"/>',
    'map-route': '<path d="M2.5 4.8 6 3.3l4 1.5 3.5-1.5v7.9L10 12.7l-4-1.5-3.5 1.5V4.8Z"/><path d="M6 3.3v7.9M10 4.8v7.9"/>',
    'external-link': '<path d="M6.5 3.5H3.5v9h9v-3M9 3.5h3.5V7M7.2 8.8l5.3-5.3"/>',
    document: '<path d="M4.2 2.5h5.2l2.6 2.6v8.4H4.2V2.5Z"/><path d="M9.4 2.5v2.6H12M6 8h4M6 10.6h3"/>',
    'code-repo': '<path d="M5.5 4.8L2.5 8l3 3.2M10.5 4.8L13.5 8l-3 3.2M9 3.6L7 12.4"/>',

    // 31 Depth roadmap topic icons (E3)
    pcb: '<rect x="5" y="5" width="6" height="6" rx="1"/><path d="M6.8 2.5V5M9.2 2.5V5M6.8 11v2.5M9.2 11v2.5M2.5 6.8H5M2.5 9.2H5M11 6.8h2.5M11 9.2h2.5"/>',
    gamepad: '<rect x="2.2" y="5" width="11.6" height="6.6" rx="3.1"/><path d="M5.6 7v2.6M4.3 8.3h2.6"/><circle cx="10.4" cy="7.7" r="0.7" fill="currentColor" stroke="none"/><circle cx="11.7" cy="9" r="0.7" fill="currentColor" stroke="none"/>',
    rocket: '<path d="M9.8 2.5c1.8.2 3.5 1.9 3.7 3.7l-3.9 3.9-3.7-3.7 3.9-3.9Z"/><path d="M5.9 6.4 3.4 7l2 2M9.6 10.1l-.6 2.5-2-2M3.3 12.7l1.8-1.8"/><circle cx="9.8" cy="6.2" r="0.9" fill="currentColor" stroke="none"/>',
    motor: '<circle cx="8" cy="8" r="3.4"/><circle cx="8" cy="8" r="1.1" fill="currentColor" stroke="none"/><path d="M8 2.3v2.1M8 11.6v2.1M2.3 8h2.1M11.6 8h2.1M4 4l1.5 1.5M10.5 10.5 12 12M12 4l-1.5 1.5M5.5 10.5 4 12"/>',
    'ruler-angle': '<path d="M3 13V3l10 10H3Z"/><path d="M5.5 10.5V7.8l2.7 2.7H5.5Z"/>',
    wrench: '<path d="M10.4 2.8a3.2 3.2 0 0 0-3.7 4.4L3 10.9a1.4 1.4 0 0 0 2 2l3.7-3.7a3.2 3.2 0 0 0 4.4-3.7l-1.9 1.9-1.6-1.6 1.8-1.8a3.2 3.2 0 0 0-1-1.2Z"/>',
    shield: '<path d="M8 2.4 12.8 4.2v3.7c0 2.9-2 4.9-4.8 5.8-2.8-.9-4.8-2.9-4.8-5.8V4.2L8 2.4Z"/><path d="M6 8l1.4 1.4L10.3 6.5"/>',
    cycle: '<path d="M12.6 7a4.8 4.8 0 0 0-8.4-2.1M3.4 9a4.8 4.8 0 0 0 8.4 2.1"/><path d="M4.1 2.7v2.3h2.3M11.9 13.3V11H9.6"/>',
    gripper: '<path d="M8 2.5v2.8M5.4 5.3h5.2M5.4 5.3 3.7 9l2 2.7M10.6 5.3l1.7 3.7-2 2.7"/><circle cx="8" cy="11.2" r="1.4"/>',
    'nav-map': '<path d="M8 2.5a3.5 3.5 0 0 1 3.5 3.5c0 2.6-3.5 6.8-3.5 6.8S4.5 8.6 4.5 6A3.5 3.5 0 0 1 8 2.5Z"/><circle cx="8" cy="6" r="1.2" fill="currentColor" stroke="none"/>',
    blocks: '<rect x="2.6" y="2.8" width="4.4" height="4.4" rx="1"/><rect x="9" y="2.8" width="4.4" height="4.4" rx="1"/><rect x="2.6" y="9" width="10.8" height="4.2" rx="1"/>',
    graduation: '<path d="M2.2 6.5 8 3.6l5.8 2.9L8 9.4 2.2 6.5Z"/><path d="M4.6 7.8v3.1c0 .9 1.5 1.8 3.4 1.8s3.4-.9 3.4-1.8V7.8M13.5 6.8v3.4"/>',
    walk: '<circle cx="8.6" cy="3.4" r="1.3"/><path d="M8 5.6 6.8 9.2l2.6 1.5.9 2.7M6.8 9.2l-2 4.2M6 6.8l2-1.2 2.4 1.4"/>',
    'robot-arm': '<path d="M3 13h10M5.2 13v-2.2l3.4-3.6 3-2.4M11.6 4.8l1.6-1M11.6 4.8l1 1.6"/><circle cx="5.2" cy="10.8" r="1.1" fill="currentColor" stroke="none"/><circle cx="8.6" cy="7.2" r="1.2" fill="currentColor" stroke="none"/>',
    soccer: '<circle cx="8" cy="8" r="5.5"/><polygon points="8,5.4 10.2,7 9.4,9.6 6.6,9.6 5.8,7"/><path d="M8 2.5v2.9M10.2 7l2.9-.8M9.4 9.6l1.8 2.4M6.6 9.6l-1.8 2.4M5.8 7l-2.9-.8"/>',
    retarget: '<circle cx="5" cy="4.2" r="1.2"/><path d="M3 7.2h4M5 5.8v3.6l-1.6 3.3M5 9.4l1.5 3.3M9.5 8h4M11.8 6.2 13.6 8l-1.8 1.8"/>',
    swarm: '<circle cx="8" cy="4.2" r="1.5"/><circle cx="4" cy="11.4" r="1.5"/><circle cx="12" cy="11.4" r="1.5"/><path d="M7.1 5.6 4.8 9.9M8.9 5.6l2.3 4.3M5.7 11.4h4.6"/>',
    'arm-link': '<circle cx="4.2" cy="11.4" r="1.6"/><circle cx="8" cy="4.8" r="1.6"/><circle cx="11.8" cy="11.4" r="1.6"/><path d="M5 10l2.2-3.8M8.8 6.2 11 10M6.2 11.4h3.6"/>',
    sim2real: '<rect x="2.2" y="4.2" width="4.6" height="7.6" rx="1" stroke-dasharray="1.8 1.5"/><rect x="9.2" y="4.2" width="4.6" height="7.6" rx="1"/><path d="M6.8 8h2.4"/>',
    boxing: '<path d="M4.2 7.2V5.3A2.3 2.3 0 0 1 6.5 3h3.2A2.6 2.6 0 0 1 12.3 5.6v2.7a3.2 3.2 0 0 1-3.2 3.2H6.4a2.2 2.2 0 0 1-2.2-2.2V7.2Z"/><path d="M6 11.5v1.8h4.5v-1.8M7.5 6.4H12"/>',
    brain: '<path d="M8 3.2v9.6M8 4.4a2.3 2.3 0 0 0-4.2 1.2 2.2 2.2 0 0 0-.5 3.8 2.2 2.2 0 0 0 4.7 1.2M8 4.4a2.3 2.3 0 0 1 4.2 1.2 2.2 2.2 0 0 1 .5 3.8 2.2 2.2 0 0 1-4.7 1.2"/>',
    'bar-chart': '<path d="M2.5 13h11M4.6 13V8.6M8 13V4.4M11.4 13V6.6"/>',
    'eye-step': '<path d="M2.2 7.4S4.4 4 8 4s5.8 3.4 5.8 3.4S11.6 10.8 8 10.8 2.2 7.4 2.2 7.4Z"/><circle cx="8" cy="7.4" r="1.5"/><path d="M3 13h3.5v-1.4h3.5V13H13"/>',
    sparkles: '<path d="M8 2.4 9.1 5.9 12.6 7 9.1 8.1 8 11.6 6.9 8.1 3.4 7l3.5-1.1L8 2.4ZM12.3 10.5l.5 1.4 1.4.5-1.4.5-.5 1.4-.5-1.4-1.4-.5 1.4-.5.5-1.4Z"/>',
    flow: '<path d="M2.5 5.4c1.8-2 3.7-2 5.5 0s3.7 2 5.5 0M2.5 8.6c1.8-2 3.7-2 5.5 0s3.7 2 5.5 0M2.5 11.8c1.8-2 3.7-2 5.5 0s3.7 2 5.5 0"/>',
    'vla-eye': '<path d="M2.2 6.8S4.4 3.5 8 3.5s5.8 3.3 5.8 3.3S11.6 10.1 8 10.1 2.2 6.8 2.2 6.8Z"/><circle cx="8" cy="6.8" r="1.4" fill="currentColor" stroke="none"/><path d="M5 12.6h6M9.4 11l1.6 1.6-1.6 1.6"/>',
    real2sim: '<path d="M8 2.5 13 5.3v5.4L8 13.5 3 10.7V5.3L8 2.5Z"/><path d="M8 8v5.5M8 8l5-2.7M8 8 3 5.3"/>',
    database: '<ellipse cx="8" cy="4.4" rx="4.6" ry="1.8"/><path d="M3.4 4.4v7.2c0 1 2.1 1.8 4.6 1.8s4.6-.8 4.6-1.8V4.4M3.4 8c0 1 2.1 1.8 4.6 1.8s4.6-.8 4.6-1.8"/>',
    'world-orb': '<circle cx="8" cy="8" r="4.6"/><path d="M3.6 9.8c2.6 1.2 6.2 1.2 8.8 0M4.5 5.5c2.2 1 4.8 1 7 0"/><circle cx="12.2" cy="4" r="1" fill="currentColor" stroke="none"/>'
  };

  var DEPTH_ICONS = {
    'motion-control': 'compass',
    'pcb-design': 'pcb',
    teleoperation: 'gamepad',
    'rocket-landing': 'rocket',
    'torque-motor-design': 'motor',
    'classical-control': 'ruler-angle',
    'humanoid-hardware-design': 'wrench',
    'safe-control': 'shield',
    rsi: 'cycle',
    'contact-manipulation': 'gripper',
    navigation: 'nav-map',
    'model-architecture': 'blocks',
    'imitation-learning': 'graduation',
    'rl-locomotion': 'walk',
    'loco-manipulation': 'robot-arm',
    'humanoid-soccer': 'soccer',
    'motion-retargeting': 'retarget',
    'humanoid-swarm-performance': 'swarm',
    'cross-embodiment': 'arm-link',
    sim2real: 'sim2real',
    'humanoid-boxing': 'boxing',
    icl: 'puzzle',
    bfm: 'brain',
    'embodied-eval': 'bar-chart',
    'perceptive-locomotion': 'eye-step',
    'motion-generation': 'sparkles',
    'robotics-diffusion-dit-flow': 'flow',
    vla: 'vla-eye',
    real2sim: 'real2sim',
    'embodied-data': 'database',
    wam: 'world-orb'
  };

  function svg(name, extraClass) {
    var key = ICONS[name] ? name : 'tag';
    var cls = 'ui-icon' + (key === 'star' ? ' ui-icon--star' : '') + (extraClass ? ' ' + extraClass : '');
    return '<svg class="' + cls + '" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' +
      ICONS[key] + '</svg>';
  }

  function depthSvg(depthKey, iconName, extraClass) {
    var key = (iconName && ICONS[iconName]) ? iconName : (DEPTH_ICONS[depthKey] || 'tag');
    return svg(key, extraClass);
  }

  function repoStarHtml() {
    return '<span class="updates-item-opensource" aria-label="含开源仓库" title="含开源仓库">' + svg('star') + '</span>';
  }

  function has(name) {
    return Object.prototype.hasOwnProperty.call(ICONS, name);
  }

  var api = {
    ICONS: ICONS,
    DEPTH_ICONS: DEPTH_ICONS,
    svg: svg,
    depthSvg: depthSvg,
    repoStarHtml: repoStarHtml,
    has: has
  };

  root.RNIcons = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
