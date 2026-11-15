// 零基础篇视频接入 Vincentwei1021/video-shotcraft Workbench（多轨源码分轨审核与导出）。
//
// Usage:
//   node workbench.mjs <ep> [--open] [--no-open] [--no-import] [--shotcraft <dir>] [--port <port>]
//
// 先执行 python3 build.py <ep> 生成 out/<ep>/timeline.json 与 seg*.wav；
// 本脚本在 out/<ep>/workbench/ 产出标准 Remotion + src/workbench.ts 工程，
// 传入 --open（或 --shotcraft / --port）时自动调用 video-shotcraft/workbench/scripts/open.mjs 打开工作台。
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (k) => args.includes(`--${k}`);
const opt = (k) => {
  const i = args.indexOf(`--${k}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : null;
};
const valueFlags = new Set(['--shotcraft', '--port']);
const positional = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && valueFlags.has(args[i - 1])));
const ep = positional[0];
if (!ep) {
  console.error('用法：node workbench.mjs <ep> [--open] [--no-open] [--no-import] [--shotcraft <dir>] [--port <port>]');
  process.exit(2);
}

const OUT = path.join(HERE, 'out', ep);
const tlPath = path.join(OUT, 'timeline.json');
if (!fs.existsSync(tlPath)) {
  console.error(`找不到 ${tlPath}；请先运行：python3 build.py ${ep}`);
  process.exit(1);
}

const epDir = path.join(HERE, 'episodes', ep);
const singleEp = path.join(HERE, 'episodes', `${ep}.js`);
const scripts = fs.existsSync(epDir)
  ? fs.readdirSync(epDir).filter((f) => f.endsWith('.js')).sort().map((f) => `episodes/${ep}/${f}`)
  : fs.existsSync(singleEp)
    ? [`episodes/${ep}.js`]
    : null;
if (!scripts || scripts.length === 0) {
  console.error(`找不到分镜源码：episodes/${ep}.js 或 episodes/${ep}/*.js`);
  process.exit(1);
}

const epCode = fs.readFileSync(path.join(HERE, scripts[scripts.length - 1]), 'utf8');
const titleMatch = epCode.match(/EpisodeVideo\s*=\s*\{[\s\S]*?\btitle\s*:\s*(['"`])([^'"`]+)\1/)
  || epCode.match(/\btitle\s*:\s*(['"`])([^'"`]+)\1/);
const epNum = ep.replace(/^ep/i, '');
const epTitle = titleMatch ? titleMatch[2] : '源码分轨';
const projectName = `零基础篇 ${epNum} · ${epTitle}`;

const DEST = path.join(OUT, 'workbench');
const SRC = path.join(DEST, 'src');
const PUB = path.join(DEST, 'public');
const SRC_PUB_NAME = `${ep}-source`;
const SRC_PUB = path.join(PUB, SRC_PUB_NAME);
const AUDIO_PUB = path.join(PUB, 'audio');

fs.mkdirSync(SRC, { recursive: true });
fs.mkdirSync(SRC_PUB, { recursive: true });
fs.mkdirSync(AUDIO_PUB, { recursive: true });

function unlinkIfExists(p) {
  try {
    const st = fs.lstatSync(p);
    if (st.isSymbolicLink() || st.isFile()) fs.unlinkSync(p);
    else if (st.isDirectory()) fs.rmSync(p, { recursive: true, force: true });
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
}

function linkItem(target, linkPath, isDir = false) {
  unlinkIfExists(linkPath);
  fs.symlinkSync(target, linkPath, isDir && process.platform === 'win32' ? 'junction' : undefined);
}

// 1) package.json（open.mjs 要求工程根目录含 package.json 与 src/workbench.ts）
fs.writeFileSync(
  path.join(DEST, 'package.json'),
  `${JSON.stringify({ name: `${ep}-review`, private: true, type: 'module' })}\n`,
);

// 2) public/audio/：仅保留本次 build.py 产出的 seg*.wav 软链
const wavFiles = fs.readdirSync(OUT).filter((f) => /^seg\d+\.wav$/.test(f)).sort();
for (const old of fs.readdirSync(AUDIO_PUB)) {
  if (/^seg\d+\.wav$/.test(old)) unlinkIfExists(path.join(AUDIO_PUB, old));
}
for (const wav of wavFiles) {
  linkItem(path.join(OUT, wav), path.join(AUDIO_PUB, wav), false);
}

// 3) public/<ep>-source/：链接舞台运行依赖并注入 postMessage 帧同步桥
for (const [name, isDir, optional] of [
  ['episodes', true, false],
  ['node_modules', true, false],
  ['robot3d.js', false, false],
  ['robots.json', false, false],
  ['vendor', true, false],
  ['assets', true, true],
]) {
  const target = path.join(HERE, name);
  const dest = path.join(SRC_PUB, name);
  if (fs.existsSync(target)) linkItem(target, dest, isDir);
  else if (optional) unlinkIfExists(dest);
  else {
    console.error(`缺少舞台依赖 ${target}（请先在 media/zero-to-ppo-video 运行 npm install）`);
    process.exit(1);
  }
}

const stageHtmlRaw = fs.readFileSync(path.join(HERE, 'stage.html'), 'utf8');
const bridgeScript = `<script>
let latestTime = 0, initializing;
window.addEventListener('message', async (e) => {
  if (e.origin !== location.origin || e.source !== parent || e.data?.type !== 'source-frame') return;
  latestTime = e.data.time;
  if (!initializing) {
    if (!e.data.timeline) return;
    initializing = (async () => {
      await window.robotsReady;
      window.setup({ ...e.data.timeline, subs: [] });
      await document.fonts.ready;
      for (const s of e.data.timeline.segments) window.renderAt(s.t0 + 0.5);
      await document.fonts.ready;
      window.fitStages();
      const sub = document.getElementById('sub');
      if (sub) sub.style.display = 'none';
    })();
  }
  try {
    await initializing;
    window.renderAt(latestTime);
    parent.postMessage({ type: 'source-ready', time: latestTime }, location.origin);
  } catch (err) {
    parent.postMessage({ type: 'source-error', time: latestTime, message: String(err) }, location.origin);
  }
});
</script></body>`;

const stageHtml = stageHtmlRaw
  .replace('<script src="episodes/EPISODE.js"></script>', scripts.map((s) => `<script src="${s}"></script>`).join('\n'))
  .replace('</body>', bridgeScript);
fs.writeFileSync(path.join(SRC_PUB, 'stage.html'), stageHtml);

// 可选：已渲好的成片与封面也挂进 public/，方便在工作台素材 tab 查阅
for (const [srcName, dstName] of [[`${ep}_video.mp4`, `${ep}_video.mp4`], ['cover.png', 'cover.png']]) {
  const srcPath = path.join(OUT, srcName);
  const dstPath = path.join(PUB, dstName);
  if (fs.existsSync(srcPath)) linkItem(srcPath, dstPath, false);
  else unlinkIfExists(dstPath);
}

// 4) src/timeline.json + src/workbench.ts + src/SourceReview.tsx
const tl = JSON.parse(fs.readFileSync(tlPath, 'utf8'));
fs.writeFileSync(path.join(SRC, 'timeline.json'), JSON.stringify(tl));
fs.writeFileSync(path.join(SRC, 'workbench.ts'), "export { WORKBENCH } from './SourceReview';\n");

// 注：不写死 revision，让 Workbench 按清单内容哈希判别 timeline.json 变更并自动重新导入，同一版则保留用户在浏览器内的微调。
const sourceReviewTsx = `import React, { useRef, useEffect, useLayoutEffect, useState } from 'react';
import {
  AbsoluteFill,
  cancelRender,
  continueRender,
  delayRender,
  getRemotionEnvironment,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import tl from './timeline.json';

const SOURCE_DIR = ${JSON.stringify(SRC_PUB_NAME)};
const PROJECT_NAME = ${JSON.stringify(projectName)};

// 主预览只用一个舞台 iframe，全程不重开：Remotion 的 clip 一出范围就卸载（循环回到开头或跨片段时会重挂），
// 重开 iframe 要重新加载 URDF、预热每一幕，白屏好几秒。卸载时用 moveBefore（Chrome 133+，移动 iframe 不重载）
// 停到页外，再挂上时搬回来；不支持 moveBefore 的浏览器退回 appendChild。缩略图与导出渲染各自独立加载。
let mainFrame: HTMLIFrameElement | null = null;
let parking: HTMLDivElement | null = null;
let mainTime = 0;

const postMain = () =>
  mainFrame?.contentWindow?.postMessage({ type: 'source-frame', timeline: tl, time: mainTime }, location.origin);

const moveTo = (parent: Element, node: Element) =>
  (parent as unknown as { moveBefore?: (n: Node, ref: Node | null) => void }).moveBefore
    ? (parent as unknown as { moveBefore: (n: Node, ref: Node | null) => void }).moveBefore(node, null)
    : parent.appendChild(node);

const getMainFrame = () => {
  if (!mainFrame) {
    parking = document.createElement('div');
    parking.style.cssText =
      'position:fixed;left:-20000px;top:0;width:1080px;height:1920px;overflow:hidden;pointer-events:none';
    document.body.appendChild(parking);
    mainFrame = document.createElement('iframe');
    mainFrame.title = ${JSON.stringify(`${ep} source scene`)};
    mainFrame.style.cssText = 'width:1080px;height:1920px;border:0;display:block';
    mainFrame.onload = postMain;
    mainFrame.src = staticFile(SOURCE_DIR + '/stage.html');
    parking.appendChild(mainFrame);
  }
  return mainFrame;
};

const Stage = ({ offset = 0 }: { offset?: number }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const host = useRef<HTMLDivElement>(null);
  const own = useRef<HTMLIFrameElement>(null);
  const loaded = useRef(false);
  const [isMain, setMain] = useState<boolean | null>(null);
  const time = offset + frame / fps;
  const timeRef = useRef(time);
  timeRef.current = time;
  const { isRendering } = getRemotionEnvironment();

  useLayoutEffect(() => {
    const h = host.current;
    if (!h) return;
    const main = !isRendering && !!h.closest('.preview-stage');
    setMain(main);
    if (!main) return;
    const fr = getMainFrame();
    moveTo(h, fr);
    postMain();
    return () => {
      if (parking && fr.parentNode !== parking) moveTo(parking, fr);
    };
  }, [isRendering]);

  useEffect(() => {
    if (isMain) {
      mainTime = time;
      postMain();
      return;
    }
    if (isMain !== false) return;
    const postOwn = (t: number) =>
      own.current?.contentWindow?.postMessage({ type: 'source-frame', timeline: tl, time: t }, location.origin);
    if (!isRendering) {
      if (loaded.current) postOwn(time);
      return;
    }
    const handle = delayRender('source-stage');
    const onMessage = (e: MessageEvent) => {
      if (e.origin !== location.origin || e.source !== own.current?.contentWindow) return;
      if (e.data?.type === 'source-ready' && Math.abs((e.data.time ?? time) - time) < 1e-4) {
        window.removeEventListener('message', onMessage);
        continueRender(handle);
      } else if (e.data?.type === 'source-error') {
        window.removeEventListener('message', onMessage);
        cancelRender(new Error(String(e.data.message || 'stage render error')));
      }
    };
    window.addEventListener('message', onMessage);
    if (loaded.current) postOwn(time);
    return () => {
      window.removeEventListener('message', onMessage);
      continueRender(handle);
    };
  }, [time, isMain, isRendering]);

  const sendOwn = () => {
    loaded.current = true;
    own.current?.contentWindow?.postMessage(
      { type: 'source-frame', timeline: tl, time: timeRef.current },
      location.origin,
    );
  };

  return (
    <AbsoluteFill>
      <div ref={host} style={{ width: 1080, height: 1920 }}>
        {isMain === false && (
          <iframe
            title=${JSON.stringify(`${ep} source scene`)}
            ref={own}
            src={staticFile(SOURCE_DIR + '/stage.html')}
            onLoad={sendOwn}
            style={{ width: 1080, height: 1920, border: 0 }}
          />
        )}
      </div>
    </AbsoluteFill>
  );
};

// staticFile 会把 @ 编码成 %40，vite dev server 不认；加载字体表时还原 @
let subStyled = false;
const styleSub = () => {
  if (subStyled) return;
  subStyled = true;
  for (const w of ['400', '700', '900']) {
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = staticFile(SOURCE_DIR + '/node_modules/').replace('%40', '@') + '@fontsource/noto-sans-sc/' + w + '.css';
    document.head.appendChild(l);
  }
  const st = document.createElement('style');
  st.textContent = '[data-source-subtitle] .katex{font-size:1.02em;font-weight:700}';
  document.head.appendChild(st);
};

const Subtitle = ({ text = '' }: { text?: string }) => {
  styleSub();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    for (let size = 52; size >= 40; size -= 4) {
      el.style.fontSize = size + 'px';
      if (1318 + el.offsetHeight <= 1440) break;
    }
  }, [text]);
  return (
    <AbsoluteFill style={{ pointerEvents: 'none' }}>
      <div
        ref={ref}
        data-source-subtitle
        lang="zh-CN"
        style={{
          position: 'absolute',
          top: 1318,
          left: 50,
          right: 50,
          fontFamily: '"Noto Sans SC", "WenQuanYi Zen Hei", sans-serif',
          fontSize: 52,
          fontWeight: 900,
          lineHeight: 1.3,
          color: '#fff',
          textShadow: '0 3px 0 #000, 0 0 18px rgba(0,0,0,.85)',
          textAlign: 'center',
          whiteSpace: 'pre-wrap',
        }}
        dangerouslySetInnerHTML={{
          __html: text
            .split(/(\\$[^$]*\\$)/g)
            .map((s) =>
              s.startsWith('$')
                ? katex.renderToString(s.slice(1, -1), { throwOnError: false })
                : s.replaceAll('&', '&amp;').replaceAll('<', '&lt;'),
            )
            .join(''),
        }}
      />
    </AbsoluteFill>
  );
};

const f = (t: number) => Math.round(t * 30);
const groups: { scene: string | number; t0: number; t1: number }[] = [];
for (const s of tl.segments) {
  const last = groups.at(-1);
  if (last && String(last.scene) === String(s.scene)) last.t1 = s.t1;
  else groups.push({ scene: s.scene, t0: s.t0, t1: s.t1 });
}

const ShotMarker = () => null;
const TOTAL = Math.max(2, Math.ceil(tl.total * 30));
const END = Math.max(TOTAL, ...tl.segments.map((s) => f(s.t0 + s.lead) + f(s.alen)));

const sceneLabel = (scene: string | number) =>
  scene === 'intro' ? '片头' : scene === 'recap' ? '总结' : scene === 'outro' ? '片尾' : \`第 \${Number(scene) + 1} 幕\`;

export const WORKBENCH = {
  name: PROJECT_NAME,
  fps: 30,
  width: 1080,
  height: 1920,
  total: TOTAL,
  background: '#16161a',
  shots: groups.map((s, i) => ({
    id: \`scene-\${i}\`,
    label: sceneLabel(s.scene),
    from: f(s.t0),
    duration: Math.max(2, (i === groups.length - 1 ? TOTAL : f(s.t1)) - f(s.t0)),
    component: ShotMarker,
    props: {},
  })),
  overlays: [
    {
      id: 'stage',
      label: '舞台（stage.html 源码）',
      from: 0,
      duration: END,
      component: Stage,
      props: { offset: 0 },
    },
  ],
  captions: tl.subs.map((s, i) => ({
    id: \`sub-\${i}\`,
    label: s.text,
    from: f(s.t0),
    duration: Math.max(1, f(Math.min(s.t1 + 0.15, tl.subs[i + 1]?.t0 ?? tl.total)) - f(s.t0)),
    component: Subtitle,
    props: { text: s.text },
    schema: [{ type: 'textarea', key: 'text', label: '字幕', default: '' }],
  })),
  sfx: tl.segments.map((s, i) => ({
    from: f(s.t0 + s.lead),
    duration: Math.max(2, f(s.alen)),
    src: \`audio/seg\${String(i).padStart(2, '0')}.wav\`,
    volume: 1,
    label: s.text,
  })),
};
`;

fs.writeFileSync(path.join(SRC, 'SourceReview.tsx'), sourceReviewTsx);
console.log(`[workbench] 已生成 ${ep} 源码分轨工程：${DEST}`);

function findShotcraftDir() {
  const candidates = [
    opt('shotcraft'),
    process.env.SHOTCRAFT_DIR,
    path.join(os.homedir(), '.codex/skills/video-shotcraft'),
    path.join(os.homedir(), '.claude/skills/video-shotcraft'),
    path.join(os.homedir(), '.agents/skills/video-shotcraft'),
    '/tmp/video-shotcraft',
  ].filter(Boolean);
  return candidates.find((d) => fs.existsSync(path.join(d, 'workbench/scripts/open.mjs'))) || null;
}

const shouldOpen = flag('open') || flag('no-open') || !!opt('shotcraft') || !!opt('port');
if (!shouldOpen) {
  console.log(`[workbench] 下一步：node workbench.mjs ${ep} --open（或 node <video-shotcraft>/workbench/scripts/open.mjs ${DEST}）`);
  process.exit(0);
}

const shotcraftDir = findShotcraftDir();
if (!shotcraftDir) {
  console.error(
    '找不到 video-shotcraft（未设置 --shotcraft / SHOTCRAFT_DIR，且默认技能目录不存在）。\n' +
    '请先克隆 https://github.com/Vincentwei1021/video-shotcraft ，再运行：\n' +
    `  node <video-shotcraft>/workbench/scripts/open.mjs ${DEST}`,
  );
  process.exit(1);
}

const openArgs = [path.join(shotcraftDir, 'workbench/scripts/open.mjs'), DEST];
if (opt('port')) openArgs.push('--port', opt('port'));
if (flag('no-open')) openArgs.push('--no-open');
if (flag('no-import')) openArgs.push('--no-import');

const res = spawnSync(process.execPath, openArgs, { stdio: 'inherit', cwd: shotcraftDir });
process.exit(res.status ?? 1);
