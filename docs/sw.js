// Robotics Notebooks Service Worker — 离线缓存支持
const CACHE_NAME = 'robotics-wiki-2026-09-28';
const CACHE_PREFIX = 'robotics-wiki-';
const PROJECT_PATH = new URL(self.registration.scope).pathname;
const asset = (path) => PROJECT_PATH + path;
// 必要外壳：离线打开首页所需的最小集合，安装时必须齐全；缺一即安装失败，
// 旧 SW 继续服务，下次访问重试，不留下半套外壳。
const SHELL_ASSETS = [
  '',
  'index.html',
  'style.css',
  'theme-init.js',
  'main.js',
  'org-route-order.js',
  'home-i18n.js',
].map(asset);
// 可选资源：其余页面外壳与小体积数据，逐个缓存，单个失败只降级该资源的离线可用性。
// 搜索索引、全图、活动全集与榜单不再预取（安装时下载量最大的四份），改为访问时
// 按 stale-while-revalidate 落缓存（超大 search-index.json 不写 CacheStorage），
// 离线可读范围由读者实际打开的内容决定。
const OPTIONAL_ASSETS = [
  'graph.html',
  'change-log.html',
  'hubs.html',
  'company.html',
  'lab.html',
  'detail.html',
  'roadmap.html',
  'star-history.html',
  '404.html',
  'wiki-type-labels.js',
  'graph-tooltip.js',
  'graph-node-size.js',
  'graph-force-params.js',
  'graph-perspective.js',
  'graph-3d.js',
  'depth-filters.js',
  'mini-graph.js',
  'company-roadmap.js',
  'kb-chat.js',
  'sponsor.js',
  'company-roadmaps.json',
  'lab-roadmaps.json',
  'vendor/d3.min.js',
  'exports/home-stats.json',
  'exports/graph-stats.json',
  'exports/mini-graph.json',
].map(asset);

function isHtmlShellPath(url) {
  return url.pathname === PROJECT_PATH || url.pathname.endsWith('.html');
}

function shouldSkipCachePut(url) {
  return url.pathname.endsWith('/search-index.json');
}

function matchFromCache(cache, request, url) {
  return cache.match(request).then((cached) => {
    if (cached || !isHtmlShellPath(url)) return cached;
    return cache.match(url.pathname).then((byPath) =>
      byPath || cache.match(request, { ignoreSearch: true })
    );
  });
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      cache.addAll(SHELL_ASSETS).then(() =>
        Promise.all(
          OPTIONAL_ASSETS.map((url) =>
            cache.add(url).catch((err) => {
              console.warn('[SW] 可选资源缓存失败（该资源离线不可用）:', url, err);
            })
          )
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => Promise.resolve(self.clients.claim()))
  );
});

self.addEventListener('fetch', (event) => {
  // 只处理同源 GET 请求
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(PROJECT_PATH)) return;
  // 视频交给浏览器原生 Range 请求（拖动进度条）：不进离线缓存，也不被改写成整段 200 响应。
  if (url.pathname.endsWith('.mp4')) return;

  const isHtml = isHtmlShellPath(url);
  const skipPut = shouldSkipCachePut(url);
  const putTarget = isHtml && url.search ? url.pathname : event.request;

  // 目录优先读网络，离线再用缓存；正文哈希 URL 与目录版本配套。
  if (url.pathname.endsWith('/sponsor.js') || url.pathname.endsWith('/site-catalog-v1.json')) {
    event.respondWith(
      fetch(event.request)
        .then((resp) => {
          if (resp && resp.status === 200 && !skipPut) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(putTarget, clone));
          }
          return resp;
        })
        .catch(() => caches.open(CACHE_NAME).then((cache) => cache.match(event.request)))
    );
    return;
  }

  event.respondWith(
    caches
      .open(CACHE_NAME)
      .then((cache) => matchFromCache(cache, event.request, url))
      .then((cached) => {
        if (cached) {
          // 哈希正文不可变；已缓存的旧版本离线可读，也不必再后台下载。
          if (/\/exports\/page-content\/[a-f0-9]{64}\.json$/.test(url.pathname)) return cached;
          // 后台刷新缓存（stale-while-revalidate）
          fetch(event.request)
            .then((resp) => {
              if (resp && resp.status === 200 && !skipPut) {
                caches.open(CACHE_NAME).then((cache) => cache.put(putTarget, resp));
              }
            })
            .catch(() => {});
          return cached;
        }
        return fetch(event.request).then((resp) => {
          if (resp && resp.status === 200 && !skipPut) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(putTarget, clone));
          }
          return resp;
        });
      })
  );
});
