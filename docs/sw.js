// Robotics Notebooks Service Worker — 离线缓存支持
const CACHE_NAME = 'robotics-wiki-2026-09-09';
const CACHE_PREFIX = 'robotics-wiki-';
const PROJECT_PATH = new URL(self.registration.scope).pathname;
// 必要外壳：离线打开首页所需的最小集合，安装时必须齐全；缺一即安装失败，
// 旧 SW 继续服务，下次访问重试，不留下半套外壳。
const SHELL_ASSETS = [
  '/Robotics_Notebooks/',
  '/Robotics_Notebooks/index.html',
  '/Robotics_Notebooks/style.css',
  '/Robotics_Notebooks/theme-init.js',
  '/Robotics_Notebooks/main.js',
];
// 可选资源：其余页面外壳与小体积数据，逐个缓存，单个失败只降级该资源的离线可用性。
// 搜索索引、全图、活动全集与榜单不再预取（安装时下载量最大的四份），改为访问时
// 按 stale-while-revalidate 落缓存，离线可读范围由读者实际打开的内容决定。
const OPTIONAL_ASSETS = [
  '/Robotics_Notebooks/graph.html',
  '/Robotics_Notebooks/change-log.html',
  '/Robotics_Notebooks/hubs.html',
  '/Robotics_Notebooks/detail.html',
  '/Robotics_Notebooks/wiki-type-labels.js',
  '/Robotics_Notebooks/graph-tooltip.js',
  '/Robotics_Notebooks/graph-node-size.js',
  '/Robotics_Notebooks/mini-graph.js',
  '/Robotics_Notebooks/vendor/d3.min.js',
  '/Robotics_Notebooks/exports/home-stats.json',
  '/Robotics_Notebooks/exports/graph-stats.json',
];

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
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // 只处理同源 GET 请求
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(PROJECT_PATH)) return;

  // 目录优先读网络，离线再用缓存；正文哈希 URL 与目录版本配套。
  if (url.pathname.endsWith('/sponsor.js') || url.pathname.endsWith('/site-catalog-v1.json')) {
    event.respondWith(
      fetch(event.request)
        .then((resp) => {
          if (resp && resp.status === 200) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return resp;
        })
        .catch(() => caches.open(CACHE_NAME).then((cache) => cache.match(event.request)))
    );
    return;
  }

  event.respondWith(
    caches.open(CACHE_NAME).then((cache) => cache.match(event.request)).then((cached) => {
      if (cached) {
        // 哈希正文不可变；已缓存的旧版本离线可读，也不必再后台下载。
        if (/\/exports\/page-content\/[a-f0-9]{64}\.json$/.test(url.pathname)) return cached;
        // 后台刷新缓存（stale-while-revalidate）
        fetch(event.request)
          .then((resp) => {
            if (resp && resp.status === 200) {
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, resp));
            }
          })
          .catch(() => {});
        return cached;
      }
      return fetch(event.request).then((resp) => {
        if (resp && resp.status === 200) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return resp;
      });
    })
  );
});
