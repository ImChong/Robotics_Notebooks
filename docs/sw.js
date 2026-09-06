// Robotics Notebooks Service Worker — 离线缓存支持
const CACHE_NAME = 'robotics-wiki-2026-09-06';
const CACHE_PREFIX = 'robotics-wiki-';
const PROJECT_PATH = new URL(self.registration.scope).pathname;
const ASSETS_TO_CACHE = [
  '/Robotics_Notebooks/',
  '/Robotics_Notebooks/index.html',
  '/Robotics_Notebooks/graph.html',
  '/Robotics_Notebooks/change-log.html',
  '/Robotics_Notebooks/hubs.html',
  '/Robotics_Notebooks/main.js',
  '/Robotics_Notebooks/vendor/d3.min.js',
  '/Robotics_Notebooks/search-index.json',
  '/Robotics_Notebooks/exports/home-stats.json',
  '/Robotics_Notebooks/exports/hub-rankings.json',
  '/Robotics_Notebooks/exports/link-graph.json',
  '/Robotics_Notebooks/exports/graph-stats.json',
  '/Robotics_Notebooks/exports/wiki-activity.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(ASSETS_TO_CACHE).catch((err) => {
        console.warn('[SW] 部分资源缓存失败（离线模式下将降级）:', err);
      });
    })
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
