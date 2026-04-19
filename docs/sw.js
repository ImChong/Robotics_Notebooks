// Robotics Notebooks Service Worker — 离线缓存支持
const CACHE_NAME = 'robotics-wiki-v1';
const ASSETS_TO_CACHE = [
  '/Robotics_Notebooks/',
  '/Robotics_Notebooks/index.html',
  '/Robotics_Notebooks/graph.html',
  '/Robotics_Notebooks/main.js',
  '/Robotics_Notebooks/search-index.json',
  '/Robotics_Notebooks/exports/link-graph.json',
  '/Robotics_Notebooks/exports/site-data-v1.json',
  '/Robotics_Notebooks/exports/index-v1.json',
  '/Robotics_Notebooks/exports/graph-stats.json',
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
          .filter((key) => key !== CACHE_NAME)
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

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) {
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
