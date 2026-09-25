/* 谛听 VeriCall · 子女守护端 Service Worker（PWA 离线壳）
 * 策略：应用壳（child.html/manifest/icon）缓存优先；/api/ 与 /ws 一律直连不缓存，
 * 保证告警与通话数据实时（后台 10s 轮询即可唤醒 SW）。
 */
const CACHE = 'vericall-child-v22';
const SHELL = ['./child.html', './manifest.json', './icon-192.png', './icon-512.png',
  './theme.js?v=22', './ui.css?v=22', './ui.js?v=22', './copy.js'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return;
  const scopePath = new URL(self.registration.scope).pathname;
  const scopedPath = url.pathname.startsWith(scopePath)
    ? url.pathname.slice(scopePath.length)
    : url.pathname.replace(/^\//, '');
  if (scopedPath.startsWith('api/') || scopedPath.startsWith('ws/')) return; // API/WS 实时不缓存
  e.respondWith(
    caches.match(e.request).then((hit) => {
      if (hit) return hit;
      return fetch(e.request).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return resp;
      });
    })
  );
});
