/* 谛听 VeriCall · 子女守护端 Service Worker（PWA 离线壳）
 * 策略：静态资源网络优先并回填缓存，离线时回退缓存；/api/ 与 /ws 一律直连不缓存。
 */
const CACHE = 'vericall-child-v25';
const SHELL = ['./child.html', './manifest.json', './icon-192.png', './icon-512.png',
  './path.js?v=25', './theme.js?v=25', './ui.css?v=25', './ui.js?v=25', './copy.js'];

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
    fetch(e.request).then((resp) => {
      if (resp && resp.ok) {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
      }
      return resp;
    }).catch(async () => {
      const hit = await caches.match(e.request);
      if (hit) return hit;
      if (e.request.mode === 'navigate') return caches.match('./child.html');
      throw new Error('offline');
    })
  );
});
