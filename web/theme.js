(() => {
  "use strict";

  const KEY = "vericall-theme";
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: light)");
  const subscribers = new Set();

  function storedTheme() {
    try {
      const value = localStorage.getItem(KEY);
      return value === "light" || value === "dark" ? value : "";
    } catch (_) {
      return "";
    }
  }

  function preferredTheme() {
    return storedTheme() || (media.matches ? "light" : "dark");
  }

  function themeColor(theme) {
    return theme === "light" ? "#f2f2f7" : "#000000";
  }

  function syncMeta(theme) {
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", themeColor(theme));
  }

  function apply(theme, options = {}) {
    const next = theme === "light" ? "light" : "dark";
    root.dataset.theme = next;
    root.style.colorScheme = next;
    if (options.persist !== false) {
      try {
        localStorage.setItem(KEY, next);
      } catch (_) {}
    }
    syncMeta(next);
    subscribers.forEach(callback => callback(next));
    try {
      if (typeof window.CustomEvent === "function") {
        window.dispatchEvent(new window.CustomEvent("vericall:themechange", {detail: {theme: next}}));
      }
    } catch (_) {}
    return next;
  }

  function toggle() {
    return apply(root.dataset.theme === "light" ? "dark" : "light");
  }

  function subscribe(callback) {
    if (typeof callback !== "function") return () => {};
    subscribers.add(callback);
    callback(root.dataset.theme || preferredTheme());
    return () => subscribers.delete(callback);
  }

  const bootstrap = document.createElement("style");
  bootstrap.id = "theme-bootstrap";
  bootstrap.textContent = `
html[data-theme="light"]{
  color-scheme:light;
  --ui-bg:#f2f2f7;
  --ui-bg-soft:#e5e5ea;
  --ui-surface:rgba(255,255,255,.82);
  --ui-surface-strong:rgba(255,255,255,.96);
  --ui-line:rgba(30,39,37,.12);
  --ui-line-strong:rgba(30,39,37,.22);
  --ui-text:#18201f;
  --ui-muted:#5b6663;
  --ui-faint:#7a8481;
  --ui-gold:#9a6a24;
  --ui-gold-soft:#744b13;
  --ui-teal:#147f75;
  --ui-teal-deep:#0e665e;
  --ui-blue:#007aff;
  --bg:#f2f2f7;
  --ink:#18201f;
  --mut:#5b6663;
  --dim:#7a8481;
  --gold:#9a6a24;
  --gold2:#744b13;
  --ok:#248a3d;
  --warn:#c93400;
  --bad:#d70015;
  --line:rgba(30,39,37,.1);
  --line2:rgba(30,39,37,.2);
  background:#f2f2f7;
}`;
  document.head.appendChild(bootstrap);

  apply(preferredTheme(), {persist: false});

  media.addEventListener?.("change", event => {
    if (!storedTheme()) apply(event.matches ? "light" : "dark", {persist: false});
  });

  window.addEventListener?.("storage", event => {
    if (event.key !== KEY) return;
    apply(event.newValue === "light" ? "light" : "dark", {persist: false});
  });

  document.addEventListener?.("DOMContentLoaded", () => syncMeta(root.dataset.theme));

  window.VeriCallTheme = {
    key: KEY,
    current: () => root.dataset.theme || "dark",
    isLight: () => root.dataset.theme === "light",
    apply,
    toggle,
    subscribe,
  };
})();
