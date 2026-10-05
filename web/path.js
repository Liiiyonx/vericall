/* Resolve app assets, API calls and WebSockets for both "/" and "/vericall/" deployments. */
(function () {
  "use strict";

  const source = document.currentScript && document.currentScript.src;
  const baseUrl = new URL("./", source || location.href);

  function relative(path) {
    return String(path || "").replace(/^\/+/, "");
  }

  function resolve(path) {
    return new URL(relative(path), baseUrl).href;
  }

  function websocket(path) {
    const url = new URL(relative(path), baseUrl);
    url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    return url.href;
  }

  window.VeriCallPath = {
    base: baseUrl.href,
    resolve,
    websocket,
  };
})();
