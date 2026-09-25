(() => {
  "use strict";

  const app = document.body.dataset.app || "index";
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || "");
  const commandKey = isMac ? "⌘K" : "Ctrl K";
  let palette;
  let paletteInput;
  let paletteList;
  let paletteItems = [];
  let activeIndex = 0;
  const dialogStack = [];
  const dialogState = new WeakMap();
  let pageWasInert = null;

  const tabList = [...document.querySelectorAll(".tab[data-p]")];

  const focusableSelector = [
    "a[href]",
    "button:not([disabled])",
    "input:not([disabled]):not([type='hidden'])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    "[tabindex]:not([tabindex='-1'])",
  ].join(",");

  function isElementVisible(element) {
    if (!element || element.hidden || element.getAttribute("aria-hidden") === "true") {
      return false;
    }
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" ||
        style.visibility === "collapse") {
      return false;
    }
    return element.getClientRects().length > 0;
  }

  function visibleFocusable(dialog) {
    return [...dialog.querySelectorAll(focusableSelector)].filter(isElementVisible);
  }

  function lockPage(activeDialog) {
    if (!pageWasInert) pageWasInert = new Map();
    [...document.body.children].forEach(element => {
      if (["SCRIPT", "STYLE", "LINK"].includes(element.tagName)) return;
      if (!pageWasInert.has(element)) pageWasInert.set(element, element.inert);
      element.inert = element !== activeDialog;
    });
    document.body.classList.add("ui-dialog-open");
  }

  function unlockPage() {
    if (pageWasInert) {
      pageWasInert.forEach((inert, element) => {
        element.inert = inert;
      });
      pageWasInert = null;
    }
    document.body.classList.remove("ui-dialog-open");
  }

  function focusFirst(dialog) {
    const preferred = dialog.querySelector("[data-dialog-autofocus]");
    const target = preferred || visibleFocusable(dialog)[0] || dialog;
    if (!target.hasAttribute("tabindex") && target === dialog) target.tabIndex = -1;
    requestAnimationFrame(() => {
      if (dialogStack.includes(dialog)) target.focus({preventScroll: true});
    });
  }

  function closeDialog(dialog, options = {}) {
    const index = dialogStack.indexOf(dialog);
    if (index === -1) return false;
    dialogStack.splice(index, 1);
    const state = dialogState.get(dialog) || {};
    dialog.classList.remove("show");
    dialog.setAttribute("aria-hidden", "true");
    delete dialog.dataset.dialogOpen;
    dialogState.delete(dialog);

    if (dialogStack.length) {
      lockPage(dialogStack[dialogStack.length - 1]);
    } else {
      unlockPage();
    }

    if (options.restoreFocus !== false && state.trigger &&
        typeof state.trigger.focus === "function" && state.trigger.isConnected) {
      requestAnimationFrame(() => state.trigger.focus({preventScroll: true}));
    }
    return true;
  }

  function openDialog(dialog, options = {}) {
    if (!dialog || dialogStack.includes(dialog)) return false;
    const trigger = Object.prototype.hasOwnProperty.call(options, "trigger")
      ? options.trigger
      : document.activeElement;
    dialogState.set(dialog, {trigger, onEscape: options.onEscape});
    dialogStack.push(dialog);
    dialog.classList.add("show");
    dialog.setAttribute("aria-hidden", "false");
    if (!dialog.hasAttribute("role")) dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.dataset.dialogOpen = "1";
    lockPage(dialog);
    focusFirst(dialog);
    return true;
  }

  function isDialogOpen(dialog) {
    return dialogStack.includes(dialog);
  }

  function closeTopDialog() {
    const dialog = dialogStack[dialogStack.length - 1];
    return dialog ? closeDialog(dialog) : false;
  }

  function setupDialogs() {
    document.addEventListener("keydown", event => {
      const dialog = dialogStack[dialogStack.length - 1];
      if (!dialog) return;

      if (event.key === "Escape") {
        if (dialog.dataset.dialogEscape === "false") return;
        event.preventDefault();
        event.stopImmediatePropagation();
        const onEscape = (dialogState.get(dialog) || {}).onEscape;
        if (typeof onEscape === "function") onEscape();
        else closeDialog(dialog);
        return;
      }

      if (event.key !== "Tab") return;
      const focusable = visibleFocusable(dialog);
      if (!focusable.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }, true);
  }

  window.VeriCallDialog = {
    open: openDialog,
    close: closeDialog,
    closeTop: closeTopDialog,
    isOpen: isDialogOpen,
    focusables: visibleFocusable,
  };

  function isTypingTarget(target) {
    if (!target) return false;
    return target.matches("input, textarea, select, [contenteditable='true']");
  }

  function shortCommandKey() {
    const key = document.querySelector(".command-trigger kbd");
    if (key) key.textContent = commandKey;
  }

  function setupTheme() {
    const theme = window.VeriCallTheme;
    const toggles = [...document.querySelectorAll("[data-theme-toggle]")];
    if (!theme || !toggles.length) return;

    toggles.forEach(button => {
      button.addEventListener("click", () => theme.toggle());
    });

    theme.subscribe(current => {
      const light = current === "light";
      const label = light ? "切换到深色模式" : "切换到浅色模式";
      toggles.forEach(button => {
        button.setAttribute("aria-label", label);
        button.setAttribute("title", label);
        button.setAttribute("aria-pressed", String(light));
      });
    });
  }

  function addSkipLink() {
    if (document.querySelector(".skip-link")) return;
    const target = document.getElementById("main-content")
      || document.querySelector("main")
      || document.querySelector(".wrap");
    if (!target) return;
    if (!target.id) target.id = "main-content";
    const link = document.createElement("a");
    link.className = "skip-link";
    link.href = "#" + target.id;
    link.textContent = "跳到主要内容";
    document.body.prepend(link);
    target.setAttribute("tabindex", "-1");
  }

  function labelForTab(tab) {
    return tab.dataset.label || tab.textContent.replace(/\s+/g, " ").trim();
  }

  function goToTab(tab, updateHash = true) {
    if (!tab) return;
    tab.click();
    if (updateHash && tab.dataset.p) {
      history.replaceState(null, "", "#" + tab.dataset.p);
    }
  }

  function restoreHash() {
    const wanted = location.hash.slice(1);
    if (!wanted) return;
    const tab = tabList.find(item => item.dataset.p === wanted);
    if (tab && !tab.classList.contains("on")) goToTab(tab, false);
  }

  function setupTabs() {
    if (!tabList.length) return;
    tabList.forEach((tab, index) => {
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", tab.classList.contains("on") ? "true" : "false");
      const panel = document.getElementById("p-" + tab.dataset.p);
      if (panel) {
        panel.setAttribute("role", "tabpanel");
        tab.setAttribute("aria-controls", panel.id);
      }
      tab.addEventListener("click", () => {
        tabList.forEach(item => item.setAttribute("aria-selected", item === tab ? "true" : "false"));
        if (isElementVisible(tab)) {
          tab.scrollIntoView({block: "nearest", inline: "nearest", behavior: "smooth"});
        }
      });
      tab.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let next = index;
        if (event.key === "ArrowLeft") next = (index - 1 + tabList.length) % tabList.length;
        if (event.key === "ArrowRight") next = (index + 1) % tabList.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabList.length - 1;
        tabList[next].focus();
        goToTab(tabList[next]);
      });
    });
  }

  function commandCandidates() {
    const items = [];
    tabList.forEach((tab, index) => {
      items.push({
        group: "导航",
        label: "前往 " + labelForTab(tab),
        hint: "切换当前工作区",
        key: String(index + 1),
        run: () => goToTab(tab),
      });
    });

    const addElementCommand = (selector, label, hint, key) => {
      const el = document.querySelector(selector);
      if (!el || el.disabled || !isElementVisible(el)) return;
      items.push({group: "操作", label, hint, key: key || "", run: () => el.click()});
    };

    if (app === "index") {
      addElementCommand("#recBtn", "开始或停止录音", "在当前设备采集一段音频", "R");
      addElementCommand("#fileIn", "上传音频文件", "支持 WAV、FLAC、MP3", "U");
      addElementCommand("#demoPlay", "运行 A 到 C 一键演示", "连续验证三种典型场景", "D");
      addElementCommand("#liveBtn", "开始或停止实时监测", "打开流式通话检测", "L");
      addElementCommand("#elderBtn", "切换适老显示", "放大文字和风险提示", "E");
      addElementCommand("#copyFromVerdict", "复制检测摘要", "把当前裁决整理成可转发的文字", "C");
      addElementCommand("#histExport", "导出检测记录 CSV", "导出当前筛选结果", "");
      addElementCommand('a[href$="child.html"]', "打开子女守护端", "进入家庭告警控制台", "");
      addElementCommand('a[href$="elder.html"]', "打开适老端全屏", "进入大字号守护界面", "");
    } else if (app === "child") {
      addElementCommand("#demoFill", "使用演示账号登录", "快速进入子女守护端", "D");
      addElementCommand("#eventExport", "导出拦截事件 CSV", "导出当前筛选结果", "E");
      addElementCommand("#eventQuery", "搜索拦截事件", "按来电、号码或结论检索", "/");
      addElementCommand("#btnLogout", "退出当前账号", "清除本机登录状态", "");
    } else if (app === "elder") {
      addElementCommand("#recBtn", "开始或结束录音", "录下通话并交给谛听判断", "R");
      addElementCommand("#liveBtn2", "开启或暂停常开守护", "说话即检测", "L");
      addElementCommand("#callChildBtn", "呼叫子女加入通话", "检测到风险时使用", "C");
    }

    if (window.VeriCallTheme) {
      items.push({
        group: "界面",
        label: window.VeriCallTheme.isLight() ? "切换到深色主题" : "切换到浅色主题",
        hint: "三端主题选择会保存在本机",
        key: "T",
        run: () => window.VeriCallTheme.toggle(),
      });
    }
    return items;
  }

  function createPalette() {
    palette = document.createElement("div");
    palette.className = "command-layer";
    palette.setAttribute("role", "dialog");
    palette.setAttribute("aria-modal", "true");
    palette.setAttribute("aria-label", "快速操作");
    palette.setAttribute("aria-hidden", "true");
    palette.innerHTML = `
      <div class="command-box">
        <div class="command-search">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
            <circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.8-3.8"></path>
          </svg>
          <input type="search" autocomplete="off" placeholder="搜索页面、动作与工具…" aria-label="搜索快速操作" data-dialog-autofocus>
          <kbd>ESC</kbd>
        </div>
        <div class="command-list" role="listbox"></div>
        <div class="command-foot">
          <span><kbd>↑↓</kbd> 选择</span>
          <span><kbd>Enter</kbd> 执行</span>
          <span><kbd>Esc</kbd> 关闭</span>
        </div>
      </div>`;
    document.body.appendChild(palette);
    paletteInput = palette.querySelector("input");
    paletteList = palette.querySelector(".command-list");

    palette.addEventListener("mousedown", event => {
      if (event.target === palette) closePalette();
    });
    paletteInput.addEventListener("input", renderPalette);
    paletteInput.addEventListener("keydown", event => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveIndex(activeIndex + 1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveIndex(activeIndex - 1);
      } else if (event.key === "Enter" && paletteItems[activeIndex]) {
        event.preventDefault();
        runPaletteItem(paletteItems[activeIndex]);
      }
    });
  }

  function normalized(text) {
    return String(text || "").toLowerCase().replace(/\s+/g, "");
  }

  function renderPalette() {
    const query = normalized(paletteInput.value);
    const all = commandCandidates();
    paletteItems = query
      ? all.filter(item => normalized(item.label + item.hint + item.group).includes(query))
      : all;
    activeIndex = Math.min(activeIndex, Math.max(0, paletteItems.length - 1));
    paletteList.innerHTML = "";
    if (!paletteItems.length) {
      paletteList.innerHTML = '<div class="command-empty">没有匹配的操作</div>';
      return;
    }

    let lastGroup = "";
    paletteItems.forEach((item, index) => {
      if (item.group !== lastGroup) {
        const heading = document.createElement("div");
        heading.className = "command-section";
        heading.textContent = item.group.toUpperCase();
        paletteList.appendChild(heading);
        lastGroup = item.group;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "command-item" + (index === activeIndex ? " active" : "");
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", index === activeIndex ? "true" : "false");
      const glyph = item.group === "导航" ? "↗" : "•";
      button.innerHTML = `
        <span class="icon" aria-hidden="true">${glyph}</span>
        <span class="copy"><strong></strong><span></span></span>
        <span class="key"></span>`;
      button.querySelector("strong").textContent = item.label;
      button.querySelector(".copy span").textContent = item.hint;
      button.querySelector(".key").textContent = item.key || "";
      button.addEventListener("mouseenter", () => {
        activeIndex = index;
        paintActiveItem();
      });
      button.addEventListener("click", () => runPaletteItem(item));
      paletteList.appendChild(button);
    });
    requestAnimationFrame(() => {
      const active = paletteList.querySelector(".command-item.active");
      if (active) active.scrollIntoView({block: "nearest"});
    });
  }

  function paintActiveItem() {
    const buttons = [...paletteList.querySelectorAll(".command-item")];
    buttons.forEach((button, index) => {
      button.classList.toggle("active", index === activeIndex);
      button.setAttribute("aria-selected", index === activeIndex ? "true" : "false");
    });
    const active = buttons[activeIndex];
    if (active) active.scrollIntoView({block: "nearest"});
  }

  function setActiveIndex(index) {
    if (!paletteItems.length) return;
    activeIndex = (index + paletteItems.length) % paletteItems.length;
    paintActiveItem();
  }

  function runPaletteItem(item) {
    closePalette();
    requestAnimationFrame(() => item.run());
  }

  function openPalette() {
    if (!palette) createPalette();
    if (window.VeriCallDialog?.isOpen(palette)) return;
    paletteInput.value = "";
    activeIndex = 0;
    renderPalette();
    document.querySelectorAll("[data-command-open]").forEach(button => button.setAttribute("aria-expanded", "true"));
    window.VeriCallDialog?.open(palette, {onEscape: closePalette});
  }

  function closePalette() {
    if (!palette?.classList.contains("show")) return;
    if (window.VeriCallDialog?.isOpen(palette)) {
      window.VeriCallDialog.close(palette);
    } else {
      palette.classList.remove("show");
      palette.setAttribute("aria-hidden", "true");
    }
    document.querySelectorAll("[data-command-open]").forEach(button => button.setAttribute("aria-expanded", "false"));
  }

  function setupShortcuts() {
    const commandTriggers = [...document.querySelectorAll("[data-command-open]")];

    commandTriggers.forEach(button => {
      button.setAttribute("aria-haspopup", "dialog");
      button.setAttribute("aria-expanded", "false");
      button.addEventListener("click", openPalette);
    });

    document.addEventListener("keydown", event => {
      const commandVisible = commandTriggers.some(isElementVisible);
      const commandPressed = isMac ? event.metaKey : event.ctrlKey;
      if (commandPressed && event.key.toLowerCase() === "k") {
        if (!commandVisible) return;
        event.preventDefault();
        palette?.classList.contains("show") ? closePalette() : openPalette();
        return;
      }
      if (event.key === "Escape") {
        closePalette();
        return;
      }
      if (isTypingTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "/") {
        if (!commandVisible) return;
        const search = document.querySelector(".hist-search input, .event-search input");
        if (search) {
          event.preventDefault();
          search.focus();
        } else {
          event.preventDefault();
          openPalette();
        }
        return;
      }
      if (event.key.toLowerCase() === "t" && window.VeriCallTheme) {
        event.preventDefault();
        window.VeriCallTheme.toggle();
        return;
      }
      const index = Number(event.key) - 1;
      const visibleTabs = tabList.filter(isElementVisible);
      if (index >= 0 && index < visibleTabs.length) {
        event.preventDefault();
        goToTab(visibleTabs[index]);
      }
    });
  }

  function setupDropZone() {
    const zone = document.getElementById("dropZone");
    if (!zone || typeof window.analyzeFile !== "function") return;
    let dragDepth = 0;
    ["dragenter", "dragover"].forEach(type => zone.addEventListener(type, event => {
      event.preventDefault();
      if (type === "dragenter") dragDepth++;
      zone.classList.add("dragging");
    }));
    zone.addEventListener("dragleave", event => {
      event.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) zone.classList.remove("dragging");
    });
    zone.addEventListener("drop", event => {
      event.preventDefault();
      dragDepth = 0;
      zone.classList.remove("dragging");
      const file = event.dataTransfer?.files?.[0];
      if (!file) return;
      if (!file.type.startsWith("audio/") && !/\.(wav|flac|mp3)$/i.test(file.name)) {
        if (typeof window.toast === "function") window.toast("请选择音频文件", true);
        return;
      }
      const fileInput = document.getElementById("fileIn");
      if (fileInput) fileInput.value = "";
      const name = document.getElementById("fileName");
      if (name) name.textContent = file.name;
      window.analyzeFile(file, file.name);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    addSkipLink();
    shortCommandKey();
    setupTheme();
    setupTabs();
    setupDialogs();
    setupShortcuts();
    setupDropZone();
    restoreHash();
  });
})();

(() => {
  "use strict";

  const TOKEN_KEY = "vericall_child_token";
  const ELDER_KEY = "vericall_elder_access_key";

  async function websocketUrl() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const directUrl = `${protocol}//${location.host}/ws/stream`;
    const elderKey = sessionStorage.getItem(ELDER_KEY);
    const token = localStorage.getItem(TOKEN_KEY);
    if (!elderKey && !token) return directUrl;

    const headers = elderKey
      ? {"X-Elder-Access-Key": elderKey}
      : {Authorization: `Bearer ${token}`};

    const response = await fetch("/api/stream/ws-ticket", {
      method: "POST",
      headers,
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(response.status === 401 ? "登录已过期，请重新登录" : "无法获取安全连接票据");
    }
    const payload = await response.json();
    if (!payload.ticket) throw new Error("服务未返回安全连接票据");
    return `${directUrl}?ticket=${encodeURIComponent(payload.ticket)}`;
  }

  window.VeriCallStream = {websocketUrl};
})();
