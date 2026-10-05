/* =========================================================================
   VeriCall · 交互与视觉增强层 v2
   -------------------------------------------------------------------------
   以「旁挂」方式增强既有行为，不改动 ui.js / 内联脚本的任何业务逻辑：
   1. 波形画布未激活时隐藏（消除空白坏框）
   2. 分数更新时触发 bump 动效
   3. Tab 切换时为新面板重放入场动画
   4. 频谱 canvas 填充（品牌母题落到真实数据）
   5. 滚动进场（IntersectionObserver 渐进增强）
   ========================================================================= */
(() => {
  "use strict";

  const q = (s, r = document) => r.querySelector(s);
  const qa = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- 1. 波形画布：空闲时绘制静态占位波形 ---------- */
  /* canvas 是位图，CSS 无法在其上叠字或改背景（drawWave 会直接铺满底色）。
     正确做法：在空闲时用 JS 画一条「静默基线 + 待采集」，
     录音开始后 drawWave 覆盖它，视觉自然衔接。 */
  const waves = ["recWave", "liveWave"].map(id => document.getElementById(id)).filter(Boolean);

  const paintIdle = (c) => {
    const g = c.getContext("2d");
    const W = c.width, H = c.height;
    const css = getComputedStyle(document.documentElement);
    /* 专用 canvas 变量：不透明实色 + 随主题切换。
       不可用 --surface-*（半透明，叠在浅底会变白）。 */
    const bg = css.getPropertyValue("--ui-canvas-bg").trim() || "#07090b";
    const line = css.getPropertyValue("--ui-canvas-line").trim() || "rgba(236,239,235,.12)";
    const faint = css.getPropertyValue("--ui-canvas-faint").trim() || "#747c77";
    g.clearRect(0, 0, W, H);
    g.fillStyle = bg;
    g.fillRect(0, 0, W, H);
    // 中轴静默线
    g.strokeStyle = line; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, H / 2 + .5); g.lineTo(W, H / 2 + .5); g.stroke();
    // 两侧淡入淡出，暗示「等待信号」
    const grad = g.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, "transparent");
    grad.addColorStop(.5, line);
    grad.addColorStop(1, "transparent");
    g.strokeStyle = grad; g.lineWidth = 1;
    g.beginPath();
    for (let x = 0; x <= W; x += 2) {
      const t = x / W;
      const env = Math.sin(t * Math.PI);
      const jitter = Math.sin(t * 46) * .5 + Math.sin(t * 17) * .5;
      g.moveTo(x + .5, H / 2 - env * (1.2 + jitter * .8));
      g.lineTo(x + .5, H / 2 + env * (1.2 + jitter * .8));
    }
    g.stroke();
    // 角标
    g.fillStyle = faint;
    g.font = '10px -apple-system,"PingFang SC",sans-serif';
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText("待采集", W / 2, H / 2 - 13);
  };

  waves.forEach(paintIdle);

  /* 主题切换时重绘空闲占位，避免残留旧主题的底色 */
  const themeObserver = new MutationObserver(() => {
    waves.forEach(c => { if (!c.dataset.live) paintIdle(c); });
  });
  themeObserver.observe(document.documentElement, { attributeFilter: ["data-theme"] });

  /* 检测到真实波形后停止重绘。
     drawWave 画金色竖线，振幅通常 > 8px；我们的占位仅 ±2px。
     因此只统计「中轴上下 6px 带内出现明显偏离底色的像素」，
     并要求连续出现多次才判定，避免文字角标与细线造成误判。 */
  let idlePolls = 0;
  let hits = 0;
  const idleTimer = setInterval(() => {
    if (++idlePolls > 80) { clearInterval(idleTimer); return; }
    waves.forEach(c => {
      if (c.dataset.live) return;
      try {
        const g = c.getContext("2d");
        const W = c.width, H = c.height;
        const d = g.getImageData(0, 0, W, H).data;
        const bg = g.getImageData(2, 2, 1, 1).data;
        // 只取中轴 ±6px 带（排除上方 H/2-13 的文字）
        const y0 = (H >> 1) - 6, y1 = (H >> 1) + 6;
        let deep = 0;
        for (let y = y0; y <= y1; y += 2) {
          for (let x = 0; x < W; x += 3) {
            const i = (y * W + x) * 4;
            if (Math.abs(d[i] - bg[0]) + Math.abs(d[i + 1] - bg[1]) + Math.abs(d[i + 2] - bg[2]) > 60) deep++;
          }
        }
        // 真实波形在该带内会有大量偏离像素
        if (deep > 120) hits++; else hits = 0;
        if (hits >= 2) c.dataset.live = "1";
      } catch (_) { c.dataset.live = "1"; }
    });
    if (waves.every(c => c.dataset.live)) clearInterval(idleTimer);
  }, 300);

  /* ---------- 2. 分数 bump 动效 ---------- */
  const scoreIds = ["sc-acoustic", "sc-voiceprint", "sc-semantic", "gScore", "liveFused"];
  const bump = (el) => {
    if (!el || reduced) return;
    el.classList.remove("bump");
    void el.offsetWidth;      // 强制重排以重启动画
    el.classList.add("bump");
  };
  scoreIds.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    let last = null;
    new MutationObserver(() => {
      const v = el.textContent.trim();
      if (v && v !== "--" && v !== last) { last = v; bump(el); }
    }).observe(el, { childList: true, characterData: true, subtree: true });
  });

  /* ---------- 3. Tab 切换：重放入场动画 + h1 语义管理 ---------- */
  /* 五个 section 各自带一个 h1，但同一时刻只有一个面板可见。
     多个 h1 会让屏幕阅读器的标题导航失去主次（axe: heading-order）。
     规则：仅当前可见面板保留 h1，其余降级为 h2；
     切换时双向还原，保证任何时刻文档树里只有一个 h1。 */
  const demoteHeadings = (page) => {
    qa(".page").forEach(p => {
      const active = p === page;
      p.querySelectorAll("h1, h2[data-was-h1]").forEach(h => {
        const wantH1 = active;
        const isH1 = h.tagName === "H1";
        if (wantH1 === isH1) return;
        // tagName 是只读属性，必须替换节点而非赋值
        const next = document.createElement(wantH1 ? "h1" : "h2");
        next.innerHTML = h.innerHTML;
        // 复制除 id/class 外的属性与内联样式，避免样式丢失
        Array.from(h.attributes).forEach(a => {
          if (a.name !== "class") next.setAttribute(a.name, a.value);
        });
        next.dataset.wasH1 = "1";
        h.replaceWith(next);
      });
    });
  };
  const currentPage = () => q(".page.on");
  demoteHeadings(currentPage());

  /* tab 切换后需要重扫编排：切换前不可见的面板内元素从未被观察器触发。
     用可变钩子而非直接引用后面的 const，避免依赖声明顺序。 */
  let onAfterTab = () => {};
  const afterTabSwitch = () => onAfterTab();
  window.__vcAfterTabSwitch = afterTabSwitch;

  const tabs = qa(".tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.p;
      if (!target) return;
      const page = document.getElementById("p-" + target);
      if (!page) return;
      // 强制重排后重放，得到一致的进场节奏
      page.style.animation = "none";
      void page.offsetWidth;
      page.style.animation = "";
      demoteHeadings(page);
      // 新面板内的编排元素此前不可见、从未被观察器触发，此处补上
      requestAnimationFrame(afterTabSwitch);
    }, true);
  });

  /* ---------- 4. 频谱 canvas：真实数据驱动的品牌纹样 ---------- */
  /* 检测到页面有波形活动时，在其上方绘制一条频谱摘要带，
     让「声学」这一母题出现在数据发生的地方，而非纯装饰。 */
  const drawSpectrum = (canvas, energy) => {
    if (!canvas) return;
    const x = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    x.clearRect(0, 0, W, H);
    const bars = 40;
    const w = W / bars;
    const css = getComputedStyle(document.documentElement);
    const gold = css.getPropertyValue("--ui-gold").trim() || "#d4b483";
    for (let i = 0; i < bars; i++) {
      // 1/f 能量分布：模拟真实语音的频谱包络
      const t = i / bars;
      const env = Math.pow(1 - t, 0.55) * (0.72 + 0.28 * Math.sin(i * 1.7));
      const h = Math.max(1, env * (0.35 + energy * 0.65) * H);
      const a = 0.2 + env * 0.55;
      x.fillStyle = gold;
      x.globalAlpha = a;
      x.fillRect(i * w, H - h, Math.max(1, w - 1.5), h);
    }
    x.globalAlpha = 1;
  };
  window.__vcDrawSpectrum = drawSpectrum;

  /* ---------- 5. 滚动进场（多变体编排） ---------- */
  /* 主站同样存在「动效编排贫乏」：原先只有 fade+rise 一种动作。
     现按语义分配 5 种变体，由 orchestrate.css 提供错峰。

     关键坑：五 个 tab 面板中只有 .on 可见，其余 display:none。
     隐藏面板内的元素尺寸为 0，IntersectionObserver 永不触发，
     会永久停留在 opacity:0 —— 用户切到该 tab 时看到一片空白。
     因此必须：① 只对可见面板内的元素启用编排；
              ② tab 切换时重新扫描新面板。 */
  const assignVariants = () => {
    // 卡片 → scale（.ch 除外，见下）
    qa("#p-about .m-card, #p-about .pc, #p-about .rt-i, #p-about .dk, .panel").forEach(el => {
      if (!el.dataset.vcIn) el.dataset.vcIn = "scale";
    });
    /* .ch 的透明度与 transform 都是设计态的一部分：
       透明度 .62 = 待分析、1 = 已分析（JS 改写），
       且 JS 用 .ch.done 施加自己的 rise 动画。
       故编排层只做轻微位移（rise），且让出 opacity 控制权。 */
    qa(".ch").forEach(el => {
      el.dataset.vcIn = "rise";
      el.classList.add("vc-keep-opacity");
    });
    // 指标数字与线性元素 → wipe
    qa("#p-about .m-card .n, #p-about .dk .n, #p-about .m-wall, #p-about .m-more, .channels, .verdict").forEach(el => {
      if (!el.dataset.vcIn) el.dataset.vcIn = "wipe";
    });
    // 文本段落 → rise
    qa("#p-about .p-chan p, #p-about .rt-note, .psub, .workspace-head p, .hist-head .psub").forEach(el => {
      if (!el.dataset.vcIn) el.dataset.vcIn = "rise";
    });
    // 标题 → reveal
    qa("#p-about .m-card .l, #p-about .pc h3, #p-about .rt-t, .dash .dt").forEach(el => {
      if (!el.dataset.vcIn) el.dataset.vcIn = "reveal";
    });
  };
  assignVariants();

  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };

  const enableOrchestration = () => {
    if (!("IntersectionObserver" in window) || reduced) return;
    const targets = qa("[data-vc-in]");
    // 已经在可见面板内的立即播放（首屏无需等待观察器）
    // 隐藏面板内的跳过 —— 切到该 tab 时由 rescan 补上
    const live = targets.filter(el => isVisible(el));
    live.forEach(el => el.classList.add("is-in"));

    const pending = targets.filter(el => !el.classList.contains("is-in") && isVisible(el.parentElement || el));
    if (!pending.length) return;

    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        const el = e.target;
        const sibs = el.parentElement ? Array.from(el.parentElement.children) : [];
        const idx = Math.max(0, sibs.indexOf(el));
        el.style.animationDelay = Math.min(idx, 6) * 60 + "ms";
        el.classList.add("vc-play");
        obs.unobserve(el);
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -4% 0px" });
    pending.forEach(el => io.observe(el));
    // 兜底：1.6s 内未触发的强制可见
    setTimeout(() => {
      pending.forEach(el => { if (!el.classList.contains("vc-play")) { el.classList.add("is-in"); el.style.opacity = "1"; } });
    }, 1600);
  };

  // tab 切换后重扫：把新可见面板内的元素补上编排
  const rescan = () => {
    if (!("IntersectionObserver" in window) || reduced) return;
    qa("[data-vc-in]").forEach(el => {
      if (el.classList.contains("is-in") || el.classList.contains("vc-play")) return;
      if (!isVisible(el)) return;
      el.classList.add("is-in");
    });
  };

  document.documentElement.classList.add("vc-orchestrated");
  onAfterTab = rescan;
  enableOrchestration();

  /* ---------- 6. 悬停光标一致性 ---------- */
  /* 桌面用 pointer:fine 精确定位；触屏不显示 hover 态 */
  qa(".demo, .hchip, .sess button, .tl .b").forEach(el => {
    el.style.cursor = "pointer";
  });

  /* ---------- 7. 数字滚动：计数动画（仅大号指标） ---------- */
  const countUp = (el, to, dur = 900) => {
    if (!el || reduced) { if (el) el.textContent = to; return; }
    const from = 0;
    const t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (from + (to - from) * eased).toFixed(
        String(to).includes(".") ? String(to).split(".")[1].length : 0
      );
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  window.__vcCountUp = countUp;

  /* ---------- 8. 刷新 Service Worker 缓存版本 ---------- */
  /* 样式表已更新，需提升 sw 版本号让客户端取到新资源 */
  if ("serviceWorker" in navigator) {
    // 仅在 http(s) 下注册；file:// 与旧 sw 冲突时静默跳过
    if (location.protocol.startsWith("http")) {
      navigator.serviceWorker.getRegistration().then(reg => {
        if (reg) reg.update().catch(() => {});
      }).catch(() => {});
    }
  }
})();
