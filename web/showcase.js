/* =========================================================================
   谛听 VeriCall · 展示页频谱场
   -------------------------------------------------------------------------
   原创视觉母题。设计依据：
   语音信号的能量按 1/f 分布 —— 低频（0–1kHz，基频与共振峰）能量最密，
   高频（>4kHz， fricative 擦音）快速衰减。据此生成频谱包络，
   使视觉形态与「声音」这一主题存在真实对应，而非任意装饰。

   渲染策略（性能）：
   - 单 canvas，一次性按设备像素比缩放，不随窗口反复重建
   - 只在尺寸变化时重绘静态层；动画用 requestAnimationFrame 驱动
   - 页面不可见时暂停动画（IntersectionObserver + visibilitychange）
   - 粒子数按视口面积自适应，窄屏自动降档
   ========================================================================= */
(() => {
  "use strict";

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- 频谱包络：1/f 能量分布 ---------- */
  /* 返回 0..1 的能量值，t ∈ [0,1] 表示归一化频率 */
  const envelope = (t) => {
    // 基频峰（说话人基频集中在低频）
    const fundamental = Math.exp(-Math.pow((t - 0.06) / 0.09, 2)) * 0.55;
    // 共振峰群：人声的 F1/F2/F3 集中在 0.1–0.4
    const formants =
      Math.exp(-Math.pow((t - 0.18) / 0.07, 2)) * 0.42 +
      Math.exp(-Math.pow((t - 0.30) / 0.09, 2)) * 0.30 +
      Math.exp(-Math.pow((t - 0.44) / 0.11, 2)) * 0.18;
    // 1/f 衰减底噪
    const pink = 0.16 / Math.pow(t + 0.14, 0.62);
    // 高频滚降（超过奈奎斯特实用范围后急降）
    const rolloff = t > 0.7 ? Math.max(0, 1 - (t - 0.7) / 0.3) : 1;

    return Math.min(1, (fundamental + formants + pink) * rolloff);
  };

  /* ---------- 主题感知取色 ---------- */
  const readTheme = () => {
    const css = getComputedStyle(document.documentElement);
    const v = (n, fb) => (css.getPropertyValue(n).trim() || fb);
    const theme = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    return {
      theme,
      gold: v("--ui-gold", "#d4b483"),
      teal: v("--ui-teal", "#70d4c6"),
      line: v("--ui-line-strong", "rgba(236,239,235,.2)"),
      bg: v("--ui-bg", "#07090b"),
      /* 图形专用配色。浅色主题下 --ui-gold 是为「文字可读」而加深的深金
         （#9a6f1f），拿它画 1px 细线在浅底上几乎不可见。
         故为图形单独取饱和度更高的金/墨，线宽也相应加大。 */
      goldLine: theme === "light" ? "#a8790a" : v("--ui-gold", "#d4b483"),
      tealLine: theme === "light" ? "#0b6f66" : v("--ui-teal", "#70d4c6"),
      lineWidth: theme === "light" ? 1.5 : 1.15,
      boost: theme === "light" ? 1.5 : 1,   // 浅底需要更高 alpha 才有存在感
    };
  };

  const rgba = (hex, a) => {
    const h = hex.replace("#", "").trim();
    const full = h.length === 3 ? h.split("").map(c => c + c).join("") : h;
    const n = parseInt(full, 16);
    if (Number.isNaN(n)) return `rgba(212,180,131,${a})`;
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  /* ---------- 频谱场：Hero / 收尾 ---------- */
  class SpectrumField {
    constructor(canvas, opts = {}) {
      this.cv = canvas;
      this.ctx = canvas.getContext("2d", { alpha: true });
      this.opts = Object.assign({ lines: 46, baseAlpha: 0.52, speed: 1 }, opts);
      this.t0 = performance.now();
      this.visible = true;
      this.running = false;
      this.dpr = 1;

      this._onResize = this._onResize.bind(this);
      this._frame = this._frame.bind(this);

      if ("ResizeObserver" in window) {
        this.ro = new ResizeObserver(this._onResize);
        this.ro.observe(this.cv);
      } else {
        window.addEventListener("resize", this._onResize);
      }

      // 不可见时停机：避免后台标签页持续耗电
      document.addEventListener("visibilitychange", () => {
        this.visible = !document.hidden;
        if (this.visible) this.start(); else this.stop();
      });

      this._onResize();
    }

    _onResize() {
      const r = this.cv.getBoundingClientRect();
      if (!r.width || !r.height) return;
      this.dpr = Math.min(2, window.devicePixelRatio || 1);
      this.cv.width = Math.round(r.width * this.dpr);
      this.cv.height = Math.round(r.height * this.dpr);
      this.w = r.width;
      this.h = r.height;
      // 窄屏降档，保证 60fps
      this.lines = Math.round(
        this.opts.lines * Math.min(1, Math.max(0.45, r.width / 1200))
      );
      this.draw(0);
    }

    start() {
      if (this.running || reduced) { this.draw((performance.now() - this.t0) / 1000); return; }
      this.running = true;
      requestAnimationFrame(this._frame);
    }

    stop() { this.running = false; }

    _frame(now) {
      if (!this.running) return;
      this.draw((now - this.t0) / 1000);
      requestAnimationFrame(this._frame);
    }

    draw(time) {
      const { ctx, w, h, dpr, lines, opts } = this;
      if (!w || !h) return;
      const th = readTheme();
      const t = time * 0.00022 * opts.speed;

      ctx.clearRect(0, 0, this.cv.width, this.cv.height);
      ctx.save();
      ctx.scale(dpr, dpr);

      const baseY = h * 0.5;
      const maxAmp = h * 0.34;
      const lh = h / (lines + 1);

      /* 水平中心：桌面偏右（78%），与 CSS 遮罩呼应，让左侧文字区保持干净；
         窄屏回到居中，否则频谱会整体滑出可视区。 */
      const cx = w >= 900 ? w * 0.78 : w * 0.5;
      const wLine = Math.min(w * 0.62, w * (w >= 900 ? 0.5 : 0.86));
      const x0 = cx - wLine / 2;

      for (let i = 0; i < lines; i++) {
        const p = i / (lines - 1);
        // 每行的频率中心：从高频到低频（自上而下）
        const freqT = 1 - p;
        const env = envelope(freqT);
        const y = baseY - p * h * 0.72;

        // 慢速相位漂移，制造「正在聆听」的呼吸感
        const phase = t * (0.6 + freqT * 1.4) + i * 0.42;
        // 双频叠加，接近真实声音的拍频
        const mod =
          Math.sin(phase) * 0.5 +
          Math.sin(phase * 2.37 + 1.1) * 0.3 +
          Math.sin(phase * 0.61 + 2.2) * 0.2;

        const amp = env * maxAmp * (0.5 + mod * 0.5);

        // 低频（靠近基线）用金色，高频用青色 —— 沿用全站色彩分工
        const useGold = freqT < 0.45;
        const col = useGold ? th.goldLine : th.tealLine;
        const alpha = Math.min(0.92,
          opts.baseAlpha * th.boost * (0.3 + env * 0.8) * (useGold ? 1 : 0.55));

        ctx.strokeStyle = rgba(col, Math.max(0.05, alpha));
        ctx.lineWidth = th.lineWidth * (useGold ? 1 : 0.8);
        ctx.beginPath();

        // 逐点采样：左右两端衰减，避免生硬的方波边缘
        const steps = 96;
        for (let s = 0; s <= steps; s++) {
          const u = s / steps;
          const x = x0 + u * wLine;
          // 高斯窗：中心最实，两端收敛
          const win = Math.exp(-Math.pow((u - 0.5) / 0.32, 2) * 2.1);
          // 细节：叠加高频纹理，避免看起来像纯正弦
          const grain =
            Math.sin(u * 34 + phase * 5.3) * 0.14 +
            Math.sin(u * 71 - phase * 3.1) * 0.08;
          const dy = (amp * win * (1 + grain)) * 0.5;
          const yy = y - dy + (i % 2 ? 0.5 : -0.5) * lh * 0.16;
          if (s === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
        }
        ctx.stroke();
      }

      // 中心轴：一条极淡的基线，暗示「信号基准」
      ctx.strokeStyle = rgba(th.goldLine, 0.14);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.max(0, x0 - w * 0.06), baseY + lh * 0.5);
      ctx.lineTo(Math.min(w, x0 + wLine + w * 0.06), baseY + lh * 0.5);
      ctx.stroke();

      ctx.restore();
    }

    destroy() {
      this.stop();
      if (this.ro) this.ro.disconnect();
      else window.removeEventListener("resize", this._onResize);
    }
  }

  /* ---------- 迷你频谱条：通道卡内 ---------- */
  /* 注意：.sc-spark 在 HTML 中是 <div>，无 getContext。
     此处按其 getBoundingClientRect 尺寸新建离屏 canvas 再绘制，
     最后以 data-URL 设为背景图 —— 保持 HTML 无需改动。 */
  const drawSpark = (el, variant) => {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const off = document.createElement("canvas");
    off.width = Math.round(r.width * dpr);
    off.height = Math.round(r.height * dpr);
    const g = off.getContext("2d");
    if (!g) return;
    g.scale(dpr, dpr);
    const W = r.width, H = r.height;
    const th = readTheme();
    const n = 46;
    const bw = W / n;

    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      // 三个通道用不同的频段偏移，形成区分
      const shift = (variant - 1) * 0.14;
      const e = envelope(Math.max(0.02, Math.min(0.98, 0.5 - t * 0.42 + shift)));
      const h = Math.max(1.5, e * H * 0.94);
      const col = t < 0.45 ? th.goldLine : th.tealLine;
      g.fillStyle = rgba(col, 0.22 + e * 0.5);
      g.fillRect(i * bw, H - h, Math.max(1, bw - 1.4), h);
    }
    el.style.backgroundImage = `url(${off.toDataURL("image/png")})`;
    el.style.backgroundSize = "100% 100%";
    el.style.backgroundRepeat = "no-repeat";
  };

  /* ---------- 启动 ---------- */
  const init = () => {
    // 频谱场
    qsa("#scField, #scField2").forEach(cv => {
      const f = new SpectrumField(cv, { lines: 46, baseAlpha: 0.52, speed: 1 });
      // 进入视口才启动，首屏之外的场不空转
      if ("IntersectionObserver" in window) {
        const io = new IntersectionObserver((es) => {
          es.forEach(e => { e.isIntersecting ? f.start() : f.stop(); });
        }, { threshold: 0.01 });
        io.observe(cv);
      } else f.start();
      // 主题切换后重绘配色
      new MutationObserver(() => f.draw((performance.now() - f.t0) / 1000))
        .observe(document.documentElement, { attributeFilter: ["data-theme"] });
    });

    // 迷你频谱：进入视口时绘制（此时才有正确宽度）
    const sparks = qsa(".sc-spark");
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es, obs) => {
        es.forEach(e => {
          if (!e.isIntersecting) return;
          drawSpark(e.target, Number(e.target.dataset.spark) || 1);
          obs.unobserve(e.target);
        });
      }, { threshold: 0.2 });
      sparks.forEach(s => io.observe(s));
    } else {
      sparks.forEach(s => drawSpark(s, Number(s.dataset.spark) || 1));
    }

    // 滚动进场（多变体编排）
    // 诊断发现：原实现只有一种「淡入+上移」，且同区块元素几乎同时进场。
    // 现改为 5 种变体按语义分配，由 orchestrate.css 提供错峰编排。
    const applyVariants = () => {
      // 卡片型 → scale（独立物件感）
      qsa(".sc-pcard, .sc-chan, .sc-hon-row").forEach(el => {
        if (!el.dataset.vcIn) el.dataset.vcIn = "scale";
      });
      // 导语段 → rise
      qsa(".sc-sec-lead, .sc-chan-d, .sc-ev-note p, .sc-hon-b p, .sc-hon-foot").forEach(el => {
        if (!el.dataset.vcIn) el.dataset.vcIn = "rise";
      });
      // 频谱条 / 表格行 → wipe（线性元素用擦入）
      qsa(".sc-spark, .sc-ev-row").forEach(el => {
        if (!el.dataset.vcIn) el.dataset.vcIn = "wipe";
      });
      // 容器本身 → rise
      qsa(".sc-sec-head, .sc-problem, .sc-chans, .sc-ev, .sc-honest, .sc-final-inner").forEach(el => {
        if (!el.dataset.vcIn) el.dataset.vcIn = "rise";
      });
    };
    applyVariants();

    const reveals = qsa("[data-vc-in]");
    if (reduced || !("IntersectionObserver" in window)) {
      // 无动效环境：直接全部可见
      reveals.forEach(el => el.classList.add("is-in"));
    } else {
      document.documentElement.classList.add("vc-orchestrated");
      const io = new IntersectionObserver((es, obs) => {
        es.forEach(e => {
          if (!e.isIntersecting) return;
          const el = e.target;
          // 错峰：同父容器内的兄弟序号决定延迟
          const sibs = el.parentElement ? Array.from(el.parentElement.children) : [];
          const idx = sibs.indexOf(el);
          // 超过 6 个后封顶，避免长列表末尾等太久
          el.style.animationDelay = Math.min(Math.max(idx, 0), 6) * 70 + "ms";
          el.classList.add("is-in");
          obs.unobserve(el);
        });
      }, { threshold: 0.12, rootMargin: "0px 0px -5% 0px" });
      reveals.forEach(el => io.observe(el));
      // 兜底：2s 内未进入视口的一律显示，杜绝内容不可见
      setTimeout(() => reveals.forEach(el => el.classList.add("is-in")), 2000);
    }

    // 导航吸顶阴影
    const nav = document.querySelector(".sc-nav");
    if (nav) {
      const onScroll = () => nav.classList.toggle("is-stuck", window.scrollY > 8);
      window.addEventListener("scroll", onScroll, { passive: true });
      onScroll();
    }

    // 平滑滚动（尊重减弱动效偏好）
    qsa(".sc-navlinks a").forEach(a => {
      a.addEventListener("click", (e) => {
        const id = a.getAttribute("href");
        if (!id || !id.startsWith("#")) return;
        const t = document.querySelector(id);
        if (!t) return;
        e.preventDefault();
        t.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
        history.replaceState(null, "", id);
      });
    });

    // 主题切换后重绘迷你频谱（配色随主题变）
    new MutationObserver(() => {
      sparks.forEach(s => { if (s.getBoundingClientRect().width) drawSpark(s, Number(s.dataset.spark) || 1); });
    }).observe(document.documentElement, { attributeFilter: ["data-theme"] });
  };

  function qsa(sel) { return Array.from(document.querySelectorAll(sel)); }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
