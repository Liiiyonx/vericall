#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听VeriCall - LA.zip 下载实时看板 (零依赖, 仅用 Python 标准库)

启动后访问 http://localhost:8765 即可看到实时进度条 / 速度 / ETA。
会自动探测 curl 进程是否存活, 并在下载完成后提示可开训。

用法:
    vericall/python.exe scripts/download_monitor.py
可选环境变量:
    VERICALL_LA_ZIP  LA.zip 路径 (默认 <VERICALL_DATA_ROOT>/LA.zip, 见 src/paths.py)
    LA_ZIP           同上, 旧变量名, 优先级更高, 保持向后兼容
    PORT             监听端口 (默认 8765)
    TOTAL_BYTES      总字节数 (默认 6320000000, 即 ASVspoof2019 LA 实际 ~6.32GB)
"""
import json
import os
import sys
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import LA_ZIP as DEFAULT_LA_ZIP  # noqa: E402

LA_ZIP = os.environ.get("LA_ZIP") or str(DEFAULT_LA_ZIP)
PORT = int(os.environ.get("PORT", "8765"))
# ASVspoof2019 LA 实际文件大小 ~6.32 GB (datashare 下载落地页标注; 注意早期 curl 进度里
# 出现的 5.55G 是误读值, 真实总大小以此为基准。可用 TOTAL_BYTES 环境变量覆盖)
TOTAL_BYTES = int(os.environ.get("TOTAL_BYTES", "6320000000"))

# 速度滑动窗口 (size=None 表示尚未初始化, 跳过首帧以防虚假速度)
_last = {"ts": time.time(), "size": None}
_ema_speed = None  # 指数滑动平均, 平滑 ETA 显示


def _proc_alive() -> bool:
    """多方式探测 curl 进程, 任一成功即认为存活"""
    try:
        out = subprocess.run("ps aux", shell=True, capture_output=True, text=True,
                             timeout=3).stdout
        if "curl" in out and "LA.zip" in out:
            return True
    except Exception:
        pass
    for cmd in ('tasklist /FI "IMAGENAME eq curl.exe"', "tasklist"):
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                 timeout=3).stdout
            if "curl.exe" in out:
                return True
        except Exception:
            pass
    return False


def _sample():
    global _last
    now = time.time()
    try:
        size = os.path.getsize(LA_ZIP) if os.path.exists(LA_ZIP) else 0
    except OSError:
        size = 0

    # 首帧: 仅初始化窗口, 不计算速度, 避免 0->当前 的虚假峰值
    prev_size = _last["size"]
    if prev_size is None:
        _last = {"ts": now, "size": size}
        speed = 0.0
    else:
        dt = now - _last["ts"]
        if dt > 0.2 and size >= prev_size:
            speed = (size - prev_size) / dt
        else:
            speed = 0.0
        _last = {"ts": now, "size": size}

    percent = min(100.0, size / TOTAL_BYTES * 100.0) if TOTAL_BYTES else 0.0
    remaining = max(0, TOTAL_BYTES - size)
    # 判活: 进程在 OR 文件在最近窗口内有增长
    grown = prev_size is not None and size > prev_size
    alive = _proc_alive() or grown
    # 平滑速度 (EMA), 避免 ETA 剧烈抖动
    global _ema_speed
    if speed > 1:
        _ema_speed = speed if _ema_speed is None else 0.6 * _ema_speed + 0.4 * speed
    disp_speed = _ema_speed if _ema_speed else speed
    eta = int(remaining / disp_speed) if disp_speed > 1 else None
    return {
        "size": size,
        "total": TOTAL_BYTES,
        "percent": round(percent, 2),
        "speed_bps": round(disp_speed, 1),
        "eta_seconds": eta,
        "curl_alive": alive,
        # 完成判定: 进程已退 AND 文件不再增长 (且已达可观体积) —— 比"达到某字节阈值"
        # 更可靠, 避免早期把 5.55G 当总大小导致提前误报 100%
        "complete": (not alive) and (not grown) and size > 100_000_000,
        "ts": int(now),
    }


def _fmt_size(b):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.2f} {unit}" if unit == "B" else f"{b/1024**(['B','KB','MB','GB','TB'].index(unit)):.2f} {unit}"
    return f"{b:.2f} B"


def _fmt_speed(bps):
    if bps <= 0:
        return "—"
    if bps >= 1024 ** 2:
        return f"{bps/1024**2:.2f} MB/s"
    if bps >= 1024:
        return f"{bps/1024:.1f} KB/s"
    return f"{bps:.0f} B/s"


def _fmt_eta(sec):
    if sec is None:
        return "计算中…"
    if sec <= 0:
        return "即将完成"
    h, m = divmod(sec, 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{h} 小时 {m} 分"
    if m:
        return f"{m} 分 {s} 秒"
    return f"{s} 秒"


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>谛听VeriCall · LA.zip 下载实时看板</title>
<style>
  :root{ --bg:#0f1420; --card:#1a2233; --accent:#3ddc97; --warn:#ffb454; --txt:#e6edf3; --sub:#8b98a9; }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       background:radial-gradient(1200px 600px at 50% -10%,#1c2740,#0f1420);color:var(--txt);
       min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
  .card{background:var(--card);border:1px solid #2a3650;border-radius:18px;padding:32px 36px;
        width:min(620px,92vw);box-shadow:0 20px 60px rgba(0,0,0,.45)}
  h1{margin:0 0 4px;font-size:20px;letter-spacing:.5px}
  .sub{color:var(--sub);font-size:13px;margin-bottom:22px}
  .barwrap{background:#0c1018;border-radius:12px;height:30px;overflow:hidden;position:relative;border:1px solid #2a3650}
  .bar{height:100%;width:0;background:linear-gradient(90deg,#2bb673,#3ddc97);
       transition:width .6s ease;border-radius:12px 0 0 12px}
  .pct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
       font-weight:700;font-size:14px;text-shadow:0 1px 2px rgba(0,0,0,.6)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:22px}
  .item{background:#0c1018;border:1px solid #232f47;border-radius:12px;padding:14px 16px}
  .k{color:var(--sub);font-size:12px;margin-bottom:6px}
  .v{font-size:18px;font-weight:700}
  .status{margin-top:20px;font-size:14px;display:flex;align-items:center;gap:10px}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent)}
  .dot.off{background:var(--warn);box-shadow:0 0 10px var(--warn)}
  .tip{margin-top:18px;font-size:12.5px;color:var(--sub);line-height:1.6}
  code{background:#0c1018;padding:2px 6px;border-radius:6px;color:var(--accent);border:1px solid #232f47}
  .done{color:var(--accent);font-weight:700}
</style>
</head>
<body>
  <div class="card">
    <h1>谛听VeriCall · LA.zip 下载看板</h1>
    <div class="sub">ASVspoof2019 LA 数据集 · 后台断点续传 · 自动刷新</div>
    <div class="barwrap">
      <div class="bar" id="bar"></div>
      <div class="pct" id="pct">0%</div>
    </div>
    <div class="grid">
      <div class="item"><div class="k">已下载</div><div class="v" id="size">—</div></div>
      <div class="item"><div class="k">总大小</div><div class="v" id="total">—</div></div>
      <div class="item"><div class="k">实时速度</div><div class="v" id="speed">—</div></div>
      <div class="item"><div class="k">预计剩余</div><div class="v" id="eta">—</div></div>
    </div>
    <div class="status">
      <span class="dot" id="dot"></span>
      <span id="statusText">检测中…</span>
    </div>
    <div class="tip" id="tip">
      下载完成后, 在 vericall 环境运行
      <code>python scripts/launch_training.py</code> 即可自动解压并启动 AASIST 通道①训练。
    </div>
  </div>
<script>
function fmt(n){const u=['B','KB','MB','GB','TB'];let i=0;let x=n;while(x>=1024&&i<u.length-1){x/=1024;i++}return x.toFixed(2)+' '+u[i]}
function tick(){
  fetch('/status?t='+Date.now()).then(r=>r.json()).then(d=>{
    document.getElementById('bar').style.width=d.percent+'%';
    document.getElementById('pct').textContent=d.percent.toFixed(1)+'%';
    document.getElementById('size').textContent=fmt(d.size);
    document.getElementById('total').textContent=fmt(d.total);
    if(d.speed_bps>1024*1024)document.getElementById('speed').textContent=(d.speed_bps/1024/1024).toFixed(2)+' MB/s';
    else if(d.speed_bps>0)document.getElementById('speed').textContent=(d.speed_bps/1024).toFixed(1)+' KB/s';
    else document.getElementById('speed').textContent='—';
    let eta='计算中…';
    if(d.eta_seconds!=null){const s=d.eta_seconds;const h=Math.floor(s/3600),m=Math.floor(s%3600/60);eta=h?`${h} 小时 ${m} 分`:(m?`${m} 分 ${s%60} 秒`:`${s} 秒`);}
    document.getElementById('eta').textContent=eta;
    const dot=document.getElementById('dot'),st=document.getElementById('statusText'),tip=document.getElementById('tip');
    if(d.complete){dot.className='dot';st.innerHTML='<span class="done">✓ 下载完成, 可开训</span>';}
    else if(d.curl_alive){dot.className='dot';st.textContent='下载中 (curl 进程存活)';}
    else{dot.className='dot off';st.textContent='⚠ curl 进程未检测到 — 下载可能已停, 喊“继续”重接续传';}
  }).catch(()=>{document.getElementById('statusText').textContent='看板连接中断, 刷新页面重试';});
}
tick();setInterval(tick,2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status"):
            self._send(200, json.dumps(_sample()).encode("utf-8"))
        elif self.path.startswith("/") :
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found")

    def log_message(self, *a):
        pass  # 静默


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[看板] 已启动: http://localhost:{PORT}")
    print(f"[看板] 监控文件: {LA_ZIP}")
    print(f"[看板] 总大小≈ {TOTAL_BYTES/1024**3:.2f} GB")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[看板] 已停止")


if __name__ == "__main__":
    main()
