#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 训练实时看板 HTTP 服务。

为什么需要它:
  training_dashboard.html 里的浏览器 JS 无法直接读本地文件(训练日志、
  桌面 exp_result/.../weights/), 必须通过 HTTP 拉取。本脚本是一个极简只读服务,
  仅暴露两个端点:
    /log     -> 训练日志全文 (text/plain)
    /weights -> weights 目录的文件清单 (text/plain)
  HTML 与 JS 通过静态托管(本目录)加载。

用法:
  python scripts/training_dashboard_server.py [port]   # 默认 8080
  然后浏览器打开 http://localhost:8080/training_dashboard.html
  日志路径可用环境变量 VERICALL_TRAINING_LOG 覆盖。
"""
import os
import sys
import time
import socket
import http.server

ROOT = os.path.dirname(os.path.abspath(__file__))
WEBROOT = os.path.abspath(os.path.join(ROOT, ".."))          # 项目根(含 HTML)
# 训练日志路径：默认 D 盘，可用 VERICALL_TRAINING_LOG 环境变量覆盖（见 src/paths.py）
LOG = os.environ.get(
    "VERICALL_TRAINING_LOG",
    os.path.join(os.environ.get("VERICALL_DATA", "D:/VeriCall_data"),
                 "training_ep30.log"))
WEIGHTS = os.path.join(
    WEBROOT, "external", "aasist", "exp_result",
    "LA_AASIST_5060_ep24_bs16", "weights")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/log"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with open(LOG, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"no log yet")
            return
        if self.path.startswith("/weights"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                items = sorted(os.listdir(WEIGHTS), reverse=True)
                lines = []
                for it in items:
                    p = os.path.join(WEIGHTS, it)
                    if os.path.isfile(p):
                        ts = time.strftime("%m-%d %H:%M:%S",
                                           time.localtime(os.path.getmtime(p)))
                        lines.append(f"{it}\t{os.path.getsize(p)}\t{ts}")
                self.wfile.write("\n".join(lines).encode("utf-8"))
            except Exception as e:
                self.wfile.write(f"error: {e}".encode("utf-8"))
            return
        # 其余按静态文件(SimpleHTTPRequestHandler 服务 WEBROOT)
        return super().do_GET()

    def log_message(self, *a):
        pass  # 静默, 不刷屏


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    os.chdir(WEBROOT)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"看板服务已启动: http://localhost:{port}/training_dashboard.html")
    print(f"  (日志: {LOG})")
    print(f"  (权重: {WEIGHTS})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
