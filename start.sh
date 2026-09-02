#!/usr/bin/env bash
# 谛听 VeriCall · 一键演示启动器（Linux / macOS）
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  谛听 VeriCall · 一键演示启动器"
echo "============================================"

# 1) 检查 python
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[错误] 未找到 python，请先安装 python 3.10+"
    exit 1
fi

# 2) 缺失依赖则自动安装（清华镜像）
if ! $PY -c "import fastapi, funasr" >/dev/null 2>&1; then
    echo "[启动器] 检测到缺失依赖，正在安装（清华镜像）..."
    $PY -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 3) 检查 AASIST 权重，缺失仅提示（声学通道以 stub 运行）
if ! $PY -c "import glob,sys; sys.exit(0 if glob.glob('external/aasist/exp_result/*/weights/best.pth') else 1)" >/dev/null 2>&1; then
    echo "[启动器] 未找到 AASIST 权重，声学通道将以 stub 运行（不阻断演示）"
fi

# 4) 探测 Ollama，不通则自动开启离线降级（联动 P0-4）
export VERICALL_OFFLINE=0
if ! $PY -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1.5)" >/dev/null 2>&1; then
    export VERICALL_OFFLINE=1
    echo "[启动器] 未探测到 Ollama，自动开启离线降级模式（VERICALL_OFFLINE=1）"
fi

# 5) 拉起服务并打开浏览器
echo "[启动器] 启动谛听 VeriCall 演示服务（http://localhost:8000）..."
( sleep 2; command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8000 || command -v open >/dev/null 2>&1 && open http://localhost:8000 ) >/dev/null 2>&1 &
$PY src/api/server.py
