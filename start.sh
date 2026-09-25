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

# 3) 启动预检：只报告状态，不因 Ollama 不通而强制离线
export VERICALL_ACOUSTIC="${VERICALL_ACOUSTIC:-xlsr_cn:wide}"
$PY scripts/preflight.py || echo "[启动器] 预检异常，继续启动服务并请检查上方输出"

# 4) 拉起服务并打开浏览器
echo "[启动器] 启动谛听 VeriCall 演示服务（http://localhost:8000）..."
( sleep 2; command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8000 || command -v open >/dev/null 2>&1 && open http://localhost:8000 ) >/dev/null 2>&1 &
$PY src/api/server.py
