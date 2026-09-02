@echo off
chcp 65001 >nul
title 谛听 VeriCall 演示启动器
cd /d "%~dp0"

echo ============================================
echo   谛听 VeriCall · 一键演示启动器
echo ============================================

REM 1) 检查 python
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 python 3.10+ 或执行 conda activate vericall
    pause
    exit /b 1
)

REM 2) 缺失依赖则自动安装（清华镜像）
python -c "import fastapi, funasr" >nul 2>nul
if errorlevel 1 (
    echo [启动器] 检测到缺失依赖，正在安装（清华镜像）...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络或手动 pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM 3) 检查 AASIST 权重，缺失仅提示（声学通道以 stub 运行，不阻断演示）
python -c "import glob,sys; sys.exit(0 if glob.glob('external/aasist/exp_result/*/weights/best.pth') else 1)" >nul 2>nul
if errorlevel 1 (
    echo [启动器] 未找到 AASIST 权重，声学通道将以 stub 运行（不阻断演示）
)

REM 4) 探测 Ollama，不通则自动开启离线降级（联动 P0-4）
set VERICALL_OFFLINE=0
python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1.5)" >nul 2>nul
if errorlevel 1 (
    set VERICALL_OFFLINE=1
    echo [启动器] 未探测到 Ollama，自动开启离线降级模式（VERICALL_OFFLINE=1）
)

REM 5) 拉起服务并打开浏览器
echo [启动器] 启动谛听 VeriCall 演示服务（http://localhost:8000）...
start "" http://localhost:8000
python src/api/server.py
pause
