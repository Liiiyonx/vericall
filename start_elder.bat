@echo off
chcp 65001 >nul
title 谛听 VeriCall · 适老一体机管家
cd /d "%~dp0"
echo ============================================
echo   谛听 VeriCall · 适老一体机管家
echo ============================================

REM 1) python 检查
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python，请先安装 python 3.10+ 或 conda activate vericall
  pause
  exit /b 1
)

REM 2) 缺失依赖自动安装（清华镜像，幂等）
python -c "import fastapi" >nul 2>nul
if errorlevel 1 (
  echo [管家] 检测到缺失依赖，正在安装（清华镜像）...
  python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

REM 3) 探测 Ollama，不通则自动开启离线降级（同 start.bat 兜底口径）
set VERICALL_OFFLINE=0
python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1.5)" >nul 2>nul
if errorlevel 1 set VERICALL_OFFLINE=1

REM 4) 后台拉起后端（最小化窗口）
start "vericall-api" /min python src/api/server.py
echo [管家] 后端启动中，等待端口就绪...
timeout /t 4 /nobreak >nul

REM 5) 全屏 kiosk 打开适老端（Edge 优先，回退 Chrome，再回退默认浏览器）
REM    --autoplay-policy 放行无手势语音播报；--kiosk 全屏常驻
set URL=http://localhost:8000/elder.html
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" (
  start "" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" --kiosk --autoplay-policy=no-user-gesture-required --edge-kiosk-type=fullscreen "%URL%"
  goto :eof
)
if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" (
  start "" "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" --kiosk --autoplay-policy=no-user-gesture-required --edge-kiosk-type=fullscreen "%URL%"
  goto :eof
)
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
  start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" --kiosk --autoplay-policy=no-user-gesture-required "%URL%"
  goto :eof
)
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" (
  start "" "%LocalAppData%\Google\Chrome\Application\chrome.exe" --kiosk --autoplay-policy=no-user-gesture-required "%URL%"
  goto :eof
)
start "" "%URL%"