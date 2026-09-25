@echo off
chcp 65001 >nul
title 谛听 VeriCall 演示启动器
cd /d "%~dp0"

echo ============================================
echo   谛听 VeriCall · 一键演示启动器
echo ============================================

REM 1) 选择 python：可显式指定 VERICALL_PYTHON 指向已装好 torch/torchaudio 的环境
set "VERICALL_PYBIN=python"
if defined VERICALL_PYTHON set "VERICALL_PYBIN=%VERICALL_PYTHON%"
call "%VERICALL_PYBIN%" --version >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到可用 Python: %VERICALL_PYBIN%
    echo         请安装 Python 3.10+，或设置 VERICALL_PYTHON 指向 conda/venv 的 python.exe
    pause
    exit /b 1
)

REM 2) 缺失依赖则自动安装（清华镜像）
call "%VERICALL_PYBIN%" -c "import fastapi, funasr" >nul 2>nul
if errorlevel 1 (
    echo [启动器] 检测到缺失依赖，正在安装（清华镜像）...
    call "%VERICALL_PYBIN%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络或手动 pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM 2.1) torchaudio 在 Windows 上可能出现 DLL 可导入失败；提前拦截并给出可执行指引
call "%VERICALL_PYBIN%" -c "import torch, torchaudio" >nul 2>nul
if errorlevel 1 (
    echo [错误] 当前 Python 无法加载 torch 或 torchaudio（常见原因：Python/torchaudio DLL 不匹配）
    echo         请改用已验证环境，例如:
    echo         set VERICALL_PYTHON=D:\path\to\env\python.exe
    echo         start.bat
    echo         可用 conda run -n vericall python 验证环境后再设置 VERICALL_PYTHON。
    pause
    exit /b 1
)

REM 3) 启动预检：只报告状态，不因 Ollama 不通而强制离线
if not defined VERICALL_ACOUSTIC set "VERICALL_ACOUSTIC=xlsr_cn:wide"
call "%VERICALL_PYBIN%" scripts\preflight.py
if errorlevel 1 echo [启动器] 预检异常，继续启动服务并请检查上方输出

REM 4) 拉起服务并打开浏览器
echo [启动器] 启动谛听 VeriCall 演示服务（http://localhost:8000）...
start "" http://localhost:8000
call "%VERICALL_PYBIN%" src/api/server.py
pause
