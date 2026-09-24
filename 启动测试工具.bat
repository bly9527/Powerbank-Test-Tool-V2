@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   充电宝数显均匀度自动测试工具 (Auto Data Extractor) v2.0
echo ========================================================
echo.
echo 正在检测运行环境并启动测试工具图形界面，请稍候...

set PY_EXE=

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_EXE=python"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    )
)

if "%PY_EXE%"=="" (
    echo.
    echo [错误] 未检测到 Python 环境！
    echo 请先安装 Python 3.8 及以上版本，或运行 脚本包\一键安装依赖.bat。
    echo.
    pause
    exit /b 1
)

echo 当前使用 Python 解释器: %PY_EXE%
"%PY_EXE%" main.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================================
    echo [提示] 软件异常退出，退出代码：%errorlevel%
    if exist crash_log.txt (
        echo ----------------- 崩溃日志摘要 -----------------
        type crash_log.txt
        echo ------------------------------------------------
    )
    echo ========================================================
    pause
)
