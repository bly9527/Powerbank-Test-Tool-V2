@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "GPU_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%GPU_PYTHON%" (
    echo 请先创建本项目 .venv 并安装 requirements.txt 中的依赖。
    pause
    exit /b 1
)
if exist "%~dp0.gpu-runtime\onnxruntime\__init__.py" (
    echo GPU 运行库已存在。请通过“启动测试工具.bat”重开软件，在资源占用区勾选 GPU 加速。
    pause
    exit /b 0
)
"%GPU_PYTHON%" -m pip install --target "%~dp0.gpu-runtime" --no-deps --only-binary=:all: onnxruntime-directml==1.24.4
if errorlevel 1 (
    echo GPU 运行库安装失败，请检查网络和 Python 版本。
    pause
    exit /b 1
)
echo 安装完成。当前正在运行的软件不会中断；分析结束后通过原启动脚本重开，并勾选 GPU 加速。
pause
