@echo off
cd /d "%~dp0"

echo ========================================================
echo   充电宝数显均匀度自动测试工具 - 依赖环境安装
echo ========================================================
echo.

set REQ_FILE=requirements.txt
if not exist "%REQ_FILE%" (
    set REQ_FILE=..\requirements.txt
)

echo 正在从 %REQ_FILE% 安装项目所需依赖组件...
pip install -r "%REQ_FILE%" -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

if %errorlevel% neq 0 (
    echo.
    echo 阿里云镜像连接失败，正在重试清华大学镜像源...
    pip install -r "%REQ_FILE%" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
)

echo.
echo ========================================================
echo   依赖安装完成！您可以返回主目录双击运行 [启动测试工具.bat]
echo ========================================================
pause
