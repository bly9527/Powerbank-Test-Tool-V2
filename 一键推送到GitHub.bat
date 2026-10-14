@echo off
cd /d "%~dp0"
echo ========================================================
echo   Pushing code to https://github.com/bly9527/Powerbank-Test-Tool-V2.git ...
echo ========================================================
echo.

git remote get-url origin

echo Starting git push...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ========================================================
    echo   [SUCCESS] Code successfully pushed to GitHub!
    echo ========================================================
) else (
    echo.
    echo [ERROR] Push failed.
    echo Please check if your VPN is enabled and complete GitHub login.
)
pause
