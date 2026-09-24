@echo off
cd /d "%~dp0"
echo ========================================================
echo   Pushing code to https://github.com/bly9527/1.git ...
echo ========================================================
echo.

git remote remove origin 2>nul
git remote add origin https://github.com/bly9527/1.git
git branch -M main

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
