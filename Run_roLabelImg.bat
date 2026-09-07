@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python with PyQt5 and lxml first.
    pause
    exit /b 1
)

python roLabelImg.py
if errorlevel 1 (
    echo.
    echo roLabelImg failed to start. Please check the error message above.
    pause
)
