@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
    echo Could not open the application directory: "%~dp0"
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python with PyQt5 and lxml first.
    pause
    exit /b 1
)

echo Python candidates on PATH:
where python
echo.
python -u launch_roLabelImg.py %*
set "launch_exit_code=%errorlevel%"
if not "%launch_exit_code%"=="0" (
    echo.
    echo roLabelImg exited with code %launch_exit_code%.
    echo Check the diagnostic output and log file path above.
    pause
)
exit /b %launch_exit_code%
