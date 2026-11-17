@echo off
setlocal
if not exist "%~dp0index.html" (
    echo The demo file index.html is missing. Download the full repo first.
    if not defined DEMO_NO_BROWSER pause
    exit /b 1
)
if not defined DEMO_NO_BROWSER start "" "%~dp0index.html"
echo Behavioral Anomaly Lab: "%~dp0index.html"
