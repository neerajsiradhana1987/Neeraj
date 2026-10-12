@echo off
REM One-click launcher for the NSE paper trading dashboard (Windows).
REM Double-click this file. Press Ctrl+C in the window to stop.
cd /d "%~dp0"
setlocal
if "%PORT%"=="" set PORT=8000
where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py -3)
%PY% -m trading_agent web --host 0.0.0.0 --port %PORT% --auto-token --open --autostart --poll 60
echo.
echo Dashboard stopped. Press any key to close this window.
pause >nul
