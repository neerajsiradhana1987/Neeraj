@echo off
setlocal EnableDelayedExpansion
REM ==========================================================================
REM  NSE Paper Agent - download and start with one click (Windows)
REM
REM  HOW TO USE
REM    1. Save this file anywhere (Desktop is fine) and double-click it.
REM    2. It downloads the agent, then starts the website and opens your browser.
REM    3. Ctrl+C in the black window stops the agent. Run the file again to
REM       restart it (it will not download twice).
REM
REM  REQUIREMENTS
REM    Windows 10/11 with Python 3.9+ from https://www.python.org/downloads
REM    During Python setup TICK "Add python.exe to PATH", then re-run this file.
REM
REM  The first download is about 60 KB; the app itself needs no other setup
REM  and works without an API key (it then uses a built-in demo strategy).
REM ==========================================================================

set "REPO=neerajsiradhana1987/Neeraj"
set "BRANCH=kilo/young-engine-v5j"
set "DIR=%USERPROFILE%\NSE-Paper-Agent"
set "PORT=8000"

echo.
echo   ============================================
echo    NSE Paper Agent  -  one click setup
echo   ============================================
echo    Install folder : %DIR%
echo    Source         : github.com/%REPO% (%BRANCH%)
echo.

REM ------------------------------------------------------------- 1. find Python
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [1/3] Python was not found on this PC.
  echo       Opening the Python download page - install it, tick
  echo       "Add python.exe to PATH", then run this file again.
  start "" "https://www.python.org/downloads/"
  echo.
  pause
  exit /b 1
)
echo [1/3] Python found.

REM ------------------------------------------------------------ 2. get the code
if exist "%DIR%\start-website.bat" goto :have_code

echo [2/3] Downloading the agent ^(about 60 KB^)...
if exist "%DIR%" rmdir /s /q "%DIR%" >nul 2>&1
md "%DIR%" >nul 2>&1

>  "%TEMP%\nse-install.ps1" echo [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
>> "%TEMP%\nse-install.ps1" echo $ErrorActionPreference = "Stop"
>> "%TEMP%\nse-install.ps1" echo $url = "https://codeload.github.com/%REPO%/zip/refs/heads/%BRANCH%"
>> "%TEMP%\nse-install.ps1" echo $zip = Join-Path $env:TEMP "nse-agent.zip"
>> "%TEMP%\nse-install.ps1" echo $dst = Join-Path $env:TEMP "nse-agent"
>> "%TEMP%\nse-install.ps1" echo try { Invoke-WebRequest -Uri $url -OutFile $zip } catch { Write-Host $_.Exception.Message; exit 1 }
>> "%TEMP%\nse-install.ps1" echo if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
>> "%TEMP%\nse-install.ps1" echo Expand-Archive -Path $zip -DestinationPath $dst -Force
>> "%TEMP%\nse-install.ps1" echo exit 0

powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\nse-install.ps1"
if errorlevel 1 (
  echo [2/3] Download failed - check your internet connection.
  echo       Manual download: https://github.com/%REPO%/archive/refs/heads/%BRANCH%.zip
  echo.
  pause
  exit /b 1
)

set "SRC="
for /d %%D in ("%TEMP%\nse-agent\*") do set "SRC=%%D"
if not defined SRC (
  echo [2/3] Unexpected download contents - nothing to install.
  echo.
  pause
  exit /b 1
)
xcopy /e /i /y /q "%SRC%\*" "%DIR%\" >nul
if not exist "%DIR%\start-website.bat" (
  echo [2/3] Install incomplete - please report this.
  echo.
  pause
  exit /b 1
)
echo [2/3] Installed to %DIR%.
goto :run

:have_code
echo [2/3] Already installed - starting it.
where git >nul 2>nul && git -C "%DIR%" pull --ff-only --quiet >nul 2>&1 && echo       Updated from GitHub.

:run
if not exist "%DIR%\start-website.bat" (
  echo start-website.bat is missing in %DIR%
  echo Delete that folder and run this file again for a fresh download.
  echo.
  pause
  exit /b 1
)

echo [3/3] Starting the website - your browser opens in a moment.
echo       Open the "phone link" printed below on your mobile to use it there.
echo.
cd /d "%DIR%"
set PORT=%PORT%
call start-website.bat
