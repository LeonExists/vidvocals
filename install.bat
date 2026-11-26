@echo off
rem Voice Filter installer. Double-click this file, or run: install.bat [-Yes] [-Presets fast] [-Port 8765] ...
rem All arguments are passed to install.ps1 (see the top of install.ps1 for the full list).
setlocal
set "VF_PAUSE=0"
if "%~1"=="" set "VF_PAUSE=1"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "VF_RC=%ERRORLEVEL%"
if "%VF_PAUSE%"=="1" pause
exit /b %VF_RC%
