@echo off
rem Voice Filter uninstaller. Double-click this file, or run: uninstall.bat [-Yes] [-KeepModels] [-RemoveAppFolder]
rem All arguments are passed to uninstall.ps1.
setlocal
set "VF_PAUSE=0"
if "%~1"=="" set "VF_PAUSE=1"
rem Leave the app folder so that it can be deleted (-RemoveAppFolder).
cd /d "%SystemRoot%"
rem The block below is read completely before it runs, so cmd does not need to read
rem this file again after uninstall.ps1 has (possibly) scheduled the folder for deletion.
(
  "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*
  if "%VF_PAUSE%"=="1" pause
  exit /b
)
