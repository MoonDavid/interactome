@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0pipelines\start_norma.ps1" -OpenBrowser
if errorlevel 1 pause
