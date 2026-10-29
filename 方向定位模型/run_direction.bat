@echo off
setlocal
cd /d "%~dp0"
python direction_inference.py
if errorlevel 1 pause
