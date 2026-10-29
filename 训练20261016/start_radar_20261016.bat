@echo off
setlocal
cd /d "%~dp0"
set PORT=
if "%PORT%"=="" (
    echo 未设置串口，启动模拟数据预览。实际使用请编辑本文件，将 PORT 改为 COM5 等串口。
    python realtime_direction_radar_20261016.py --simulate
) else (
    python realtime_direction_radar_20261016.py --port %PORT% --baud 9600
)
pause
