@echo off
setlocal
cd /d "%~dp0"
python train_direction_models.py --data-dir "..\方向传感数据" --output-dir models --method position_order
pause
