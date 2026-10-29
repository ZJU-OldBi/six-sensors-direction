@echo off
setlocal
cd /d "%~dp0"
python train_theory_selected_model.py --data-dir "..\方向传感数据" --output-dir models
pause
