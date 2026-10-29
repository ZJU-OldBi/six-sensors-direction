@echo off
setlocal
cd /d "%~dp0"
python sequence_eight_class_classifier.py train --data-dir "..\方向传感数据" --output-dir models
pause
