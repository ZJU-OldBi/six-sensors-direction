@echo off
setlocal
cd /d "%~dp0"
python analyze_direction_data.py --data-dir "..\方向传感数据" --output models\direction_data_quality.csv
python train_direction_models.py --data-dir "..\方向传感数据" --output-dir models --method all
python backtest_training_data.py --data-dir "..\方向传感数据" --model models\direction_model_order.joblib --mode cv --output models\training_backtest_cv.csv
python backtest_training_data.py --data-dir "..\方向传感数据" --model models\direction_model_sequence_8class.joblib --method sequence_8class --mode cv --output models\sequence_8class_backtest_cv.csv
pause
