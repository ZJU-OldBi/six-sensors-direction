# 2026-10-16 四方向气体来源定位

本文件夹是今天数据的独立训练工程。原始数据放在 `方向传感数据_20261016` 中。训练标签只使用 `0度`、`90度`、`180度`、`270度` 四个目录；今天目录中若存在 `45度` 数据，会被保留但不会参与本次训练。

方向判断只看响应先后：1、3、5 号是主通道，要求三路都检测到正向响应后才输出方向；2、4、6 号作为低权重参考顺序，缺失时不阻止判断。主通道顺序决定主要结果，参考通道只在有可靠响应时提供辅助信息。模型不使用峰值、T50、响应幅值、面积或任意传感器之间的时间间隔。

每个 CSV 的每一路单独估计前段基线。基线噪声使用 MAD 估计，信号经过短滑动平均后，第一次连续 3 个采样点超过 `3×MAD` 且当前点仍在上升的位置就是响应开始时间。向下偏离不会触发响应。实时模式默认先采集 60 帧基线；只有某一路同时超过 `5×MAD` 和基线绝对值的 10%，并连续保持 5 帧，才确认气体并开始判断。

## Windows 训练

```powershell
cd 训练20261016
python -m pip install -r ..\方向定位模型\requirements.txt
.\train_20261016.bat
```

也可以直接运行：

```powershell
python direction_order_20261016.py train
```

训练输出位于 `models`：

```text
direction_order_model_20261016.joblib   训练模型
quality_20261016.csv                    每个 CSV 的响应质量和顺序
backtest_20261016.csv                   留一回溯结果
metrics_20261016.json                   准确率和混淆矩阵
```

## 判断一个 CSV

```powershell
python direction_order_20261016.py predict --csv "..\方向传感数据_20261016\0度\sensor_data_xxx.csv"
```

程序会输出 `gas_detected`、`primary_order`、`reference_order`、`angle_deg` 和 `confidence`。没有超过基线波动阈值时只输出未检测到气体；即使检测到气体但 1、3、5 号没有全部可靠响应，也不会输出方向。

## 串口实时判断

采集程序每行前六个字段应为六个传感器数值，后面的报警字段可以存在：

```powershell
python direction_order_20261016.py live --model models\direction_order_model_20261016.joblib --port COM5 --baud 9600
```

Linux：

```bash
chmod +x train_20261016.sh
./train_20261016.sh
python3 direction_order_20261016.py live --model models/direction_order_model_20261016.joblib --port /dev/ttyUSB0 --baud 9600
```

实时流程是：采集 60 帧基线，等待明显超过基线的持续正向响应，触发后再采集 60 帧确认，使用 1、3、5 的响应顺序判断一次方向，然后清空窗口重新等待下一次气体。输出为 JSON，便于后续通过 UDP、串口或机器狗 SDK 转发。

## 实时雷达可视化

双击 `start_radar_20261016.bat` 可以直接打开雷达界面。没有配置串口时会自动使用内置模拟数据预览界面；模拟数据完成一次检测后会显示来源箭头和角度。实际接入传感器前，编辑这个 bat 文件，把：

```bat
set PORT=
```

改成例如：

```bat
set PORT=COM5
```

然后再次双击即可读取 COM5。也可以在命令行启动：

```powershell
python realtime_direction_radar_20261016.py --port COM5 --baud 9600
```

Linux 使用：

```bash
chmod +x start_radar_20261016.sh
./start_radar_20261016.sh
python3 realtime_direction_radar_20261016.py --port /dev/ttyUSB0 --baud 9600
```

界面中红色点是 1、3、5 主通道，灰色点是 2、4、6 参考通道，橙色箭头是气体来源方向。未确认到明显超过基线波动阈值的持续向上响应时，角度保持 `--`；捕获结束后只判断一次，点击“复位并重新检测”才能开始下一次。需要更长等待时间时可使用 `--baseline-frames 100 --capture-frames 100`。
