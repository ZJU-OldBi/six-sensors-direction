# 六传感器气体来源方向定位

本文件夹已经整合为两步使用：

1. 点击 `train_all.bat`，扫描上一级 `方向传感数据` 中的全部 CSV，重新训练模型并生成回溯、质量报告。
2. 点击 `run_direction.bat`，打开实时判断界面，在模型下拉框中选择模型，连接六传感器串口后进行一次检测。

Linux 对应运行 `train_all.sh` 和 `run_direction.sh`。

## 传感器方位

六个传感器组成内接正六边形，程序约定正前方为 0°，顺时针为正：

| 传感器 | 方位角 |
|---|---:|
| S1 | 0° |
| S2 | 60° |
| S3 | 120° |
| S4 | 180° |
| S5 | 240° |
| S6 | 300° |

因此，如果 S2 最先发生可靠上升，S2 的 60° 方位是 `first_response` 模型的基准角；随后响应的传感器会按配置比例继续修正这个角度。其它顺序分类模型则根据响应顺序输出训练得到的角度类别。实时判断允许窗口结束时只有部分传感器响应。

## 响应起点判定

程序不把普通波动当作响应，也不把向下变化当作响应：

1. 实时启动后先收集约 3 秒安静数据，每个传感器分别取中位数作为自己的基线，并用 MAD 估计自己的正常波动噪声。
2. 对每个传感器只计算 `当前值 - 自身基线` 的正向部分；传感器之间的绝对量程不同，不直接比较原始值。
3. 对正向变化做短窗口平滑。首个触发通道需要满足 `max(5×噪声, 基线绝对值的 10%)`、连续 3 点和明显正斜率，避免普通波动启动检测。
4. 首路触发后，S1–S6 仍逐路独立检测；远处传感器允许较小但持续的正向变化（至少超过自身噪声和约 1% 的归一化基线），这样不会因基线量程较大而漏掉 S2/S4/S6。每路均回溯到最早越过阈值并出现正斜率的采样点作为响应起点，同一帧并列时按编号排序。
5. **六个传感器 S1–S6 全部参与触发**：其中任意一路首次出现可靠的正向响应，就从该路的响应起点开始计时 5 秒；不是等待六路全部响应。
6. 这 5 秒内六路都使用各自独立的基线、噪声、正向变化和持续上升判据，并分别记录每一路的响应起点。界面始终显示六路完整顺序；窗口内未响应的通道会标记为“未响应”。
7. 如果 6 路在 5 秒内全部响应，则不必等满 5 秒，立即判断。未响应通道仅在模型内部作为“晚于已响应通道”的占位，界面不会把它显示成真实响应。

程序输出中同时保留 `response_order_all` 和 `response_rank_s1_to_s6`：前者始终包含 S1–S6，后者按 S1 到 S6 给出实际名次；没有在窗口内确认的通道记为 `null`，不会与真实响应混淆。

上述规则同时用于 CSV 训练和实时检测，训练和实时判断不会使用两套不同的起点定义。按 **Reset** 可清空本次检测并重新收集基线；每次检测只判断一次。

`first_response` 模型的默认顺序修正比例（第 2 至第 6 个响应）为 `0.50, 0.25, 0.15, 0.07, 0.03`。例如首个响应为 S5（240°），第二个响应为 S3（120°），第二路相对第一路的最短角度差为 -120°，则修正后角度为 `240 - 120×0.50 = 180°`。比例可以通过启动参数调整。

观察窗口默认 5 秒，可以在启动时临时调整：

```powershell
python direction_inference.py --port COM5 --observe-seconds 5 --primary-sensors 1,2,3,4,5,6 `
  --first-response-weights 0.50,0.25,0.15,0.07,0.03
```

任意一路 S1–S6 出现可靠响应后，5 秒后必定结束本次判断；只有在六路都没有达到响应条件时，程序才继续等待。可按 **Reset** 放弃本次检测。

## 一键重新训练（Windows）

先安装依赖：

```powershell
cd 方向定位模型
python -m pip install -r requirements.txt
```

然后双击：

```text
train_all.bat
```

或在终端执行：

```powershell
python train_direction_models.py --data-dir "..\方向传感数据" --output-dir models --method all
```

训练脚本把每个 CSV 当作一个样本，父目录名 `0度`、`45度`、…、`315度` 作为标签。它会生成四种可选择模型：

- `direction_model_order.joblib`：完整六路响应名次分类。
- `direction_model_sequence_8class.joblib`：严格输出 `0°、45°、90°、135°、180°、225°、270°、315°` 八类。
- `direction_model_first_response.joblib`：以最先响应传感器的方位为基准，再按后续响应顺序逐项修正方向；完整顺序仅作一致性验证。
- `direction_model_position_order.joblib`：把六边形位置和完整响应顺序结合，用于几何顺序比较。

训练还会生成：

- `models/direction_data_quality.csv`、`direction_data_quality_report.md`：每个 CSV 的响应质量、SNR、起点和顺序。
- `models/direction_metrics_*.json`：各模型分层交叉验证指标。
- `models/training_backtest_cv.csv`：`order` 模型逐个 CSV 的留出式回溯。
- `models/sequence_8class_backtest_cv.csv`：八分类模型逐个 CSV 的留出式回溯。

新增、删除或替换 CSV 后，重新双击 `train_all.bat` 即可覆盖模型。原始 CSV 不会被删除或移动。

## 实时判断（Windows）

双击：

```text
run_direction.bat
```

也可以在终端指定传感器串口：

```powershell
python direction_inference.py --port COM5 --baud 9600
```

界面顶部的模型下拉框列出 `models` 目录中的全部 `.joblib` 文件；点击 **Load selected model** 可切换模型，点击 **Browse…** 可选择其它目录中的模型。点击 **Connect sensor serial** 后读取采集程序每行前六个数值作为 S1–S6。

检测过程中的状态栏会依次提示：建立基线 → 等待 S1–S6 任一路触发 → 5 秒观察窗（带倒计时和当前顺序）。判断完成后，雷达图显示：

- 已响应传感器的排名 `#1` 起，未响应的显示 `-`；
- 本次判断使用的响应顺序，以及窗口内没有响应的传感器；
- 气体来源方向箭头；
- 对 `first_response` 模型显示基准传感器、每个顺序修正项和最终连续角度。

## Linux

```bash
cd 方向定位模型
python3 -m pip install -r requirements.txt
chmod +x train_all.sh run_direction.sh
./train_all.sh
./run_direction.sh --port /dev/ttyUSB0 --baud 9600
```

## 机器狗接口

默认只在终端打印 JSON。可以通过 UDP 或另一条串口发送方向结果：

```bash
python3 direction_inference.py \
  --port /dev/ttyUSB0 \
  --robot-output udp:192.168.1.20:9000
```

发送示例：

```json
{"type":"gas_direction","angle_deg":60.0,"confidence":0.86,"method":"first_response","timestamp":1710000000.0}
```

接入机器狗时仍应在接收端设置置信度阈值、通信超时急停、速度限制和人工急停。方向判断只负责输出方位，不直接替代机器狗的安全控制器。
