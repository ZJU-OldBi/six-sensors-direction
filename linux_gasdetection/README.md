# Linux 气体来源方向识别系统（linux_gasdetection）

本文件夹是 Windows 端「方向定位模型」的 Linux 移植版，实现相同的功能：

- **实时方向判断界面**：读取六路气体传感器串口数据，自动建立基线，任一传感器出现可靠正向上升后进入 5 秒观察窗，输出气体来源方向角、置信度和响应顺序，并用雷达图显示。
- **无界面模式**：在没有显示器或通过 SSH 登录（机器狗、树莓派、Jetson）时，用命令行完成同样的判断，并输出 JSON 给机器狗控制器。
- **CSV 回放与自检**：没有硬件也能验证安装是否正确。
- **训练脚本**：在 Linux 上用采集好的 CSV 重新训练模型（同一套特征与检测规则）。
- **结果对接**：JSON 可以打印到终端、通过 UDP 或另一条串口发送。

检测规则与 Windows 版本完全一致（同一模型、同一份数据，两端输出的角度和置信度相同，已逐项核对）。

---

## 1. 与 Windows 版本文件对照

| Windows（方向定位模型） | Linux（本文件夹） | 说明 |
|---|---|---|
| `direction_inference.py`（Tk 界面） | `direction_inference.py` + `direction_core.py` | 界面功能一致；检测规则抽到 `direction_core.py`，GUI 与 CLI 共用 |
| — | `direction_cli.py` | 新增：无界面命令行模式（机器狗部署、SSH、systemd 服务） |
| `direction_features.py` | `direction_features.py` | 特征提取完全一致（基线、起点、响应顺序） |
| `train_direction_models.py` | `train_direction_models.py` | 训练四种模型 |
| `analyze_direction_data.py` | `analyze_direction_data.py` | 数据质量报告 |
| `backtest_training_data.py` | `backtest_training_data.py` | 留出式回溯测试 |
| `train_all.bat` / `run_direction.bat` | `train_all.sh` / `run_gui.sh` / `run_cli.sh` | 一键脚本 |
| `models\*.joblib`（Windows 训练） | `models/*.joblib` | 已预置 Windows 训练好的模型，可直接使用；也可在 Linux 上重新训练覆盖 |

---

## 2. 环境要求

适用于 Ubuntu / Debian 系发行版（树莓派、Jetson、普通 PC 均可）。

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-tk
```

- `python3` / `python3-venv`：运行与隔离环境（必需）。
- `python3-tk`：只有图形界面 `run_gui.sh` 需要；纯命令行模式不需要。

**串口权限**（一次性设置）：把当前用户加入 `dialout` 组，然后注销重新登录（或重启）：

```bash
sudo usermod -aG dialout $USER
```

接入传感器后确认设备文件：

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

---

## 3. 快速开始

```bash
# 1) 把整个 linux_gasdetection 文件夹拷贝到 Linux 机器任意目录
# 2) 赋予脚本执行权限（Windows 拷过来通常没有执行位）
cd linux_gasdetection
chmod +x *.sh

# 3) 一次性准备环境（自动创建 .venv 并安装依赖，需要联网一次）
./setup.sh

# 4) 选择一种方式运行
./run_gui.sh --port /dev/ttyUSB0      # 图形界面
./run_cli.sh --port /dev/ttyUSB0      # 无界面实时检测
```

没有硬件时可以先做离线验证（samples 文件夹带有两条测试数据）：

```bash
./run_cli.sh --replay "samples/0度/sample_0deg.csv" --speed 0
./run_cli.sh --self-test --data-dir samples
```

---

## 4. 脚本说明

| 脚本 | 作用 | 常用示例 |
|---|---|---|
| `setup.sh` | 创建 `.venv` 虚拟环境并安装 `requirements.txt`（numpy、scikit-learn、joblib、matplotlib、pyserial）；检查 python3-tk 并给出提示 | `./setup.sh` |
| `run_gui.sh` | 启动图形界面（`direction_inference.py`），参数原样转发 | `./run_gui.sh --port /dev/ttyUSB0` |
| `run_cli.sh` | 启动无界面模式（`direction_cli.py`），参数原样转发 | `./run_cli.sh --list-ports` |
| `train_all.sh` | 一键重新训练：质量报告 → 四种模型 → 两次回溯测试（等价 Windows 的 `train_all.bat`） | `./train_all.sh --data-dir /path/to/方向传感数据` |

`run_gui.sh` / `run_cli.sh` / `train_all.sh` 会在首次运行时自动调用 `setup.sh`，不需要手工先跑 setup。

---

## 5. 图形界面使用（run_gui.sh）

界面与 Windows 版本操作方式一致：

1. **选择模型**：顶部下拉框选择 `models/` 中的模型，点 **Load selected model**；模型不在本目录时用 **Browse...**。
   默认模型为 `direction_model_first_response.joblib`（不存在时自动改用 `direction_model_order.joblib`）。
2. **连接传感器**：点 **Refresh ports** 刷新串口列表，选择 `/dev/ttyUSB0` 等设备，点 **Connect sensor serial**。
3. **检测**：程序先收集约 3 秒安静基线；放气后任意一路传感器可靠上升即开始计时，状态栏显示 5 秒观察窗倒计时和当前响应顺序；窗口结束（或六路提前全部响应）后锁定结果。
4. **查看结果**：结果区显示估计方向角、置信度、完整六路响应顺序（未响应的显示 `(no response)`）；雷达图上每个传感器显示名次 `#1` 起，未响应显示 `-`，红色箭头为估计来源方向；`first_response` 模型还会列出每个顺序修正项。
5. **下一次检测**：按 **Reset** 清空并重新采集基线。每次只判断一次。
6. **无硬件测试**：点 **Replay CSV...** 选择一条记录好的 CSV，Speed 可选 1x～50x 或 Instant。

常用启动参数（与 Windows 相同，路径改为 Linux 风格）：

```bash
./run_gui.sh --port /dev/ttyUSB0 --baud 9600 --observe-seconds 5
./run_gui.sh --replay "samples/180度/sample_180deg.csv"
./run_gui.sh --robot-output udp:192.168.1.20:9000        # 检测结果同时发给机器狗
```

> 界面文字为英文，这是为了避开部分 Linux 系统缺中文字体导致 matplotlib 显示方块的问题。

---

## 6. 无界面模式使用（run_cli.sh）

五种模式：

```bash
./run_cli.sh --list-ports                          # 列出串口设备后退出
./run_cli.sh --port /dev/ttyUSB0                   # 连续实时检测（每次判断后自动重新武装）
./run_cli.sh --port /dev/ttyUSB0 --once            # 只判断一次，输出结果后退出
./run_cli.sh --replay FILE [--speed N]             # 回放 CSV 并输出判断结果（1.0=实时，0=最快）
./run_cli.sh --self-test --data-dir DIR            # 逐个分类数据目录中的 CSV，统计准确率
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model PATH` | `models/direction_model_first_response.joblib` | 使用的模型文件 |
| `--baud` | `9600` | 串口波特率 |
| `--observe-seconds` | `5` | 触发后的观察窗长度（秒） |
| `--primary-sensors` | `1,2,3,4,5,6` | 触发传感器集合，任意一路响应即开始窗口 |
| `--first-response-weights` | `0.50,0.25,0.15,0.07,0.03` | first_response 的第 2～6 个响应修正权重 |
| `--once` | 关 | 判断一次后退出（默认持续检测） |
| `--cooldown` | `3.0` | 两次检测之间的等待秒数，随后重新采集基线 |
| `--robot-output` | 无（仅打印） | `udp:HOST:PORT` 或 `serial:DEVICE` |
| `--speed` | `1.0` | 回放速度（0 表示最快） |

**输出约定**：判断结果是一行 JSON 输出到 **stdout**（机器狗程序直接读这一行即可）；状态提示（基线、触发、倒计时）输出到 **stderr**，不影响解析。

```json
{"type": "gas_direction", "angle_deg": 59.4, "confidence": 0.19465166031456602, "method": "first_response"}
```

运行示例（机器狗连续检测 + UDP 上报）：

```bash
./run_cli.sh --port /dev/ttyUSB0 --robot-output udp:192.168.1.20:9000
```

---

## 7. 在 Linux 上重新训练模型

`models/` 里已经放好了 Windows 训练好的模型，通常不需要重新训练。以下情况建议在 Linux 上重训：

- 加载模型时报 scikit-learn 版本警告或错误（Linux 环境版本与训练环境不同）；
- 在 Linux 上采集了新的实验数据；
- 想更换默认模型。

数据目录结构与 Windows 相同：每条 CSV 实验放在以角度命名的子文件夹里（文件夹名必须含“度”，如 `0度`、`45度`……`315度`）：

```text
方向传感数据/
├─ 0度/    sensor_data_2026-10-15_11-50-28.csv
├─ 45度/   ...
├─ 90度/   ...
└─ 315度/  ...
```

CSV 需要包含 `sensor1`～`sensor6` 六列，`timestamp` 列可选（大小写不敏感，多余列自动忽略），格式与采集程序输出一致：

```text
timestamp,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,alarm_status,alarm_channel
2026-10-15T11:49:55.145500,189.0,2327.0,125.0,653.0,2503.0,949.0,0,0
```

训练命令（默认读取上一级 `../方向传感数据`，与 Windows 布局一致）：

```bash
./train_all.sh
# 或指定数据目录
./train_all.sh --data-dir /home/robot/data/方向传感数据
# 或指定输出目录
./train_all.sh --data-dir /home/robot/data/方向传感数据 --output-dir models
```

训练完成后 `models/` 会更新以下内容：

- `direction_model_order.joblib`、`direction_model_sequence_8class.joblib`、`direction_model_first_response.joblib`、`direction_model_position_order.joblib`：四种模型；
- `direction_metrics_*.json`：交叉验证指标；
- `direction_data_quality.csv / .json / _report.md`：数据质量报告（good / warning / bad）；
- `training_backtest_cv.csv`、`sequence_8class_backtest_cv.csv`：逐条回溯结果。

训练不修改、不删除原始 CSV。单独运行某个分析脚本也可以（都在本文件夹内直接 `python3` 调用，参数与 Windows 相同）。

---

## 8. 与机器狗对接

判断完成时，程序通过 `--robot-output` 指定的通道发送一行 JSON：

```json
{"type":"gas_direction","angle_deg":60.0,"confidence":0.86,"method":"first_response","timestamp":1710000000.0}
```

- UDP：`--robot-output udp:192.168.1.20:9000`
- 串口：`--robot-output serial:/dev/ttyUSB1 --robot-baud 115200`
- 不指定时只打印到终端（stdout）。

**安全要求**：接收端必须设置置信度阈值、通信超时急停、速度限制和人工急停。方向判断只负责输出方位，不能替代机器狗的安全控制器。

作为开机自启服务（systemd 示例，按实际路径修改）：

```ini
[Unit]
Description=Gas direction detection
After=network.target

[Service]
Type=simple
User=robot
WorkingDirectory=/home/robot/linux_gasdetection
ExecStart=/home/robot/linux_gasdetection/run_cli.sh --port /dev/ttyUSB0 --robot-output udp:192.168.1.20:9000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

保存为 `/etc/systemd/system/gas-direction.service` 后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gas-direction.service
journalctl -u gas-direction.service -f        # 查看状态与检测结果
```

---

## 9. 文件夹结构

```text
linux_gasdetection/
├─ setup.sh                     # 环境准备（venv + 依赖）
├─ run_gui.sh                   # 启动图形界面
├─ run_cli.sh                   # 启动无界面模式
├─ train_all.sh                 # 一键重新训练
├─ direction_inference.py       # 图形界面（Tk）
├─ direction_cli.py             # 无界面模式（实时/单次/回放/自检）
├─ direction_core.py            # 共享检测核心（基线、触发、观察窗、结果输出）
├─ direction_features.py        # 特征提取（与 Windows 完全一致）
├─ train_direction_models.py    # 训练脚本
├─ analyze_direction_data.py    # 数据质量报告
├─ backtest_training_data.py    # 回溯测试
├─ requirements.txt             # Python 依赖
├─ models/                      # 模型（已预置 Windows 训练结果）
├─ samples/                     # 两条测试用 CSV（0度 / 180度）
└─ README.md                    # 本文档
```

---

## 10. 常见问题

| 现象 | 原因与解决 |
|---|---|
| `Cannot open a display` | 无图形环境（SSH/串口终端）。改用 `./run_cli.sh --port /dev/ttyUSB0` |
| 打不开串口 / `Permission denied` | 未加入 dialout 组：`sudo usermod -aG dialout $USER` 后重新登录；或用 `sudo` 临时运行 |
| 找不到 `/dev/ttyUSB0` | 用 `./run_cli.sh --list-ports` 或 `ls /dev/ttyUSB* /dev/ttyACM*` 确认设备名，检查 USB 线和供电 |
| `python3-tk` 缺失（GUI 弹窗报错） | `sudo apt install -y python3-tk` 后重新 `./run_gui.sh` |
| 无法联网安装依赖 | 在能联网的机器上 `pip download -r requirements.txt -d wheels`，拷到 Linux 后 `pip install --no-index --find-links wheels -r requirements.txt` |
| 加载模型出现 sklearn 版本警告 | 一般仍可用；若报错，在本机执行 `./train_all.sh` 重新训练即可 |
| 一直不触发 | 检查串口数据格式（每行前 6 个数值）；只有连续 3 点以上的正向上升才会计为响应，噪声/下降不会触发 |
| 结果不稳定 | 看置信度与响应顺序；置信度低或顺序并列时应复测。可切到 `first_response` 模型对照比较 |
| 判断完想继续测 | GUI：按 Reset；CLI：默认自动重新武装（`--cooldown` 秒后重新采集基线） |

---

## 附：检测规则速览（与 Windows 相同）

1. 启动后先收集约 3 秒安静数据，每路取中位数作为自身基线，用 MAD 估计噪声。
2. 只统计高于基线的正向变化；首个触发通道须超过 `max(5×噪声, 基线绝对值的 10%)` 且连续 3 点上升、斜率明显。
3. 任意一路 S1～S6 首次可靠上升即开始 5 秒观察窗；窗口内六路各自独立判定，回溯到最早的上升起点排序。
4. 六路在窗口内全部响应则提前结束；未响应的通道记为“未响应”，不参与显示但不会阻止判断。
5. 结果锁定，等待 Reset（GUI）或自动重新武装（CLI）。
