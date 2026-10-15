# -*- coding: utf-8 -*-
"""
五路气体传感阵列方向识别。

串口数据格式沿用 sensor_monitor.py / 单片机程序：
    VOC1,VOC2,VOC3,VOC4,VOC5,alarm,alarm_channel

训练数据目录默认使用“方向数据”，其中 0.csv、45.csv ... 315.csv
会被读取为 8 个方向模板。实时检测时脚本先维护空闲基线，检测到
气体响应后再显示五路相对基线波形、方向雷达图和估计角度。
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover - only matters on machines without pyserial
    serial = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SENSOR_COUNT = 5
CSV_ENCODINGS = ("utf-8-sig", "gb18030", "gbk", "utf-16", "latin1")
DEFAULT_SAMPLE_PERIOD = 0.1
TRAINING_BASLINE_SAMPLES = 10


@dataclass
class SensorFrame:
    values: np.ndarray
    timestamp: datetime
    alarm: int = 0
    alarm_channel: int = 0


@dataclass
class EventFeatures:
    vector: np.ndarray
    baseline: np.ndarray
    peak_delta: np.ndarray
    onset_indices: list[Optional[int]]
    order: list[int]
    global_onset: int
    sample_period: float


@dataclass
class DirectionTemplate:
    angle: float
    path: Path
    features: EventFeatures


@dataclass
class DirectionResult:
    angle: float
    nearest_angle: float
    confidence: float
    probabilities: dict[float, float]
    distances: dict[float, float]
    order: list[int]


def read_text_with_fallback(path: Path) -> str:
    data = path.read_bytes()
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError("unknown", data, 0, 1, f"无法解码 {path}: {last_error}")


def read_sensor_csv(path: Path) -> tuple[np.ndarray, list[datetime]]:
    text = read_text_with_fallback(path)
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise ValueError(f"{path} 是空文件")

    header = [cell.strip() for cell in rows[0]]
    data_rows = rows[1:] if "传感器1" in header else rows
    values: list[list[float]] = []
    timestamps: list[datetime] = []

    for row in data_rows:
        if len(row) < 6:
            continue
        try:
            values.append([float(row[i]) for i in range(1, 6)])
            timestamps.append(datetime.fromisoformat(row[0].strip()))
        except (ValueError, IndexError):
            continue

    if not values:
        raise ValueError(f"{path} 中没有可用的五路传感器数据")
    return np.asarray(values, dtype=float), timestamps


def parse_serial_line(line: str) -> Optional[tuple[np.ndarray, int, int]]:
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        numbers = [float(p) for p in parts[:5]]
        alarm = int(float(parts[5])) if len(parts) >= 6 else 0
        alarm_channel = int(float(parts[6])) if len(parts) >= 7 else 0
    except ValueError:
        return None
    return np.asarray(numbers, dtype=float), alarm, alarm_channel


def estimate_sample_period(timestamps: list[datetime]) -> float:
    if len(timestamps) < 3:
        return DEFAULT_SAMPLE_PERIOD
    deltas = np.asarray(
        [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))],
        dtype=float,
    )
    deltas = deltas[(deltas > 0.01) & (deltas < 2.0)]
    if len(deltas) == 0:
        return DEFAULT_SAMPLE_PERIOD
    return float(np.median(deltas))


def moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.empty_like(values, dtype=float)
    for sensor_index in range(values.shape[1]):
        smoothed[:, sensor_index] = np.convolve(padded[:, sensor_index], kernel, mode="valid")
    return smoothed


def first_crossing(series: np.ndarray, threshold: float, start: int, hold: int = 2) -> Optional[int]:
    if threshold <= 0:
        return None
    end = len(series) - hold + 1
    for idx in range(start, max(start, end)):
        if np.all(series[idx : idx + hold] >= threshold):
            return idx
    return None


def extract_event_features(
    samples: np.ndarray,
    timestamps: Optional[list[datetime]] = None,
    baseline_samples: int = TRAINING_BASLINE_SAMPLES,
    min_peak_delta: float = 35.0,
) -> EventFeatures:
    if samples.ndim != 2 or samples.shape[1] != SENSOR_COUNT:
        raise ValueError("samples 必须是 N x 5 的数组")
    if len(samples) < 8:
        raise ValueError("样本太短，无法提取方向特征")

    baseline_count = int(max(3, min(baseline_samples, len(samples) // 3)))
    baseline = np.median(samples[:baseline_count], axis=0)
    sample_period = estimate_sample_period(timestamps or [])
    smooth = moving_average(samples, window=3)
    positive_delta = np.maximum(smooth - baseline, 0.0)
    peak_delta = np.max(positive_delta, axis=0)

    t20: list[Optional[int]] = []
    t40: list[Optional[int]] = []
    for sensor_index in range(SENSOR_COUNT):
        peak = peak_delta[sensor_index]
        if peak < min_peak_delta:
            t20.append(None)
            t40.append(None)
            continue
        t20.append(
            first_crossing(
                positive_delta[:, sensor_index],
                max(min_peak_delta, peak * 0.20),
                baseline_count,
                hold=2,
            )
        )
        t40.append(
            first_crossing(
                positive_delta[:, sensor_index],
                max(min_peak_delta, peak * 0.40),
                baseline_count,
                hold=2,
            )
        )

    finite_t20 = [idx for idx in t20 if idx is not None]
    global_onset = min(finite_t20) if finite_t20 else baseline_count
    max_lag_seconds = 6.0
    missing_time = max_lag_seconds + 1.0

    def relative_time(index: Optional[int]) -> float:
        if index is None:
            return missing_time
        return min(max((index - global_onset) * sample_period, 0.0), missing_time)

    t20_seconds = np.asarray([relative_time(idx) for idx in t20], dtype=float) / missing_time
    t40_seconds = np.asarray([relative_time(idx) for idx in t40], dtype=float) / missing_time

    sort_key = [(relative_time(idx), sensor_id + 1) for sensor_id, idx in enumerate(t20)]
    order = [sensor_id for _, sensor_id in sorted(sort_key)]
    ranks = np.empty(SENSOR_COUNT, dtype=float)
    for rank, sensor_id in enumerate(order):
        ranks[sensor_id - 1] = rank / max(SENSOR_COUNT - 1, 1)

    early_profiles: list[np.ndarray] = []
    for offset_seconds in (0.4, 0.8, 1.5, 2.5, 4.0):
        idx = min(len(samples) - 1, global_onset + int(round(offset_seconds / sample_period)))
        profile = positive_delta[idx]
        total = float(np.sum(profile))
        early_profiles.append(profile / total if total > 1e-9 else np.zeros(SENSOR_COUNT))

    peak_total = float(np.sum(peak_delta))
    peak_profile = peak_delta / peak_total if peak_total > 1e-9 else np.zeros(SENSOR_COUNT)

    vector = np.concatenate(
        [
            t20_seconds * 2.4,
            t40_seconds * 1.4,
            ranks * 0.7,
            np.concatenate(early_profiles) * 1.1,
            peak_profile * 1.2,
        ]
    )

    return EventFeatures(
        vector=vector.astype(float),
        baseline=baseline.astype(float),
        peak_delta=peak_delta.astype(float),
        onset_indices=t20,
        order=order,
        global_onset=global_onset,
        sample_period=sample_period,
    )


class DirectionModel:
    def __init__(self, train_dir: Path):
        self.train_dir = train_dir
        self.templates = self._load_templates(train_dir)

    @staticmethod
    def _load_templates(train_dir: Path) -> list[DirectionTemplate]:
        if not train_dir.exists():
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")
        templates: list[DirectionTemplate] = []
        for path in sorted(train_dir.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else 9999):
            if not path.stem.isdigit():
                continue
            angle = float(path.stem) % 360.0
            samples, timestamps = read_sensor_csv(path)
            features = extract_event_features(samples, timestamps, baseline_samples=TRAINING_BASLINE_SAMPLES)
            templates.append(DirectionTemplate(angle=angle, path=path, features=features))
        if len(templates) < 2:
            raise ValueError(f"{train_dir} 中至少需要两个方向 CSV 模板")
        return templates

    @property
    def angles(self) -> list[float]:
        return [template.angle for template in self.templates]

    def classify(
        self,
        samples: np.ndarray,
        timestamps: Optional[list[datetime]] = None,
        baseline_samples: int = TRAINING_BASLINE_SAMPLES,
    ) -> DirectionResult:
        features = extract_event_features(samples, timestamps, baseline_samples=baseline_samples)
        distances: dict[float, float] = {}
        for template in self.templates:
            distances[template.angle] = float(np.linalg.norm(features.vector - template.features.vector))

        ordered = sorted(distances.items(), key=lambda item: item[1])
        dist_values = np.asarray([dist for _, dist in ordered], dtype=float)
        adjusted = dist_values - float(np.min(dist_values))
        nonzero = adjusted[adjusted > 1e-9]
        if len(nonzero) == 0:
            temperature = 1.0
        else:
            temperature = max(float(np.percentile(nonzero, 35)) * 0.35, 1e-6)
        weights = np.exp(-adjusted / temperature)
        weights = weights / np.sum(weights)
        probabilities = {angle: float(weight) for (angle, _), weight in zip(ordered, weights)}

        x = 0.0
        y = 0.0
        for angle, probability in probabilities.items():
            radians = math.radians(angle)
            x += probability * math.sin(radians)
            y += probability * math.cos(radians)
        nearest_angle = ordered[0][0]
        confidence = float(weights[0])
        weighted_angle = math.degrees(math.atan2(x, y)) % 360.0
        angle = nearest_angle if confidence >= 0.55 else weighted_angle
        return DirectionResult(
            angle=angle,
            nearest_angle=nearest_angle,
            confidence=confidence,
            probabilities=dict(sorted(probabilities.items())),
            distances=distances,
            order=features.order,
        )

    def describe_templates(self) -> list[str]:
        lines: list[str] = []
        for template in self.templates:
            onset = [
                "-" if idx is None else f"{idx * template.features.sample_period:.1f}s"
                for idx in template.features.onset_indices
            ]
            order = " → ".join(str(sensor_id) for sensor_id in template.features.order)
            peak = ", ".join(f"{v:.0f}" for v in template.features.peak_delta)
            lines.append(
                f"{template.angle:>5.0f}°  顺序 {order:<13}  起始 {onset}  峰值Δ [{peak}]"
            )
        return lines


class SerialReader(threading.Thread):
    def __init__(self, port: str, baud: int, output_queue: queue.Queue[SensorFrame], stop_event: threading.Event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        if serial is None:
            self.output_queue.put(RuntimeError("未安装 pyserial，无法读取串口"))
            return
        try:
            with serial.Serial(self.port, self.baud, timeout=0.1) as ser:
                buffer = ""
                while not self.stop_event.is_set():
                    data = ser.read(128)
                    if not data:
                        continue
                    chunk = data.decode("utf-8", errors="ignore")
                    buffer += chunk
                    lines = re.split(r"\r?\n", buffer)
                    buffer = lines[-1]
                    for line in lines[:-1]:
                        parsed = parse_serial_line(line)
                        if parsed is None:
                            continue
                        values, alarm, alarm_channel = parsed
                        self.output_queue.put(
                            SensorFrame(values=values, timestamp=datetime.now(), alarm=alarm, alarm_channel=alarm_channel)
                        )
        except Exception as exc:  # pragma: no cover - interactive hardware path
            self.output_queue.put(exc)


class ReplayReader(threading.Thread):
    def __init__(
        self,
        path: Path,
        output_queue: queue.Queue[SensorFrame],
        stop_event: threading.Event,
        speed: float = 1.0,
    ):
        super().__init__(daemon=True)
        self.path = path
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.speed = max(speed, 0.05)

    def run(self) -> None:
        try:
            samples, timestamps = read_sensor_csv(self.path)
            period = estimate_sample_period(timestamps)
            for row, ts in zip(samples, timestamps):
                if self.stop_event.is_set():
                    break
                self.output_queue.put(SensorFrame(values=row, timestamp=ts))
                time.sleep(period / self.speed)
        except Exception as exc:
            self.output_queue.put(exc)


class GasDirectionApp:
    def __init__(
        self,
        root: tk.Tk,
        model: DirectionModel,
        baud: int = 9600,
        baseline_seconds: float = 3.0,
        pretrigger_seconds: float = 2.0,
        event_seconds: float = 5.0,
    ):
        self.root = root
        self.model = model
        self.baud = baud
        self.sample_period = DEFAULT_SAMPLE_PERIOD
        self.baseline_size = max(10, int(round(baseline_seconds / DEFAULT_SAMPLE_PERIOD)))
        self.pretrigger_size = max(10, int(round(pretrigger_seconds / DEFAULT_SAMPLE_PERIOD)))
        self.event_target_samples = max(20, int(round(event_seconds / DEFAULT_SAMPLE_PERIOD)))

        self.frame_queue: queue.Queue[SensorFrame | Exception] = queue.Queue()
        self.stop_event = threading.Event()
        self.reader: Optional[threading.Thread] = None

        self.idle_samples: list[np.ndarray] = []
        self.pretrigger_samples: list[SensorFrame] = []
        self.event_frames: list[SensorFrame] = []
        self.active_event = False
        self.event_baseline_count = self.pretrigger_size
        self.trigger_count = 0
        self.last_result: Optional[DirectionResult] = None
        self.hide_idle = tk.BooleanVar(value=True)

        self.colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
        self._build_ui()
        self._refresh_ports()
        self.root.after(80, self._process_queue)

    def _build_ui(self) -> None:
        self.root.title("气体来源方向识别")
        self.root.geometry("1280x780")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="串口").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(controls, textvariable=self.port_var, width=18)
        self.port_box.pack(side=tk.LEFT, padx=(6, 10))
        ttk.Button(controls, text="刷新", command=self._refresh_ports).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="连接串口", command=self._connect_serial).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="回放CSV", command=self._choose_replay).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="停止", command=self._stop_reader).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(controls, text="空闲时隐藏曲线", variable=self.hide_idle, command=self._redraw).pack(
            side=tk.LEFT, padx=14
        )

        self.status_var = tk.StringVar(value="未连接。请连接串口，或选择 CSV 回放测试。")
        ttk.Label(main, textvariable=self.status_var).pack(fill=tk.X, pady=(8, 4))

        result_bar = ttk.Frame(main)
        result_bar.pack(fill=tk.X, pady=(0, 8))
        self.angle_var = tk.StringVar(value="方向角：--")
        self.nearest_var = tk.StringVar(value="最近模板：--")
        self.order_var = tk.StringVar(value="响应顺序：--")
        self.confidence_var = tk.StringVar(value="置信度：--")
        for var in (self.angle_var, self.nearest_var, self.confidence_var, self.order_var):
            ttk.Label(result_bar, textvariable=var, font=("Microsoft YaHei", 12, "bold")).pack(side=tk.LEFT, padx=12)

        current_bar = ttk.Frame(main)
        current_bar.pack(fill=tk.X, pady=(0, 8))
        self.sensor_vars = [tk.StringVar(value=f"S{i}: --") for i in range(1, SENSOR_COUNT + 1)]
        for idx, var in enumerate(self.sensor_vars):
            label = ttk.Label(current_bar, textvariable=var, foreground=self.colors[idx])
            label.pack(side=tk.LEFT, padx=14)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.wave_ax = self.fig.add_subplot(1, 2, 1)
        self.radar_ax = self.fig.add_subplot(1, 2, 2, projection="polar")
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.wave_lines = [
            self.wave_ax.plot([], [], color=self.colors[i], lw=2, label=f"传感器{i + 1}")[0]
            for i in range(SENSOR_COUNT)
        ]
        self.wave_ax.legend(loc="upper left")
        self._draw_empty_radar()
        self._redraw()

    def _refresh_ports(self) -> None:
        if serial is None:
            self.port_box["values"] = []
            return
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def _connect_serial(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("提示", "请先选择串口")
            return
        self._start_reader(SerialReader(port, self.baud, self.frame_queue, self.stop_event))
        self.status_var.set(f"已连接 {port}，等待气体响应...")

    def _choose_replay(self) -> None:
        path = filedialog.askopenfilename(
            title="选择方向 CSV 数据",
            initialdir=str(self.model.train_dir),
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self._start_reader(ReplayReader(Path(path), self.frame_queue, self.stop_event, speed=1.0))
        self.status_var.set(f"正在回放 {Path(path).name}")

    def _start_reader(self, reader: threading.Thread) -> None:
        self._stop_reader()
        self._reset_detection(clear_display=True)
        self.stop_event = threading.Event()
        if isinstance(reader, SerialReader):
            reader.stop_event = self.stop_event
        elif isinstance(reader, ReplayReader):
            reader.stop_event = self.stop_event
        self.reader = reader
        self.reader.start()

    def _stop_reader(self) -> None:
        if self.reader and self.reader.is_alive():
            self.stop_event.set()
            self.reader.join(timeout=0.5)
        self.reader = None

    def _reset_detection(self, clear_display: bool = False) -> None:
        self.idle_samples.clear()
        self.pretrigger_samples.clear()
        self.event_frames.clear()
        self.active_event = False
        self.event_baseline_count = self.pretrigger_size
        self.trigger_count = 0
        self.last_result = None
        if clear_display:
            self.angle_var.set("方向角：--")
            self.nearest_var.set("最近模板：--")
            self.confidence_var.set("置信度：--")
            self.order_var.set("响应顺序：--")
            self._draw_empty_radar()
            self._redraw()

    def _process_queue(self) -> None:
        processed = 0
        while processed < 80:
            try:
                item = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if isinstance(item, Exception):
                self.status_var.set(f"读取失败：{item}")
                self._stop_reader()
                break
            self._handle_frame(item)
        self._redraw()
        self.root.after(80, self._process_queue)

    def _handle_frame(self, frame: SensorFrame) -> None:
        for i, value in enumerate(frame.values):
            self.sensor_vars[i].set(f"S{i + 1}: {value:.0f}")

        self.pretrigger_samples.append(frame)
        self.pretrigger_samples = self.pretrigger_samples[-self.pretrigger_size :]

        if not self.active_event:
            self.idle_samples.append(frame.values)
            self.idle_samples = self.idle_samples[-self.baseline_size :]
            if len(self.idle_samples) < min(12, self.baseline_size):
                self.status_var.set("正在建立空闲基线...")
                return

            baseline = np.median(np.asarray(self.idle_samples), axis=0)
            noise = np.std(np.asarray(self.idle_samples), axis=0)
            delta = frame.values - baseline
            dynamic_threshold = np.maximum(90.0, noise * 6.0)
            is_gas = (
                frame.alarm == 1
                or np.max(delta) > 180.0
                or np.sum(delta > dynamic_threshold) >= 2
                or np.sum(np.maximum(delta, 0.0)) > 320.0
            )

            self.trigger_count = self.trigger_count + 1 if is_gas else 0
            if self.trigger_count >= 2:
                self.active_event = True
                self.event_frames = list(self.pretrigger_samples)
                self.event_baseline_count = len(self.event_frames)
                self.status_var.set("检测到气体，正在提取响应先后顺序...")
            elif self.hide_idle.get():
                self.event_frames.clear()
            return

        self.event_frames.append(frame)
        baseline_count = self.event_baseline_count
        enough_samples = len(self.event_frames) >= baseline_count + max(18, self.event_target_samples // 3)
        final_window = len(self.event_frames) >= baseline_count + self.event_target_samples
        if enough_samples:
            self._classify_current_event(baseline_count=baseline_count)
            if final_window:
                self.status_var.set("已识别，可继续观察；停止后可重新连接或回放。")

    def _classify_current_event(self, baseline_count: int) -> None:
        samples = np.asarray([frame.values for frame in self.event_frames], dtype=float)
        timestamps = [frame.timestamp for frame in self.event_frames]
        try:
            result = self.model.classify(samples, timestamps, baseline_samples=max(5, baseline_count))
        except ValueError:
            return
        self.last_result = result
        self.angle_var.set(f"方向角：{result.angle:.1f}°")
        self.nearest_var.set(f"最近模板：{result.nearest_angle:.0f}°")
        self.confidence_var.set(f"置信度：{result.confidence:.2f}")
        self.order_var.set("响应顺序：" + " → ".join(str(sensor_id) for sensor_id in result.order))
        self._draw_radar(result)

    def _redraw(self) -> None:
        show_data = self.active_event or not self.hide_idle.get()
        if show_data and self.event_frames:
            samples = np.asarray([frame.values for frame in self.event_frames], dtype=float)
            baseline_count = min(max(3, self.event_baseline_count), len(samples))
            baseline = np.median(samples[:baseline_count], axis=0)
            delta = samples - baseline
            times = np.arange(len(samples)) * DEFAULT_SAMPLE_PERIOD
            for idx, line in enumerate(self.wave_lines):
                line.set_data(times, delta[:, idx])
            self.wave_ax.set_title("五路传感器实时响应波形（相对基线）")
            self.wave_ax.set_xlabel("时间 / s")
            self.wave_ax.set_ylabel("响应值 ΔADC")
            self.wave_ax.grid(True, alpha=0.25)
            self.wave_ax.relim()
            self.wave_ax.autoscale_view()
        else:
            for line in self.wave_lines:
                line.set_data([], [])
            self.wave_ax.set_title("未检测到气体")
            self.wave_ax.set_xlabel("时间 / s")
            self.wave_ax.set_ylabel("响应值 ΔADC")
            self.wave_ax.set_xlim(0, 5)
            self.wave_ax.set_ylim(0, 1)
            self.wave_ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def _draw_empty_radar(self) -> None:
        self.radar_ax.clear()
        self.radar_ax.set_theta_zero_location("N")
        self.radar_ax.set_theta_direction(-1)
        self.radar_ax.set_title("方向雷达图")
        ticks = np.radians(self.model.angles)
        self.radar_ax.set_xticks(ticks)
        self.radar_ax.set_xticklabels([f"{int(angle)}°" for angle in self.model.angles])
        self.radar_ax.set_ylim(0, 1)
        self.radar_ax.text(0, 0.45, "等待气体", ha="center", va="center", fontsize=13)

    def _draw_radar(self, result: DirectionResult) -> None:
        self.radar_ax.clear()
        self.radar_ax.set_theta_zero_location("N")
        self.radar_ax.set_theta_direction(-1)
        angles = sorted(result.probabilities)
        theta = np.radians(angles)
        values = np.asarray([result.probabilities[angle] for angle in angles], dtype=float)
        theta_closed = np.r_[theta, theta[0]]
        values_closed = np.r_[values, values[0]]
        self.radar_ax.plot(theta_closed, values_closed, color="#0b7285", lw=2)
        self.radar_ax.fill(theta_closed, values_closed, color="#66d9e8", alpha=0.35)
        self.radar_ax.bar(theta, values, width=math.radians(28), color="#4dabf7", alpha=0.45)
        arrow_theta = math.radians(result.angle)
        self.radar_ax.annotate(
            "",
            xy=(arrow_theta, 1.0),
            xytext=(arrow_theta, 0.05),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": "#e03131"},
        )
        self.radar_ax.set_xticks(theta)
        self.radar_ax.set_xticklabels([f"{int(angle)}°" for angle in angles])
        self.radar_ax.set_ylim(0, max(0.45, float(values.max()) * 1.15))
        self.radar_ax.set_title(f"方向雷达图：{result.angle:.1f}°")


def run_self_test(model: DirectionModel) -> int:
    print("训练模板：")
    for line in model.describe_templates():
        print("  " + line)
    print("\n离线回放识别：")
    ok = 0
    total = 0
    for template in model.templates:
        samples, timestamps = read_sensor_csv(template.path)
        result = model.classify(samples, timestamps, baseline_samples=TRAINING_BASLINE_SAMPLES)
        total += 1
        matched = abs(((result.nearest_angle - template.angle + 180) % 360) - 180) < 1e-6
        ok += int(matched)
        print(
            f"  {template.path.name:<8} -> 估计 {result.angle:6.1f}°，"
            f"最近模板 {result.nearest_angle:>5.0f}°，置信度 {result.confidence:.2f}，"
            f"顺序 {'→'.join(map(str, result.order))}"
        )
    print(f"\n自检结果：{ok}/{total} 个训练文件最近模板识别正确")
    return 0 if ok == total else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="五路气体传感阵列来源方向识别")
    parser.add_argument("--train-dir", default="方向数据", help="方向训练 CSV 文件夹")
    parser.add_argument("--port", default="", help="启动后自动连接的串口，例如 COM3")
    parser.add_argument("--baud", type=int, default=9600, help="串口波特率，默认 9600")
    parser.add_argument("--replay", default="", help="启动后自动回放某个 CSV 文件")
    parser.add_argument("--self-test", action="store_true", help="只做离线自检，不打开界面")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    train_dir = Path(args.train_dir)
    model = DirectionModel(train_dir)

    if args.self_test:
        return run_self_test(model)

    root = tk.Tk()
    app = GasDirectionApp(root, model=model, baud=args.baud)
    if args.port:
        app.port_var.set(args.port)
        root.after(300, app._connect_serial)
    if args.replay:
        replay_path = Path(args.replay)
        root.after(300, lambda: app._start_reader(ReplayReader(replay_path, app.frame_queue, app.stop_event)))
        app.status_var.set(f"即将回放 {replay_path.name}")
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_reader(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
