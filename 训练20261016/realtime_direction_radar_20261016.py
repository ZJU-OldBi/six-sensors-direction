"""Real-time gas direction radar for the 2026-10-16 four-direction model."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from direction_order_20261016 import (
    PRIMARY,
    REFERENCE,
    analyze_record,
    feature_vector,
    parse_serial_line,
)


SENSOR_ANGLES = np.arange(0.0, 360.0, 60.0)


class RadarApp:
    def __init__(self, root: tk.Tk, model_path: Path, port: str | None,
                 baud: int, simulate: bool, baseline_frames: int = 30,
                 capture_frames: int = 30):
        self.root = root
        self.root.title("2026-10-16 气体来源实时方向雷达")
        self.root.geometry("980x700")
        self.model_path = model_path
        self.artifact = joblib.load(model_path)
        self.model = self.artifact["model"]
        self.port_name = port
        self.baud = baud
        self.simulate = simulate
        self.baseline_frames = baseline_frames
        self.capture_frames = capture_frames
        self.input_queue: queue.Queue[list[float]] = queue.Queue()
        self.stop_event = threading.Event()
        self.serial_thread: threading.Thread | None = None
        self.simulation_index = 0
        self.simulation_angle = 0
        self.reset_session()
        self.build_ui()
        self.draw_radar()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self.poll_input)
        if self.simulate:
            self.root.after(100, self.feed_simulation)
        elif self.port_name:
            self.start_serial()

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        figure = plt.Figure(figsize=(7.2, 6.0), dpi=100)
        self.axes = figure.add_subplot(111, projection="polar")
        self.canvas = FigureCanvasTkAgg(figure, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        panel = ttk.Frame(main)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="实时气体来源", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w", pady=(8, 18))
        self.status_var = tk.StringVar(value="正在建立基线")
        ttk.Label(panel, textvariable=self.status_var, wraplength=240).grid(
            row=1, column=0, sticky="w", pady=(0, 20))
        self.angle_var = tk.StringVar(value="方向角度：--")
        ttk.Label(panel, textvariable=self.angle_var, font=("Segoe UI", 20, "bold")).grid(
            row=2, column=0, sticky="w", pady=(0, 18))
        self.order_var = tk.StringVar(value="主通道顺序：--")
        ttk.Label(panel, textvariable=self.order_var, wraplength=240).grid(
            row=3, column=0, sticky="w", pady=(0, 8))
        self.reference_var = tk.StringVar(value="参考通道顺序：--")
        ttk.Label(panel, textvariable=self.reference_var, wraplength=240).grid(
            row=4, column=0, sticky="w", pady=(0, 18))
        self.confidence_var = tk.StringVar(value="置信度：--")
        ttk.Label(panel, textvariable=self.confidence_var).grid(row=5, column=0, sticky="w")
        self.source_var = tk.StringVar(value="数据源：模拟数据" if self.simulate else f"数据源：{self.port_name or '未连接'}")
        ttk.Label(panel, textvariable=self.source_var, wraplength=240).grid(row=6, column=0, sticky="w", pady=(14, 20))
        ttk.Button(panel, text="复位并重新检测", command=self.reset_session).grid(
            row=7, column=0, sticky="ew", pady=(8, 8))
        ttk.Button(panel, text="退出", command=self.close).grid(row=8, column=0, sticky="ew")
        note = "主判据：S1、S3、S5 响应先后\nS2、S4、S6 仅作参考\n未确认气体前不输出方向"
        ttk.Label(panel, text=note, foreground="#555", wraplength=240).grid(
            row=9, column=0, sticky="w", pady=(24, 0))

    def reset_session(self):
        self.frames: list[list[float]] = []
        self.baseline: np.ndarray | None = None
        self.noise: np.ndarray | None = None
        self.state = "baseline"
        self.trigger_hits = 0
        self.simulation_index = 0
        self.response_order = "--"
        self.reference_order = "--"
        self.angle = None
        self.confidence = None
        if hasattr(self, "status_var"):
            self.status_var.set(f"正在建立基线：0/{self.baseline_frames} 帧")
            self.angle_var.set("方向角度：--")
            self.order_var.set("主通道顺序：--")
            self.reference_var.set("参考通道顺序：--")
            self.confidence_var.set("置信度：--")
            self.draw_radar()

    def start_serial(self):
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()

    def read_serial(self):
        try:
            import serial
            serial_port = serial.Serial(self.port_name, self.baud, timeout=1.0)
        except Exception as error:
            self.input_queue.put([float("nan")] * 6)
            self.root.after(0, lambda: messagebox.showerror("串口错误", str(error)))
            return
        try:
            while not self.stop_event.is_set():
                line = serial_port.readline().decode("utf-8", errors="ignore")
                values = parse_serial_line(line)
                if values is not None:
                    self.input_queue.put(values)
        finally:
            serial_port.close()

    def feed_simulation(self):
        if self.stop_event.is_set():
            return
        if self.state == "locked":
            return
        frame_no = self.simulation_index
        base = np.asarray([180.0, 2500.0, 85.0, 1950.0, 930.0, 930.0])
        rng = np.random.default_rng(frame_no + 20261016)
        values = base + rng.normal(0.0, [1.0, 2.0, 1.0, 2.0, 2.0, 2.0])
        event_start = self.baseline_frames + 4
        orders = {
            0: (0, 2, 4),
            90: (2, 0, 4),
            180: (4, 2, 0),
            270: (0, 4, 2),
        }
        if frame_no >= event_start:
            order = orders[self.simulation_angle]
            for rank, sensor in enumerate(order):
                age = frame_no - event_start - rank * 5
                if age >= 0:
                    values[sensor] += min(80.0, age * 12.0)
        self.input_queue.put(values.tolist())
        self.simulation_index += 1
        self.root.after(100, self.feed_simulation)

    def poll_input(self):
        processed = 0
        while processed < 20:
            try:
                values = self.input_queue.get_nowait()
            except queue.Empty:
                break
            if not np.all(np.isfinite(values)):
                processed += 1
                continue
            self.add_frame(values)
            processed += 1
        self.root.after(40, self.poll_input)

    def add_frame(self, values: list[float]):
        if self.state == "locked":
            return
        values = np.asarray(values, dtype=float)
        self.frames.append(values.tolist())
        if self.state == "baseline":
            if len(self.frames) >= self.baseline_frames:
                base = np.asarray(self.frames[:self.baseline_frames], dtype=float)
                self.baseline = np.median(base, axis=0)
                self.noise = np.maximum(1.4826 * np.median(np.abs(base - self.baseline), axis=0), 1e-6)
                self.state = "waiting"
                self.status_var.set("基线已建立，等待持续向上响应")
            else:
                self.status_var.set(f"正在建立基线：{len(self.frames)}/{self.baseline_frames} 帧")
        elif self.state == "waiting":
            positive = values - self.baseline
            # A real event must be clearly above both baseline noise and the
            # 10% upward-change requirement, then remain there for several
            # frames before response-order capture starts.
            threshold = np.maximum(5.0 * self.noise, 0.10 * np.maximum(np.abs(self.baseline), 1.0))
            if np.any(positive >= threshold):
                self.trigger_hits += 1
            else:
                self.trigger_hits = max(0, self.trigger_hits - 1)
            if self.trigger_hits >= 5:
                self.state = "capturing"
                self.capture_start = len(self.frames)
                self.status_var.set("检测到气体，正在捕获响应顺序")
        elif self.state == "capturing":
            captured = len(self.frames) - self.capture_start
            self.status_var.set(f"检测到气体，捕获响应：{captured}/{self.capture_frames} 帧")
            if captured >= self.capture_frames:
                self.finish_detection()
        self.update_live_order()
        self.draw_radar()

    def update_live_order(self):
        if len(self.frames) < 12:
            return
        info = analyze_record(np.asarray(self.frames, dtype=float), baseline_samples=self.baseline_frames)
        self.response_order = info["primary_order"]
        self.reference_order = info["reference_order"]
        self.order_var.set(f"主通道顺序：{self.response_order}")
        self.reference_var.set(f"参考通道顺序：{self.reference_order}")

    def finish_detection(self):
        info = analyze_record(np.asarray(self.frames, dtype=float), baseline_samples=self.baseline_frames)
        self.response_order = info["primary_order"]
        self.reference_order = info["reference_order"]
        self.order_var.set(f"主通道顺序：{self.response_order}")
        self.reference_var.set(f"参考通道顺序：{self.reference_order}")
        if not info["gas_detected"]:
            self.state = "waiting"
            self.status_var.set("未确认气体，继续等待")
            return
        if not info["primary_complete"]:
            self.state = "locked"
            self.status_var.set("已检测到气体，但 1、3、5 未全部响应，暂不判断")
            return
        probabilities = self.model.predict_proba([feature_vector(info)])[0]
        classes = np.asarray(self.model.classes_, dtype=float)
        best = int(np.argmax(probabilities))
        self.angle = float(classes[best])
        self.confidence = float(probabilities[best])
        self.state = "locked"
        self.angle_var.set(f"方向角度：{self.angle:.0f}°")
        self.confidence_var.set(f"置信度：{self.confidence:.1%}")
        self.status_var.set("方向已判断，点击复位开始下一次检测")

    def draw_radar(self):
        if not hasattr(self, "axes"):
            return
        ax = self.axes
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.35)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(np.deg2rad(SENSOR_ANGLES))
        ax.set_xticklabels([f"S{i}" for i in range(1, 7)], fontsize=12)
        ax.grid(alpha=0.28)
        response = np.zeros(6)
        if self.frames and self.baseline is not None:
            current = np.asarray(self.frames[-1], dtype=float)
            rise = np.maximum(current - self.baseline, 0.0)
            scale = np.maximum(8.0 * self.noise, np.max(rise) if np.max(rise) > 0 else 1.0)
            response = np.clip(rise / scale, 0.0, 1.0)
        theta = np.deg2rad(SENSOR_ANGLES)
        ax.plot(np.r_[theta, theta[0]], np.r_[response, response[0]], color="#1976d2", linewidth=2)
        ax.fill(np.r_[theta, theta[0]], np.r_[response, response[0]], color="#64b5f6", alpha=0.25)
        ax.scatter(theta[list(PRIMARY)], response[list(PRIMARY)], s=80, color="#d32f2f", label="主通道 1/3/5", zorder=3)
        ax.scatter(theta[list(REFERENCE)], response[list(REFERENCE)], s=55, color="#757575", label="参考通道 2/4/6", zorder=3)
        if self.angle is not None:
            direction = np.deg2rad(self.angle)
            ax.plot([direction, direction], [0, 1.12], color="#ef6c00", linewidth=3)
            ax.scatter([direction], [1.12], color="#ef6c00", s=90, zorder=5)
            ax.text(direction, 1.25, f"来源 {self.angle:.0f}°", ha="center", va="center", color="#ef6c00", fontweight="bold")
        title = "气体来源雷达"
        if self.state != "locked":
            title += "（等待检测）"
        ax.set_title(title, pad=20, fontsize=16)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
        self.canvas.draw_idle()

    def close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="实时气体来源雷达")
    base = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, default=base / "models" / "direction_order_model_20261016.joblib")
    parser.add_argument("--port", help="串口，例如 COM5 或 /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--simulate", action="store_true", help="使用内置模拟数据预览雷达")
    parser.add_argument("--baseline-frames", type=int, default=60)
    parser.add_argument("--capture-frames", type=int, default=60)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"模型不存在，请先训练：{args.model}")
    if not args.port and not args.simulate:
        args.simulate = True
    root = tk.Tk()
    RadarApp(root, args.model, args.port, args.baud, args.simulate,
             args.baseline_frames, args.capture_frames)
    root.mainloop()


if __name__ == "__main__":
    main()
