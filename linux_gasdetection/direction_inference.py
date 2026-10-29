"""Gas source direction interface for Linux (Tk GUI).

This is the Linux port of the Windows program ``方向定位模型/direction_inference.py``
with the same behaviour:

1. connect the sensor serial port and collect a quiet baseline,
2. the first reliable positive rise on any of S1-S6 starts a fixed
   observation window (default 5 s),
3. every sensor onset inside the window is collected and the direction is
   judged once; the result locks until Reset.

Extra Linux conveniences: automatic /dev/ttyUSB* /dev/ttyACM* port listing,
CSV replay (so the interface can be tested without hardware) and the same
optional UDP/serial JSON sinks for the robot controller.

If no display is available (for example an SSH session), use the headless
tool instead: ``python3 direction_cli.py --port /dev/ttyUSB0``.
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from direction_core import (
    SENSOR_COUNT,
    DetectionSession,
    DirectionEstimator,
    ReplayReader,
    SerialFrameReader,
    create_sink,
    list_serial_ports,
)


REPLAY_SPEEDS = ("1x", "2x", "5x", "10x", "50x", "Instant")


class DirectionGUI:
    def __init__(self, root: tk.Tk, estimator: DirectionEstimator,
                 port: str | None, baud: int, sink, initial_replay: Path | None = None):
        self.status_var = tk.StringVar(value="Ready")
        self.root = root
        self.estimator = estimator
        self.port = port
        self.baud = baud
        self.sink = sink
        self.serial_port = None
        self.reader_thread = None
        self.stop_event = None
        self.session = DetectionSession(estimator, sink, self.update_plot, self.show_result, self.show_status)
        self.root.title(f"Gas direction inference - {estimator.method}")
        self.root.geometry("1050x760")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        if initial_replay is not None:
            self.root.after(600, lambda: self.start_replay(initial_replay))

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Model:").pack(side=tk.LEFT, padx=(0, 4))
        self.model_var = tk.StringVar(value=self.estimator.model_path.name)
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, width=36, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=4)
        self.refresh_model_choices()
        ttk.Button(top, text="Load selected model", command=self.load_selected_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Browse...", command=self.browse_model).pack(side=tk.LEFT, padx=4)

        second = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        second.pack(fill=tk.X)
        ttk.Label(second, text="Serial port:").pack(side=tk.LEFT, padx=(0, 4))
        self.port_var = tk.StringVar(value=self.port or "")
        self.port_combo = ttk.Combobox(second, textvariable=self.port_var, width=24)
        self.port_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(second, text="Refresh ports", command=self.refresh_ports).pack(side=tk.LEFT, padx=4)
        ttk.Button(second, text="Connect sensor serial", command=self.connect_sensor).pack(side=tk.LEFT, padx=4)
        ttk.Button(second, text="Replay CSV...", command=self.choose_replay).pack(side=tk.LEFT, padx=4)
        ttk.Label(second, text="Speed:").pack(side=tk.LEFT, padx=(12, 2))
        self.speed_var = tk.StringVar(value="1x")
        ttk.Combobox(second, textvariable=self.speed_var, values=REPLAY_SPEEDS, width=7, state="readonly").pack(side=tk.LEFT)
        ttk.Button(second, text="Stop", command=self.stop_reader).pack(side=tk.LEFT, padx=10)
        ttk.Button(second, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Armed - waiting for a connection")
        ttk.Label(self.root, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(pady=3)

        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.figure.add_subplot(111, polar=True)
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_ylim(0, 1.25)
        self.ax.set_yticklabels([])
        self.ax.set_title("Gas source direction (front = 0 deg)")
        sensor_angles = np.deg2rad(np.arange(0, 360, 60))
        self.ax.scatter(sensor_angles, np.ones(SENSOR_COUNT), c=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"], s=150)
        for index, theta in enumerate(sensor_angles, start=1):
            self.ax.text(theta, 1.13, f"S{index}", ha="center", va="center", fontweight="bold")
        self.rank_texts = [
            self.ax.text(theta, 0.82, "", ha="center", va="center", fontsize=10, fontweight="bold",
                         bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#34495e", "alpha": 0.9})
            for theta in sensor_angles
        ]
        self.direction_line, = self.ax.plot([], [], color="#d62728", linewidth=3, marker="o", markersize=9)
        self.sensor_line, = self.ax.plot([], [], color="#1f77b4", linewidth=2, alpha=0.8)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_var = tk.StringVar(value="No direction has been judged")
        ttk.Label(self.root, textvariable=self.result_var, font=("Arial", 13, "bold")).pack(pady=8)
        self.sequence_var = tk.StringVar(value="Response order: waiting for gas detection")
        ttk.Label(self.root, textvariable=self.sequence_var, font=("Arial", 11, "bold")).pack(pady=(0, 10))
        self.refresh_ports()

    # ---------------------------------------------------------------- status
    def show_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def show_result(self, angle, confidence, diagnostics):
        def update():
            angle_text = f"{int(angle)}" if float(angle).is_integer() else f"{angle:.1f}"
            fixed_label = " (sequential first-response estimate)" if self.estimator.method == "first_response" else ""
            self.result_var.set(f"Estimated source: {angle_text} deg{fixed_label}    confidence: {confidence:.1%}")
            sequence = diagnostics["response_sequence"]
            ranks = diagnostics["response_rank"]
            unresponded = diagnostics.get("unresponded_sensors", [])
            responded = [sensor for sensor in sequence if sensor not in unresponded]
            response_order_all = diagnostics.get("response_order_all")
            if response_order_all is None:
                response_order_all = responded + unresponded
            sequence_text = "Response order (all 6): " + " -> ".join(
                f"S{sensor}" if sensor in responded else f"S{sensor}(no response)"
                for sensor in response_order_all
            )
            if "first_sensor_azimuth_deg" in diagnostics:
                sequence_text += f" | First sensor: S{diagnostics['first_response_sensor']} ({diagnostics['first_sensor_azimuth_deg']:.0f} deg)"
            if self.estimator.method == "first_response" and diagnostics.get("first_response_corrections"):
                correction_text = ", ".join(
                    f"S{item['sensor']} d{item['signed_difference_from_first_deg']:+.1f}x{item['weight']:.2f}={item['correction_deg']:+.1f}"
                    for item in diagnostics["first_response_corrections"]
                )
                sequence_text += f" | sequential corrections: {correction_text}"
            if "first_response_sensor" in diagnostics:
                score = diagnostics.get("validation_score")
                score_text = f" ({score:.0%})" if score is not None else ""
                sequence_text += f" | Validation: {diagnostics['validation']}{score_text}"
            self.sequence_var.set(sequence_text)
            for sensor, rank in enumerate(ranks[:SENSOR_COUNT]):
                self.rank_texts[sensor].set_text("-" if sensor + 1 in unresponded else f"#{rank + 1}")
            theta = np.deg2rad(angle)
            self.direction_line.set_data([theta, theta], [0, 1.08])
            self.canvas.draw_idle()
        self.root.after(0, update)

    def update_plot(self, samples):
        if samples.size == 0:
            return
        def update():
            baseline = self.session.baseline
            if baseline is None:
                baseline_count = min(max(8, len(samples) // 8), len(samples))
                baseline = np.median(samples[:baseline_count], axis=0)
            # Match the detector: downward movement is not a gas response.
            response = np.maximum(samples[-1] - baseline, 0.0)
            response = response / max(float(np.max(response)), 1e-9)
            theta = np.deg2rad(np.arange(0, 360, 60))
            self.sensor_line.set_data(np.r_[theta, theta[0]], np.r_[response, response[0]])
            self.canvas.draw_idle()
        self.root.after(0, update)

    def reset(self):
        self.session.reset()
        self.result_var.set("No direction has been judged")
        self.sequence_var.set("Response order: waiting for gas detection")
        self.direction_line.set_data([], [])
        self.sensor_line.set_data([], [])
        for rank_text in self.rank_texts:
            rank_text.set_text("")
        self.canvas.draw_idle()

    # ----------------------------------------------------------------- models
    def refresh_model_choices(self):
        model_dir = Path(__file__).resolve().parent / "models"
        choices = sorted(path.name for path in model_dir.glob("*.joblib"))
        current = self.estimator.model_path.name
        if current not in choices and self.estimator.model_path.exists():
            choices.insert(0, current)
        self.model_combo["values"] = choices
        if choices and not self.model_var.get():
            self.model_var.set(choices[0])

    def load_selected_model(self):
        selected = self.model_var.get().strip()
        if not selected:
            self.browse_model()
            return
        self.load_model(Path(__file__).resolve().parent / "models" / selected)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("Direction models", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        self.load_model(Path(path))

    def load_model(self, path: Path):
        try:
            self.estimator = DirectionEstimator(path)
            self.session.estimator = self.estimator
            self.root.title(f"Gas direction inference - {self.estimator.method}")
            self.model_var.set(path.name)
            self.refresh_model_choices()
            self.reset()
        except Exception as error:
            messagebox.showerror("Model load failed", str(error))

    # --------------------------------------------------------------- sources
    def refresh_ports(self):
        ports = list_serial_ports()
        values = [port.device for port in ports]
        self.port_combo["values"] = values
        if not self.port_var.get() and values:
            self.port_var.set(values[0])

    def connect_sensor(self):
        device = (self.port or self.port_var.get()).strip()
        if not device:
            ports = list_serial_ports()
            if not ports:
                messagebox.showerror("No serial port", "No sensor serial port was found. Check the cable and try Refresh ports.")
                return
            device = ports[0].device
        self.port = None
        self.port_var.set(device)
        self.stop_reader()
        self.reset()
        self.stop_event = threading.Event()
        self.reader_thread = SerialFrameReader(device, self.baud, self.session, self.stop_event, self.on_reader_error)
        self.reader_thread.start()
        self.show_status(f"Sensor connected: {device} @ {self.baud} baud")

    def choose_replay(self):
        path = filedialog.askopenfilename(
            title="Choose a recorded CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.start_replay(Path(path))

    def start_replay(self, path: Path):
        speed_text = self.speed_var.get().rstrip("x")
        speed = 0.0 if self.speed_var.get() == "Instant" else float(speed_text)
        self.stop_reader()
        self.reset()
        self.stop_event = threading.Event()
        self.reader_thread = ReplayReader(
            path, self.session, speed=speed, stop_event=self.stop_event,
            on_error=self.on_reader_error,
            on_end=lambda: self.show_status("Replay finished"),
        )
        self.reader_thread.start()
        self.show_status(f"Replaying {path.name} at {self.speed_var.get()}")

    def stop_reader(self):
        if self.reader_thread is not None and self.reader_thread.is_alive():
            if self.stop_event is not None:
                self.stop_event.set()
            self.reader_thread.join(timeout=0.5)
        self.reader_thread = None

    def on_reader_error(self, message: str):
        self.root.after(0, messagebox.showerror, "Read failed", message)
        self.root.after(0, self.status_var.set, f"Read failed: {message}")

    def close(self):
        if self.stop_event is not None:
            self.stop_event.set()
        self.stop_reader()
        self.sink.close()
        self.root.destroy()


def build_estimator(args) -> DirectionEstimator:
    model_dir = Path(__file__).resolve().parent / "models"
    model_path = args.model
    if model_path is None:
        model_path = model_dir / "direction_model_first_response.joblib"
        if not model_path.exists():
            model_path = model_dir / "direction_model_order.joblib"
    model_path = Path(model_path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Train models with train_all.sh or copy a .joblib into models/.")
    estimator = DirectionEstimator(model_path)
    if args.observe_seconds is not None:
        estimator.config.primary_observe_seconds = float(args.observe_seconds)
    if args.primary_sensors:
        try:
            sensors = tuple(int(value.strip()) - 1 for value in args.primary_sensors.split(",") if value.strip())
        except ValueError as error:
            raise SystemExit("--primary-sensors must look like 1,3,5") from error
        if not sensors or any(index not in range(SENSOR_COUNT) for index in sensors):
            raise SystemExit("--primary-sensors must be sensor numbers between 1 and 6")
        estimator.config.primary_sensors = sensors
    if args.first_response_weights:
        try:
            weights = tuple(float(value.strip()) for value in args.first_response_weights.split(",") if value.strip())
        except ValueError as error:
            raise SystemExit("--first-response-weights must contain numbers") from error
        if len(weights) != SENSOR_COUNT - 1 or any(value < 0 for value in weights):
            raise SystemExit("--first-response-weights must contain five non-negative numbers")
        estimator.config.first_response_rank_weights = weights
    return estimator


def main():
    parser = argparse.ArgumentParser(description="Linux gas direction interface: connect the sensor serial port and judge one direction per detection.")
    parser.add_argument("--model", type=Path, default=None, help="Model file; default models/direction_model_first_response.joblib")
    parser.add_argument("--port", help="Sensor serial device, e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--robot-output", help="Optional udp:HOST:PORT or serial:DEVICE JSON output")
    parser.add_argument("--robot-baud", type=int, default=115200)
    parser.add_argument("--observe-seconds", type=float, default=None, help="Observation window length after the first response (default 5 s).")
    parser.add_argument("--primary-sensors", default=None, help="Optional trigger sensor list; any one starts the window, default 1,2,3,4,5,6.")
    parser.add_argument("--first-response-weights", default=None, help="Five comma-separated correction weights for first_response.")
    parser.add_argument("--replay", type=Path, default=None, help="Start by replaying a recorded CSV instead of a live port.")
    args = parser.parse_args()

    estimator = build_estimator(args)
    try:
        sink = create_sink(args.robot_output, args.robot_baud)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit(
            "Cannot open a display (is DISPLAY set?). Use the headless tool instead:\n"
            "    python3 direction_cli.py --port /dev/ttyUSB0\n"
            f"Details: {error}"
        )
    DirectionGUI(root, estimator, args.port, args.baud, sink, initial_replay=args.replay)
    root.mainloop()


if __name__ == "__main__":
    main()
