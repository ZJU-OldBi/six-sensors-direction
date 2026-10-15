"""Six-channel gas sensor monitor.

Serial protocol (one frame per line):
sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,alarm_status,alarm_channel\n
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import serial
import serial.tools.list_ports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SENSOR_COUNT = 6
FRAME_FIELD_COUNT = SENSOR_COUNT + 2
BAUD_RATE = 9600
MAX_DATA_POINTS = 100
COLORS = ("#00a8ff", "#87ceeb", "#00ffff", "#ff8c00", "#ff4757", "#b388ff")

matplotlib.rcParams["axes.unicode_minus"] = False


class SensorMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Gas Sensor Monitor - 6 Channels")
        self.root.geometry("1280x860")
        self.root.minsize(980, 700)
        self.root.configure(bg="#0a1128")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.ser = None
        self.is_connected = False
        self.is_recording = False
        self.stop_event = threading.Event()
        self.buffer = ""
        self.x_index = 0
        self.timestamps = []
        self.data = {f"sensor{i}": [] for i in range(1, SENSOR_COUNT + 1)}
        self.recorded_data = []
        self.alarm_status = 0
        self.alarm_channel = 0
        self.sensor_vars = {}
        self.lines = {}
        self.data_labels = {}

        self.create_styles()
        self.create_layout()
        self.reader_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.reader_thread.start()

    def create_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#0a1128")
        style.configure("TLabel", background="#0a1128", foreground="#d7efff", font=("Arial", 11))
        style.configure("Title.TLabel", font=("Arial", 22, "bold"), foreground="#ffffff")
        style.configure("Subtitle.TLabel", font=("Arial", 12), foreground="#87ceeb")
        style.configure("Value.TLabel", font=("Arial", 18, "bold"), foreground="#00a8ff")
        style.configure("TButton", background="#003366", foreground="#ffffff", font=("Arial", 10, "bold"))
        style.map("TButton", background=[("active", "#005a9c")])
        style.configure("TCheckbutton", background="#0a1128", foreground="#d7efff", font=("Arial", 10))
        style.map("TCheckbutton", background=[("active", "#0a1128")])

    def create_layout(self):
        main = ttk.Frame(self.root, padding=18)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Gas Sensor Monitoring System", style="Title.TLabel").pack()
        ttk.Label(main, text="Real-time acquisition and visualization for six sensor channels", style="Subtitle.TLabel").pack(pady=(4, 16))

        self.status_var = tk.StringVar(value="Disconnected")
        self.status_label = ttk.Label(main, textvariable=self.status_var, font=("Arial", 12, "bold"))
        self.status_label.pack(pady=(0, 12))
        self.set_status("Disconnected", "#ff4757")

        controls = ttk.Frame(main)
        controls.pack(pady=(0, 14))
        self.connect_btn = ttk.Button(controls, text="Connect serial port", command=self.connect_serial)
        self.connect_btn.pack(side=tk.LEFT, padx=6)
        self.start_btn = ttk.Button(controls, text="Start recording", command=self.start_recording, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn = ttk.Button(controls, text="Stop and save", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        selectors = ttk.Frame(main)
        selectors.pack(fill=tk.X, pady=(0, 12))
        for index in range(1, SENSOR_COUNT + 1):
            variable = tk.BooleanVar(value=True)
            self.sensor_vars[f"sensor{index}"] = variable
            ttk.Checkbutton(selectors, text=f"Sensor {index}", variable=variable, command=self.update_chart_visibility).pack(side=tk.LEFT, padx=12)

        chart_frame = ttk.Frame(main)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.figure = Figure(figsize=(10, 4.5), dpi=100, facecolor="#0a1128")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#1c2b4a")
        self.ax.grid(color="#2b567b", alpha=0.5)
        self.ax.set_xlabel("Sample time", color="#00a8ff")
        self.ax.set_ylabel("Sensor value", color="#00a8ff")
        self.ax.tick_params(axis="x", colors="#87ceeb")
        self.ax.tick_params(axis="y", colors="#87ceeb")
        for index in range(1, SENSOR_COUNT + 1):
            line, = self.ax.plot([], [], color=COLORS[index - 1], label=f"Sensor {index}", linewidth=1.8)
            self.lines[f"sensor{index}"] = line
        legend = self.ax.legend(loc="upper right", facecolor="#1c2b4a", edgecolor="#00a8ff")
        for text in legend.get_texts():
            text.set_color("#d7efff")
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        values = ttk.Frame(main)
        values.pack(fill=tk.X, pady=(0, 10))
        for index in range(1, SENSOR_COUNT + 1):
            panel = ttk.Frame(values)
            panel.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Label(panel, text=f"Sensor {index}").pack()
            value = ttk.Label(panel, text="0.00", style="Value.TLabel")
            value.pack()
            self.data_labels[f"sensor{index}"] = value

        alarm = ttk.Frame(main)
        alarm.pack()
        ttk.Label(alarm, text="Alarm status:").pack(side=tk.LEFT, padx=(0, 5))
        self.alarm_status_value = ttk.Label(alarm, text="Normal", style="Value.TLabel")
        self.alarm_status_value.pack(side=tk.LEFT, padx=(0, 24))
        ttk.Label(alarm, text="Alarm channel:").pack(side=tk.LEFT, padx=(0, 5))
        self.alarm_channel_value = ttk.Label(alarm, text="None", style="Value.TLabel")
        self.alarm_channel_value.pack(side=tk.LEFT)

    def set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    def connect_serial(self):
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            messagebox.showerror("No serial port", "No usable serial port was found.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Select serial port")
        dialog.geometry("480x170")
        dialog.transient(self.root)
        dialog.grab_set()
        selected = tk.StringVar(value=ports[0].device)
        descriptions = "\n".join(f"{port.device} - {port.description}" for port in ports)
        ttk.Label(dialog, text="Available ports:").pack(pady=(16, 4))
        ttk.Combobox(dialog, textvariable=selected, values=[port.device for port in ports], state="readonly").pack(fill=tk.X, padx=30)
        ttk.Label(dialog, text=descriptions, justify=tk.LEFT).pack(pady=8)

        def open_port():
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = serial.Serial(selected.get(), BAUD_RATE, timeout=0.1)
                self.is_connected = True
                self.connect_btn.configure(state=tk.DISABLED)
                self.start_btn.configure(state=tk.NORMAL)
                self.set_status(f"Connected: {selected.get()} ({BAUD_RATE} baud)", "#2ed573")
                dialog.destroy()
            except serial.SerialException as error:
                messagebox.showerror("Connection failed", str(error))

        ttk.Button(dialog, text="Connect", command=open_port).pack(pady=(0, 10))

    def start_recording(self):
        self.recorded_data = []
        self.is_recording = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.set_status("Recording", "#2ed573")

    def stop_recording(self):
        self.is_recording = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.set_status("Connected - recording stopped", "#ffb142")
        self.save_data()

    def save_data(self):
        if not self.recorded_data:
            messagebox.showinfo("No data", "No data was received while recording.")
            return
        filename = f"sensor_data_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], initialfile=filename)
        if not path:
            return
        try:
            with Path(path).open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["timestamp"] + [f"sensor{i}" for i in range(1, SENSOR_COUNT + 1)] + ["alarm_status", "alarm_channel"])
                writer.writerows(self.recorded_data)
            messagebox.showinfo("Saved", f"Data saved to:\n{path}")
        except OSError as error:
            messagebox.showerror("Save failed", str(error))

    def read_serial(self):
        while not self.stop_event.is_set():
            try:
                if self.is_connected and self.ser and self.ser.is_open:
                    chunk = self.ser.read(256)
                    if chunk:
                        self.buffer += chunk.decode("utf-8", errors="replace")
                        frames = self.buffer.split("\n")
                        self.buffer = frames.pop()
                        for frame in frames:
                            self.process_frame(frame.strip())
            except (serial.SerialException, OSError) as error:
                self.is_connected = False
                self.root.after(0, lambda detail=str(error): self.set_status(f"Serial error: {detail}", "#ff4757"))
            time.sleep(0.01)

    def process_frame(self, frame):
        fields = [field.strip() for field in frame.split(",")]
        if len(fields) != FRAME_FIELD_COUNT:
            return
        try:
            sensors = [float(value) for value in fields[:SENSOR_COUNT]]
            alarm_status = int(fields[SENSOR_COUNT])
            alarm_channel = int(fields[SENSOR_COUNT + 1])
        except ValueError:
            return
        for index, value in enumerate(sensors, start=1):
            samples = self.data[f"sensor{index}"]
            samples.append(value)
            if len(samples) > MAX_DATA_POINTS:
                samples.pop(0)
        self.timestamps.append(datetime.now().strftime("%H:%M:%S"))
        if len(self.timestamps) > MAX_DATA_POINTS:
            self.timestamps.pop(0)
        self.alarm_status = alarm_status
        self.alarm_channel = alarm_channel
        if self.is_recording:
            self.recorded_data.append([datetime.now().isoformat()] + sensors + [alarm_status, alarm_channel])
        self.x_index += 1
        if self.x_index % 5 == 0:
            self.root.after(0, self.update_ui)

    def update_ui(self):
        if not self.data["sensor1"]:
            return
        for index in range(1, SENSOR_COUNT + 1):
            key = f"sensor{index}"
            self.data_labels[key].configure(text=f"{self.data[key][-1]:.2f}")
            self.lines[key].set_data(range(len(self.data[key])), self.data[key])
        self.alarm_status_value.configure(text="Alarm" if self.alarm_status else "Normal", foreground="#ff4757" if self.alarm_status else "#2ed573")
        self.alarm_channel_value.configure(text=str(self.alarm_channel) if self.alarm_channel > 0 else "None", foreground="#ff4757" if self.alarm_channel > 0 else "#2ed573")
        point_count = len(self.data["sensor1"])
        step = max(1, point_count // 10)
        ticks = list(range(0, point_count, step))
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([self.timestamps[tick] for tick in ticks], rotation=35, ha="right")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def update_chart_visibility(self):
        for key, line in self.lines.items():
            line.set_visible(self.sensor_vars[key].get())
        self.canvas.draw_idle()

    def close(self):
        self.stop_event.set()
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    SensorMonitor(window)
    window.mainloop()
