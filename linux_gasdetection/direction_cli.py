"""Headless gas-direction detection for Linux (no GUI required).

Use this on a robot, over SSH, or in a systemd service.  It shares the exact
same detection rule with the GUI (direction_core.DetectionSession).

Modes
-----
live      ``--port /dev/ttyUSB0``            continuous detection; re-arms after each event
once      ``--port /dev/ttyUSB0 --once``     exit after the first locked direction
replay    ``--replay FILE [--speed N]``      feed a recorded CSV and print the direction
self-test ``--self-test --data-dir DIR``     classify every CSV under an angle folder tree
ports     ``--list-ports``                   list detected serial devices and exit

The machine-readable result is one JSON object per detected event on stdout
(or sent to UDP / another serial port with ``--robot-output``); progress and
status messages go to stderr.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

from direction_core import (
    SENSOR_COUNT,
    DetectionSession,
    DirectionEstimator,
    NullSink,
    ReplayReader,
    SerialFrameReader,
    create_sink,
    list_serial_ports,
)
from direction_features import angle_from_folder, load_csv


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


def angle_text(angle: float, method: str) -> str:
    return f"{int(angle)}" if method == "sequence_8class" else f"{angle:.1f}"


def run_live(estimator: DirectionEstimator, args) -> int:
    result_event = threading.Event()
    latest = {}

    def on_result(angle, confidence, diagnostics):
        latest["result"] = (angle, confidence, diagnostics)
        result_event.set()

    def on_status(text):
        print(f"[status] {text}", file=sys.stderr, flush=True)

    sink = create_sink(args.robot_output, args.robot_baud)
    session = DetectionSession(estimator, sink, on_result=on_result, on_status=on_status)
    stop_event = threading.Event()
    reader = SerialFrameReader(
        args.port, args.baud, session, stop_event,
        on_error=lambda message: print(f"[error] {message}", file=sys.stderr, flush=True),
    )
    reader.start()
    print(f"[info] reading {args.port} at {args.baud} baud; press Ctrl+C to stop", file=sys.stderr, flush=True)
    try:
        while True:
            if result_event.wait(timeout=0.2):
                angle, confidence, diagnostics = latest["result"]
                print(f"[result] direction {angle_text(angle, estimator.method)} deg, confidence {confidence:.1%}, order: {diagnostics['response_order_all_text']}",
                      file=sys.stderr, flush=True)
                if args.once:
                    return 0
                result_event.clear()
                # Re-arm automatically: wait for the gas to clear, then start
                # a fresh quiet baseline for the next detection.
                time.sleep(max(0.0, args.cooldown))
                session.reset()
    except KeyboardInterrupt:
        print("\n[info] stopped by user", file=sys.stderr, flush=True)
        return 0
    finally:
        stop_event.set()
        sink.close()


def run_replay(estimator: DirectionEstimator, args) -> int:
    result_event = threading.Event()
    latest = {}
    done = threading.Event()

    def on_result(angle, confidence, diagnostics):
        latest["result"] = (angle, confidence, diagnostics)
        result_event.set()

    def on_status(text):
        print(f"[status] {text}", file=sys.stderr, flush=True)

    sink = create_sink(args.robot_output, args.robot_baud)
    session = DetectionSession(estimator, sink, on_result=on_result, on_status=on_status)
    reader = ReplayReader(
        args.replay, session, speed=args.speed,
        on_error=lambda message: print(f"[error] {message}", file=sys.stderr, flush=True),
        on_end=done.set,
    )
    reader.start()
    done.wait()
    reader.join(timeout=0.5)
    if "result" not in latest and session.state != "locked":
        # The recording ended while the observation window was still open:
        # judge with every response collected until end-of-file.
        session.force_finish()
    if "result" not in latest:
        print("[result] no gas event was detected in this recording", file=sys.stderr, flush=True)
        sink.close()
        return 1
    angle, confidence, diagnostics = latest["result"]
    print(f"[result] direction {angle_text(angle, estimator.method)} deg, confidence {confidence:.1%}, order: {diagnostics['response_order_all_text']}",
          file=sys.stderr, flush=True)
    sink.close()
    return 0


def circular_error(predicted: float, actual: float) -> float:
    difference = abs(float(predicted) - float(actual)) % 360.0
    return min(difference, 360.0 - difference)


def run_self_test(estimator: DirectionEstimator, args) -> int:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")
    rows = []
    for csv_path in sorted(data_dir.glob("**/*.csv")):
        try:
            label = angle_from_folder(csv_path.parent)
            samples, timestamps = load_csv(csv_path)
            angle, confidence, diagnostics = estimator.predict(samples, timestamps)
            rows.append((csv_path, label, angle, confidence, None))
        except (OSError, ValueError) as error:
            rows.append((csv_path, None, None, None, str(error)))
    if not rows:
        raise SystemExit(f"No CSV files found under {data_dir}")

    print(f"Self-test with model: {estimator.model_path} (method: {estimator.method})")
    print(f"{'file':<52} {'true':>5} {'est':>7} {'err':>6} {'conf':>6}  order")
    correct = 0
    errors = []
    for csv_path, label, angle, confidence, problem in rows:
        if problem is not None:
            print(f"{csv_path.name:<52} {'-':>5} {'-':>7} {'-':>6} {'-':>6}  {problem}")
            continue
        error = circular_error(angle, label)
        errors.append(error)
        ok = error <= 22.5
        correct += int(ok)
        order = "->".join(f"S{sensor}" for sensor in diagnostics.get("response_sequence", []))
        print(f"{csv_path.name:<52} {label:>5} {angle:>7.1f} {error:>6.1f} {confidence:>6.1%}  {order}")
    usable = len(errors)
    print("-" * 100)
    print(f"files: {len(rows)}, classified: {usable}, within 22.5 deg: {correct}/{usable}"
          + (f" ({correct / usable:.1%})" if usable else ""))
    if errors:
        print(f"mean circular error: {np.mean(errors):.1f} deg")
    return 0 if usable and correct == usable else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless six-sensor gas direction detection (Linux).")
    parser.add_argument("--model", type=Path, default=None, help="Model file; default models/direction_model_first_response.joblib")
    parser.add_argument("--port", default=None, help="Sensor serial device, e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--robot-output", default=None, help="Optional udp:HOST:PORT or serial:DEVICE JSON output")
    parser.add_argument("--robot-baud", type=int, default=115200)
    parser.add_argument("--observe-seconds", type=float, default=None, help="Observation window length after the first response (default 5 s).")
    parser.add_argument("--primary-sensors", default=None, help="Optional trigger sensor list; any one starts the window, default 1,2,3,4,5,6.")
    parser.add_argument("--first-response-weights", default=None, help="Five comma-separated correction weights for first_response.")
    parser.add_argument("--once", action="store_true", help="Exit after the first locked direction (live mode).")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds between two detections before the baseline is re-collected.")
    parser.add_argument("--replay", type=Path, default=None, help="Replay a recorded CSV instead of reading a live port.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed factor; 1.0 = real time, 0 = as fast as possible.")
    parser.add_argument("--self-test", action="store_true", help="Classify every CSV under --data-dir and report accuracy.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "方向传感数据",
                        help="Angle folder tree used by --self-test.")
    parser.add_argument("--list-ports", action="store_true", help="List detected serial devices and exit.")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports detected (check the cable and permissions).")
            return 1
        for port in ports:
            print(f"{port.device:<24} {port.description}")
        return 0

    estimator = build_estimator(args)
    if args.self_test:
        return run_self_test(estimator, args)
    if args.replay is not None:
        return run_replay(estimator, args)
    if args.port:
        return run_live(estimator, args)
    raise SystemExit("Choose a mode: --port DEVICE, --replay FILE, --self-test or --list-ports (see --help).")


if __name__ == "__main__":
    raise SystemExit(main())
