"""Four-direction gas localization using response onset order.

The primary evidence is the order in which sensors 1, 3 and 5 cross their
own positive baseline-noise threshold. Sensors 2, 4 and 6 are optional
reference channels. No peak, T50, response amplitude or response interval is
used as a direction feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ANGLES = (0, 90, 180, 270)
PRIMARY = (0, 2, 4)       # S1, S3, S5
REFERENCE = (1, 3, 5)     # S2, S4, S6
SENSOR_COUNT = 6


def load_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV 没有表头")
        fields = {str(name).strip().lower(): name for name in reader.fieldnames if name}
        required = [f"sensor{i}" for i in range(1, 7)]
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"缺少列: {', '.join(missing)}")
        rows = []
        for row in reader:
            try:
                rows.append([float(row[fields[name]]) for name in required])
            except (TypeError, ValueError):
                continue
    if len(rows) < 12:
        raise ValueError("有效采样点少于 12 个")
    return np.asarray(rows, dtype=float)


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    window = int(window) | 1
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.column_stack([
        np.convolve(padded[:, col], kernel, mode="valid")
        for col in range(values.shape[1])
    ])


def first_onset(signal: np.ndarray, baseline: float, noise: float,
                sustained: int = 3) -> float:
    """Find first upward threshold crossing, with no peak/T50 calculation."""
    rise = np.maximum(np.asarray(signal, dtype=float) - baseline, 0.0)
    threshold = max(3.0 * noise, 1e-9)
    sustained = max(1, int(sustained))
    for index in range(1, len(rise) - sustained + 1):
        before = rise[index - 1]
        current = rise[index]
        if current < threshold or current <= before:
            continue
        if np.all(rise[index:index + sustained] >= threshold):
            if current == before:
                return float(index)
            fraction = (threshold - before) / (current - before)
            return float(index - 1 + np.clip(fraction, 0.0, 1.0))
    return float("nan")


def analyze_record(samples: np.ndarray, baseline_fraction: float = 0.15,
                   baseline_samples: int | None = None) -> dict:
    """Remove each baseline and return only onset-order information."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != SENSOR_COUNT:
        raise ValueError("需要 N×6 的传感器数据")
    n = len(values)
    if baseline_samples is None:
        baseline_count = min(max(12, int(n * baseline_fraction)), max(12, n // 3))
    else:
        baseline_count = min(n - 1, max(12, int(baseline_samples)))
    baseline = np.median(values[:baseline_count], axis=0)
    baseline_noise = np.maximum(
        1.4826 * np.median(np.abs(values[:baseline_count] - baseline), axis=0),
        1e-9,
    )
    smoothed = smooth(values, 5)
    onsets = np.asarray([
        first_onset(smoothed[:, sensor], baseline[sensor], baseline_noise[sensor])
        for sensor in range(SENSOR_COUNT)
    ])
    detected = bool(np.any(np.isfinite(onsets)))
    primary_onsets = onsets[list(PRIMARY)]
    reference_onsets = onsets[list(REFERENCE)]
    primary_complete = bool(np.all(np.isfinite(primary_onsets)))
    reference_order = order_text(reference_onsets, REFERENCE)
    primary_order = order_text(primary_onsets, PRIMARY)
    return {
        "baseline": baseline.tolist(),
        "baseline_noise": baseline_noise.tolist(),
        "onsets": onsets.tolist(),
        "gas_detected": detected,
        "primary_complete": primary_complete,
        "primary_order": primary_order,
        "reference_order": reference_order,
        "primary_onsets": primary_onsets.tolist(),
        "reference_onsets": reference_onsets.tolist(),
    }


def order_text(onsets: np.ndarray, sensor_indices: tuple[int, ...]) -> str:
    finite = [(float(value), sensor + 1) for value, sensor in zip(onsets, sensor_indices)
              if np.isfinite(value)]
    finite.sort(key=lambda item: (item[0], item[1]))
    return "->".join(str(sensor) for _, sensor in finite) if finite else "未检测"


def primary_permutation(info: dict) -> tuple[int, ...] | None:
    values = np.asarray(info["primary_onsets"], dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return tuple(int(index) for index in np.argsort(values, kind="stable"))


def feature_vector(info: dict) -> np.ndarray:
    """Encode primary ranks strongly and reference ranks as optional context."""
    primary = np.asarray(info["primary_onsets"], dtype=float)
    reference = np.asarray(info["reference_onsets"], dtype=float)
    if not np.all(np.isfinite(primary)):
        raise ValueError("1、3、5 号传感器没有全部检测到正向响应")
    primary_rank = np.empty(3, dtype=float)
    primary_rank[np.argsort(primary, kind="stable")] = np.arange(3, dtype=float)
    reference_rank = np.where(np.isfinite(reference), reference, np.nan)
    if np.all(np.isfinite(reference_rank)):
        reference_rank = np.empty(3, dtype=float)
        reference_rank[np.argsort(reference, kind="stable")] = np.arange(3, dtype=float)
        reference_present = np.ones(3, dtype=float)
    else:
        present = np.isfinite(reference)
        reference_rank = np.where(present, 0.5, 1.0)
        reference_present = present.astype(float)
    return np.concatenate([primary_rank * 3.0, reference_rank, reference_present * 0.25])


def angle_from_folder(path: Path) -> int:
    text = path.name
    match = re.fullmatch(r"\s*(\d+)\s*度\s*", text)
    if match and int(match.group(1)) in ANGLES:
        return int(match.group(1))
    raise ValueError(f"只接受 0、90、180、270 度目录，当前为: {path.name}")


def collect(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    features, labels, paths, rows = [], [], [], []
    ignored = []
    for path in sorted(data_dir.glob("**/*.csv")):
        try:
            angle = angle_from_folder(path.parent)
        except ValueError:
            ignored.append(str(path))
            continue
        try:
            info = analyze_record(load_csv(path))
            row = {"source_csv": str(path), "angle_deg": angle, **info}
            rows.append(row)
            if info["primary_complete"]:
                features.append(feature_vector(info))
                labels.append(angle)
                paths.append(str(path))
        except (OSError, ValueError) as error:
            rows.append({"source_csv": str(path), "angle_deg": angle,
                         "status": "error", "error": str(error)})
    if ignored:
        print(f"忽略非四方向数据 {len(ignored)} 个（例如 45 度目录）")
    if not features:
        raise RuntimeError("没有找到 1、3、5 号均有响应的有效样本")
    return np.asarray(features), np.asarray(labels), paths, rows


def classifier() -> Pipeline:
    return Pipeline([
        ("classifier", SVC(C=2.0, kernel="rbf", gamma="scale",
                           probability=True, class_weight="balanced",
                           random_state=42)),
    ])


def circular_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    difference = np.abs(np.asarray(predicted) - np.asarray(actual))
    return np.minimum(difference, 360.0 - difference)


def write_quality(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_csv", "angle_deg", "status", "gas_detected",
              "primary_complete", "primary_order", "reference_order",
              "onsets", "error"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            info = dict(row)
            info["status"] = "good" if info.get("primary_complete") else (
                "gas_but_primary_incomplete" if info.get("gas_detected") else "no_confirmed_gas")
            info["gas_detected"] = str(bool(info.get("gas_detected", False)))
            info["primary_complete"] = str(bool(info.get("primary_complete", False)))
            info["onsets"] = ",".join("" if not np.isfinite(value) else f"{value:.2f}"
                                       for value in info.get("onsets", []))
            writer.writerow({field: info.get(field, "") for field in fields})


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    X, y, paths, rows = collect(data_dir)
    write_quality(rows, output_dir / "quality_20261016.csv")
    counts = {int(angle): int(np.sum(y == angle)) for angle in ANGLES}
    if any(count < 2 for count in counts.values()):
        raise RuntimeError(f"每个方向至少需要 2 个完整主通道样本，当前数量: {counts}")
    model = classifier()
    loo = LeaveOneOut()
    predicted = cross_val_predict(model, X, y, cv=loo)
    metrics = {
        "angles": list(ANGLES),
        "primary_sensors": [1, 3, 5],
        "reference_sensors": [2, 4, 6],
        "sample_count": int(len(y)),
        "class_counts": counts,
        "leave_one_out_accuracy": float(accuracy_score(y, predicted)),
        "leave_one_out_circular_mae_deg": float(np.mean(circular_error(predicted, y))),
        "confusion_matrix": confusion_matrix(y, predicted, labels=list(ANGLES)).tolist(),
        "note": "方向特征只包含 1/3/5 顺序名次；2/4/6 仅作低权重参考。",
    }
    model.fit(X, y)
    artifact = {
        "model": model,
        "angles": list(ANGLES),
        "primary_sensors": [1, 3, 5],
        "reference_sensors": [2, 4, 6],
        "threshold_rule": "first upward crossing above max(3*MAD baseline noise, epsilon), sustained 3 samples",
        "source_data_dir": str(data_dir),
        "feature_names": ["response_rank_s1_x3", "response_rank_s3_x3", "response_rank_s5_x3", "reference_rank_s2", "reference_rank_s4", "reference_rank_s6", "reference_present_s2", "reference_present_s4", "reference_present_s6"],
        "metrics": metrics,
        "sample_paths": paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "direction_order_model_20261016.joblib"
    joblib.dump(artifact, model_path)
    (output_dir / "metrics_20261016.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    backtest_path = output_dir / "backtest_20261016.csv"
    backtest_rows = []
    for source, actual, pred, info in zip(paths, y, predicted, [r for r in rows if r.get("primary_complete")]):
        backtest_rows.append({"source_csv": source, "true_angle_deg": int(actual),
                              "predicted_angle_deg": int(pred),
                              "primary_order": info["primary_order"],
                              "reference_order": info["reference_order"],
                              "circular_error_deg": float(circular_error(np.asarray([pred]), np.asarray([actual]))[0])})
    with backtest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(backtest_rows[0]))
        writer.writeheader()
        writer.writerows(backtest_rows)
    print(json.dumps({"model": str(model_path), **metrics}, ensure_ascii=False, indent=2))


def predict_file(model_path: Path, csv_path: Path) -> dict:
    artifact = joblib.load(model_path)
    info = analyze_record(load_csv(csv_path))
    result = {"csv": str(csv_path), "gas_detected": info["gas_detected"],
              "primary_order": info["primary_order"], "reference_order": info["reference_order"]}
    if not info["gas_detected"]:
        result.update({"status": "no_confirmed_gas", "message": "未检测到超过基线波动阈值的持续向上响应"})
        return result
    if not info["primary_complete"]:
        result.update({"status": "primary_incomplete", "message": "已检测到气体，但 1、3、5 号没有全部形成可靠响应，暂不判断方向"})
        return result
    probabilities = artifact["model"].predict_proba([feature_vector(info)])[0]
    classes = np.asarray(artifact["model"].classes_, dtype=float)
    best = int(np.argmax(probabilities))
    result.update({"status": "ok", "angle_deg": int(classes[best]),
                   "confidence": float(probabilities[best]),
                   "probabilities": {str(int(angle)): float(prob)
                                     for angle, prob in zip(classes, probabilities)}})
    return result


def parse_serial_line(line: str) -> list[float] | None:
    parts = [part.strip() for part in line.replace(";", ",").split(",")]
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values[:6] if len(values) >= 6 else None


def live(args: argparse.Namespace) -> None:
    import serial
    port = serial.Serial(args.port, args.baud, timeout=1.0)
    frames: list[list[float]] = []
    baseline = None
    noise = None
    trigger_hits = 0
    print(f"正在监听 {args.port}。先采集 {args.baseline_frames} 帧基线。")
    try:
        while True:
            line = port.readline().decode("utf-8", errors="ignore").strip()
            values = parse_serial_line(line)
            if values is None:
                continue
            frames.append(values)
            if len(frames) < args.baseline_frames:
                continue
            if len(frames) == args.baseline_frames:
                print("基线已建立，等待持续向上响应。")
            if baseline is None:
                base = np.asarray(frames[:args.baseline_frames], dtype=float)
                baseline = np.median(base, axis=0)
                noise = np.maximum(1.4826 * np.median(np.abs(base - baseline), axis=0), 1e-6)
            positive = np.asarray(values, dtype=float) - baseline
            threshold = np.maximum(5.0 * noise, 0.10 * np.maximum(np.abs(baseline), 1.0))
            if np.any(positive >= threshold):
                trigger_hits += 1
            else:
                trigger_hits = 0
            if trigger_hits < 5:
                continue
            print("已明确超过基线波动阈值，并连续保持 5 帧，开始捕获响应顺序……")
            for _ in range(args.capture_frames):
                line = port.readline().decode("utf-8", errors="ignore").strip()
                values = parse_serial_line(line)
                if values is not None:
                    frames.append(values)
            artifact = joblib.load(args.model)
            final_info = analyze_record(np.asarray(frames, dtype=float), baseline_samples=args.baseline_frames)
            if final_info["primary_complete"]:
                probs = artifact["model"].predict_proba([feature_vector(final_info)])[0]
                classes = artifact["model"].classes_
                index = int(np.argmax(probs))
                print(json.dumps({"status": "ok", "angle_deg": int(classes[index]),
                                  "confidence": float(probs[index]),
                                  "primary_order": final_info["primary_order"],
                                  "reference_order": final_info["reference_order"]}, ensure_ascii=False))
            else:
                print(json.dumps({"status": "primary_incomplete",
                                  "primary_order": final_info["primary_order"],
                                  "reference_order": final_info["reference_order"]}, ensure_ascii=False))
            frames = []
            baseline = None
            noise = None
            trigger_hits = 0
            print("本次判断结束，重新采集基线。")
    finally:
        port.close()


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="2026-10-16 四方向响应顺序定位")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="训练并生成质量表、回溯表和模型")
    train_parser.add_argument("--data-dir", type=Path, default=base / "方向传感数据_20261016")
    train_parser.add_argument("--output-dir", type=Path, default=base / "models")
    pred_parser = sub.add_parser("predict", help="判断一个 CSV 的气体来源方向")
    pred_parser.add_argument("--model", type=Path, default=base / "models" / "direction_order_model_20261016.joblib")
    pred_parser.add_argument("--csv", type=Path, required=True)
    live_parser = sub.add_parser("live", help="串口实时判断，检测到气体后只判断一次")
    live_parser.add_argument("--model", type=Path, default=base / "models" / "direction_order_model_20261016.joblib")
    live_parser.add_argument("--port", required=True)
    live_parser.add_argument("--baud", type=int, default=9600)
    live_parser.add_argument("--baseline-frames", type=int, default=60)
    live_parser.add_argument("--capture-frames", type=int, default=60)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        print(json.dumps(predict_file(args.model, args.csv), ensure_ascii=False, indent=2))
    else:
        live(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"运行失败: {error}", file=sys.stderr)
        raise
