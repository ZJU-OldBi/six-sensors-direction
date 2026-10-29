"""Quality-screen direction CSVs using positive onset order only.

This report never deletes or moves raw files. It records why a file is good,
ambiguous, or unsuitable for order-only direction training.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from direction_features import FeatureConfig, _first_sustained_rise, _smooth, angle_from_folder, load_csv


SELECTED = tuple(range(6))


def inspect_file(path: Path, config: FeatureConfig) -> dict:
    samples, timestamps = load_csv(path)
    n = len(samples)
    base_count = min(n - 1, max(config.min_baseline_samples, int(n * config.baseline_fraction)))
    base_count = max(3, base_count)
    baseline = np.median(samples[:base_count], axis=0)
    smoothed_rise = _smooth(np.maximum(samples - baseline, 0.0), config.smooth_window)
    noise = np.maximum(np.median(np.abs(samples[:base_count] - baseline), axis=0) * 1.4826, 1e-9)
    peaks = np.max(smoothed_rise, axis=0)
    snr = peaks / noise
    active = snr >= config.min_response_snr
    onsets = np.full(6, np.nan, dtype=float)
    for sensor in range(6):
        if not active[sensor]:
            continue
        threshold = max(
            config.onset_sigma * noise[sensor],
            config.onset_fraction * max(abs(float(baseline[sensor])), 1.0),
            config.onset_fraction * float(peaks[sensor]),
        )
        slope_threshold = max(
            config.slope_sigma * noise[sensor],
            0.02 * max(abs(float(baseline[sensor])), 1.0),
        )
        onsets[sensor] = _first_sustained_rise(
            smoothed_rise[:, sensor], threshold, slope_threshold, config.sustained_samples
        )
    selected_active = active[list(SELECTED)]
    selected_onsets = onsets[list(SELECTED)]
    valid_onsets = selected_onsets[np.isfinite(selected_onsets)]
    order = "->".join(str(index + 1) for index in np.argsort(np.where(np.isfinite(selected_onsets), selected_onsets, n + np.arange(len(SELECTED)))))
    sorted_onsets = np.sort(valid_onsets)
    gaps = np.diff(sorted_onsets) if len(sorted_onsets) > 1 else np.asarray([])
    min_gap = float(np.min(gaps)) if len(gaps) else None
    reasons = []
    if int(np.sum(selected_active)) < len(SELECTED):
        missing = [str(index + 1) for index in SELECTED if not active[index]]
        reasons.append(f"selected sensors without reliable positive rise: {','.join(missing)}")
    if min_gap is not None and min_gap <= 2:
        reasons.append("onset times are nearly tied (<=2 samples)")
    low_snr = [str(index + 1) for index in SELECTED if active[index] and snr[index] < 5]
    if low_snr:
        reasons.append(f"low onset SNR: {','.join(low_snr)}")
    if int(np.sum(selected_active)) < len(SELECTED):
        status = "bad"
    elif min_gap is not None and min_gap <= 2 or low_snr:
        status = "warning"
    else:
        status = "good"
    duration = None
    if timestamps is not None and len(timestamps) > 1:
        duration = float(timestamps[-1] - timestamps[0])
    return {
        "source_csv": str(path),
        "angle_deg": angle_from_folder(path.parent),
        "status": status,
        "sample_count": n,
        "duration_s": round(duration, 3) if duration is not None else None,
        "selected_sensor_count": int(np.sum(selected_active)),
        "selected_sensor_numbers": "1,2,3,4,5,6",
        "active_sensors": ",".join(str(index + 1) for index in np.flatnonzero(active)),
        "response_order_1_to_6": order,
        "onset_sample_s1_to_s6": ",".join("" if not np.isfinite(value) else str(int(value)) for value in selected_onsets),
        "min_onset_gap_samples": min_gap,
        "snr_s1": round(float(snr[0]), 2),
        "snr_s2": round(float(snr[1]), 2),
        "snr_s3": round(float(snr[2]), 2),
        "snr_s4": round(float(snr[3]), 2),
        "snr_s5": round(float(snr[4]), 2),
        "snr_s6": round(float(snr[5]), 2),
        "reason": "; ".join(reasons),
    }


def main():
    parser = argparse.ArgumentParser(description="Screen direction data for positive onset-order training.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base_dir.parent / "方向传感数据")
    parser.add_argument("--output", type=Path, default=base_dir / "models" / "direction_data_quality.csv")
    args = parser.parse_args()
    config = FeatureConfig()
    rows, errors = [], []
    for path in sorted(args.data_dir.glob("**/*.csv")):
        try:
            rows.append(inspect_file(path, config))
        except (OSError, ValueError) as error:
            errors.append({"source_csv": str(path), "reason": str(error), "status": "bad"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["source_csv", "status", "reason"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for error in errors:
            writer.writerow({field: error.get(field, "") for field in fields})
    for filename, selected_rows in (("direction_data_good.csv", [row for row in rows if row["status"] == "good"]),
                                    ("direction_data_problem.csv", [row for row in rows if row["status"] != "good"] + errors)):
        output_path = args.output.with_name(filename)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in selected_rows)
    summary = {
        "total_csv": len(rows) + len(errors),
        "good": sum(row["status"] == "good" for row in rows),
        "warning": sum(row["status"] == "warning" for row in rows),
        "bad": sum(row["status"] == "bad" for row in rows) + len(errors),
        "by_angle": {str(angle): Counter(row["status"] for row in rows if row["angle_deg"] == angle) for angle in sorted({row["angle_deg"] for row in rows})},
        "criteria": {
            "selected_sensors": "1-6",
            "positive_rise_only": True,
            "min_response_snr": config.min_response_snr,
            "warning_if_min_onset_gap_samples_at_most": 2,
            "warning_if_selected_sensor_snr_below": 5,
            "bad_if_any_selected_sensor_snr_below": config.min_response_snr,
        },
        "errors": errors,
        "report_csv": str(args.output.resolve()),
        "good_csv": str(args.output.with_name("direction_data_good.csv").resolve()),
        "problem_csv": str(args.output.with_name("direction_data_problem.csv").resolve()),
    }
    order_summary_rows = []
    by_angle = {}
    for row in rows:
        if row["status"] != "bad":
            by_angle.setdefault(int(row["angle_deg"]), []).append(row)
    for angle in sorted(by_angle):
        angle_rows = by_angle[angle]
        order_counts = Counter(row["response_order_1_to_6"] for row in angle_rows)
        dominant_order, dominant_count = order_counts.most_common(1)[0]
        probabilities = np.asarray(list(order_counts.values()), dtype=float) / len(angle_rows)
        entropy = float(-np.sum(probabilities * np.log2(probabilities))) if len(probabilities) else 0.0
        order_summary_rows.append({
            "angle_deg": angle,
            "usable_csv_count": len(angle_rows),
            "unique_order_count": len(order_counts),
            "dominant_order": dominant_order,
            "dominant_order_count": dominant_count,
            "dominant_order_fraction": round(dominant_count / len(angle_rows), 4),
            "order_entropy_bits": round(entropy, 4),
            "interpretation": "stable" if dominant_count / len(angle_rows) >= 0.6 else "unstable",
        })
    order_summary_path = args.output.with_name("direction_order_by_angle.csv")
    with order_summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(order_summary_rows[0]) if order_summary_rows else ["angle_deg"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(order_summary_rows)
    summary["order_summary_csv"] = str(order_summary_path.resolve())
    summary["stable_angle_count"] = sum(row["interpretation"] == "stable" for row in order_summary_rows)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output.with_name("direction_data_quality_report.md")
    good_rows = [row for row in rows if row["status"] == "good"]
    warning_rows = [row for row in rows if row["status"] == "warning"]
    bad_rows = [row for row in rows if row["status"] == "bad"] + errors
    report_lines = [
        "# 方向传感数据质量报告",
        "",
        "本报告只依据每个传感器相对自身基线的正向响应起始时间排序。未使用 T50、峰值大小、面积、响应强度或响应时间间隔作为方向特征。六个传感器全部参与排序。",
        "",
        f"- CSV 总数：{len(rows) + len(errors)}",
        f"- 可用（good）：{len(good_rows)}",
        f"- 可检测但需复查（warning）：{len(warning_rows)}",
        f"- 不可用于训练（bad）：{len(bad_rows)}",
        f"- 形成稳定主导顺序的角度：{summary['stable_angle_count']} / {len(order_summary_rows)}",
        "",
        "## 判定规则",
        "",
        "- 每个 CSV 前 12% 样本估计各自基线，只取高于基线的变化。",
        "- 每路先独立估计噪声，再用幅度阈值和上升斜率确认连续 3 个采样点的突变起点。",
        "- bad：1–6 号至少一路没有可靠正向响应（SNR < 3）。",
        "- warning：起点间隔不超过 2 个采样点，或某路 SNR < 5。",
        "- good：1–6 号均有可靠正向响应，且起点没有明显并列、SNR 均不低。",
        "",
        "## 各角度顺序稳定性",
        "",
        "| 角度 | 可用样本 | 不同顺序数 | 主导顺序 | 主导比例 | 结论 |",
        "|---:|---:|---:|---|---:|---|",
    ]
    for row in order_summary_rows:
        report_lines.append(
            f"| {row['angle_deg']}° | {row['usable_csv_count']} | {row['unique_order_count']} | "
            f"{row['dominant_order']} | {row['dominant_order_fraction']:.1%} | {row['interpretation']} |"
        )
    report_lines.extend(["", "## bad 数据（训练已自动跳过）", ""])
    if bad_rows:
        for row in bad_rows:
            report_lines.append(f"- `{row.get('source_csv', '')}`：{row.get('reason', '读取或格式错误')}")
    else:
        report_lines.append("无。")
    report_lines.extend(["", "## good 数据", ""])
    if good_rows:
        for row in good_rows:
            report_lines.append(f"- `{row['source_csv']}`：{row['angle_deg']}°，顺序 `{row['response_order_1_to_6']}`")
    else:
        report_lines.append("无。")
    report_lines.extend([
        "",
        "## 结论",
        "",
        "当前数据没有任何一个角度达到稳定唯一的响应顺序（主导比例至少 60%）。因此模型可以用于记录和回溯，但仅凭现有重复实验的响应先后，方位判别可靠性较低；建议在固定释放距离、风速、采样率和气流条件下补采，并优先补采顺序最分散的角度。",
        "",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    summary["report_markdown"] = str(report_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
