"""Backtrack direction estimates for every CSV used as a training sample.

Default mode is stratified cross-validation: a CSV is predicted by a model that
was not fitted on that CSV. This is the meaningful way to inspect training
data backtracking. ``--mode fitted`` is also available for diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from direction_features import ANGLES, FeatureConfig
from train_direction_models import collect_samples, make_classifier


def circular_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 360.0
    return min(difference, 360.0 - difference)


def continuous_angle(classes: np.ndarray, probabilities: np.ndarray) -> float:
    vector = np.sum(probabilities * np.exp(1j * np.deg2rad(classes)))
    return float(np.rad2deg(np.angle(vector)) % 360.0)


def response_order(feature: np.ndarray) -> str:
    ordered_sensors = np.argsort(feature, kind="stable") + 1
    return "->".join(str(sensor) for sensor in ordered_sensors)


def cross_validated_predictions(X: np.ndarray, y: np.ndarray):
    counts = Counter(map(int, y))
    if len(counts) < 2 or min(counts.values()) < 2:
        raise ValueError("At least two experiments at each trained angle are required for cross-validation backtesting.")
    cv = StratifiedKFold(n_splits=min(5, min(counts.values())), shuffle=True, random_state=42)
    predicted = np.empty(len(y), dtype=float)
    probabilities = np.empty((len(y), len(ANGLES)), dtype=float)
    for train_indices, test_indices in cv.split(X, y):
        model = make_classifier().fit(X[train_indices], y[train_indices])
        fold_probabilities = model.predict_proba(X[test_indices])
        # All labels are present in every stratified training fold, but align
        # columns by angle explicitly so the result stays correct if changed.
        aligned = np.zeros((len(test_indices), len(ANGLES)), dtype=float)
        for column, angle in enumerate(model.classes_):
            aligned[:, list(ANGLES).index(int(angle))] = fold_probabilities[:, column]
        probabilities[test_indices] = aligned
        predicted[test_indices] = model.predict(X[test_indices])
    return predicted, probabilities


def fitted_predictions(X: np.ndarray, model_path: Path, method: str):
    artifact = joblib.load(model_path)
    if artifact.get("method") != method:
        raise ValueError(f"Model method is {artifact.get('method')!r}, expected {method!r}.")
    model = artifact["model"]
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(X), len(ANGLES)), dtype=float)
    for column, angle in enumerate(classes):
        aligned[:, list(ANGLES).index(int(angle))] = probabilities[:, column]
    return model.predict(X), aligned


def main():
    parser = argparse.ArgumentParser(description="Backtrack every training CSV with the response-order direction model.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base_dir.parent / "方向传感数据")
    parser.add_argument("--model", type=Path, default=base_dir / "models" / "direction_model_order.joblib")
    parser.add_argument("--method", choices=("order", "sequence_8class"), default="order")
    parser.add_argument("--mode", choices=("cv", "fitted"), default="cv", help="cv is held-out backtesting; fitted is training-set diagnostics.")
    parser.add_argument("--output", type=Path, default=base_dir / "models" / "training_backtest_cv.csv")
    args = parser.parse_args()

    X, y, paths, _ = collect_samples(args.data_dir, args.method, FeatureConfig())
    if args.mode == "cv":
        predicted, probabilities = cross_validated_predictions(X, y)
        mode_description = "cross_validated_held_out"
    else:
        if not args.model.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")
        predicted, probabilities = fitted_predictions(X, args.model, args.method)
        mode_description = "fitted_training_set"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    classes = np.asarray(ANGLES, dtype=float)
    for index, source in enumerate(paths):
        estimate = (float(predicted[index]) if args.method == "sequence_8class"
                    else continuous_angle(classes, probabilities[index]))
        confidence = float(np.max(probabilities[index]))
        rows.append({
            "source_csv": source,
            "true_angle_deg": int(y[index]),
            "response_order": response_order(X[index]),
            "predicted_class_deg": int(predicted[index]),
            "estimated_angle_deg": round(estimate, 3),
            "confidence": round(confidence, 6),
            "circular_error_deg": round(circular_distance(estimate, float(y[index])), 3),
            "mode": mode_description,
        })
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    class_accuracy = float(np.mean(predicted == y))
    mean_error = float(np.mean([row["circular_error_deg"] for row in rows]))
    summary = {
        "mode": mode_description,
        "sample_count": len(rows),
        "class_accuracy": class_accuracy,
        "continuous_circular_mae_deg": mean_error,
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
