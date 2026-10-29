"""Train direction classifiers from repeated CSV experiments.

Examples:
  python train_direction_models.py
  python train_direction_models.py --data-dir "../方向传感数据" --output-dir models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from direction_features import ANGLES, FeatureConfig, angle_from_folder, extract_features, load_csv


METHODS = ("order", "sequence_8class", "first_response", "position_order")


def collect_samples(data_dir: Path, method: str, config: FeatureConfig):
    features, labels, paths, diagnostics = [], [], [], []
    errors = []
    for csv_path in sorted(data_dir.glob("**/*.csv")):
        try:
            label = angle_from_folder(csv_path.parent)
            samples, timestamps = load_csv(csv_path)
            feature, info = extract_features(samples, timestamps, method, config)
            features.append(feature)
            labels.append(label)
            paths.append(str(csv_path))
            diagnostics.append(info)
        except (OSError, ValueError) as error:
            errors.append(f"{csv_path}: {error}")
    if errors:
        print(f"Skipped {len(errors)} file(s):")
        for error in errors[:10]:
            print(f"  - {error}")
    if not features:
        raise RuntimeError(f"No valid CSV samples found under {data_dir}")
    return np.asarray(features), np.asarray(labels), paths, diagnostics


def circular_error(predicted, actual):
    delta = np.abs(np.asarray(predicted) - np.asarray(actual))
    return np.minimum(delta, 360 - delta)


def make_classifier():
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", SVC(C=3.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)),
    ])


def train_one(data_dir: Path, output_dir: Path, method: str, config: FeatureConfig):
    X, y, paths, diagnostics = collect_samples(data_dir, method, config)
    counts = {int(angle): int(np.sum(y == angle)) for angle in ANGLES if np.sum(y == angle)}
    if method == "sequence_8class" and set(counts) != set(ANGLES):
        missing = sorted(set(ANGLES) - set(counts))
        raise ValueError(f"sequence_8class requires all eight angle classes; missing {missing}")
    classifier = make_classifier()
    metrics = {"sample_count": int(len(y)), "class_counts": counts}
    min_count = min(counts.values()) if counts else 0
    if len(counts) >= 2 and min_count >= 2:
        splits = min(5, min_count)
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        predicted = cross_val_predict(classifier, X, y, cv=cv)
        metrics["cv_folds"] = splits
        metrics["cv_accuracy"] = float(accuracy_score(y, predicted))
        metrics["cv_circular_mae_deg"] = float(np.mean(circular_error(predicted, y)))
        metrics["cv_confusion_matrix"] = confusion_matrix(y, predicted, labels=list(ANGLES)).tolist()
    else:
        metrics["cv_folds"] = 0
        metrics["cv_note"] = "At least two samples per angle are needed for cross-validation."
    classifier.fit(X, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": classifier,
        "method": method,
        "angles": list(ANGLES),
        "feature_names": diagnostics[0]["feature_names"],
        "feature_config": config.__dict__,
        "sensor_count": 6,
        "metrics": metrics,
        "source_data_dir": str(data_dir.resolve()),
        "sample_paths": paths,
        "direction_rule": "first responding sensor azimuth is the physical source direction; classifier uses the complete positive-rise response order.",
    }
    if method == "first_response":
        profiles = {}
        for angle in sorted(set(y.astype(int))):
            ranks = np.asarray([info["response_rank"] for label, info in zip(y, diagnostics) if int(label) == angle], dtype=float)
            profiles[str(angle)] = {
                "mean_response_rank": np.mean(ranks, axis=0).tolist(),
                "sample_count": int(len(ranks)),
            }
        artifact["validation_profiles"] = profiles
    model_path = output_dir / f"direction_model_{method}.joblib"
    metrics_path = output_dir / f"direction_metrics_{method}.json"
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{method:8s}] samples={len(y):3d}, classes={len(counts)} -> {model_path}")
    if "cv_accuracy" in metrics:
        print(f"           CV accuracy={metrics['cv_accuracy']:.3f}, circular MAE={metrics['cv_circular_mae_deg']:.1f} deg")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train six-sensor gas direction models.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "方向传感数据")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all", help="Train one model or all integrated model variants.")
    args = parser.parse_args()
    config = FeatureConfig()
    methods = METHODS if args.method == "all" else (args.method,)
    print(f"Data: {args.data_dir.resolve()}")
    for method in methods:
        train_one(args.data_dir, args.output_dir, method, config)
    print("Training complete. Use direction_inference.py --model <joblib> to load a model.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise
