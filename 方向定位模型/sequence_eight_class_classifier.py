"""Classify a complete six-sensor response sequence into eight directions.

The only valid output classes are 0, 45, 90, 135, 180, 225, 270, and 315
degrees. The classifier intentionally never returns an interpolated angle.

Examples:
  python sequence_eight_class_classifier.py train
  python sequence_eight_class_classifier.py predict --sequence 1,4,3,2,5,6
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from direction_features import ANGLES, FeatureConfig
from train_direction_models import collect_samples, make_classifier


def sequence_to_ranks(sequence_text: str) -> np.ndarray:
    try:
        sequence = [int(value.strip()) for value in sequence_text.replace("->", ",").split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("Sequence must contain sensor numbers, e.g. 1,4,3,2,5,6") from error
    if sorted(sequence) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Sequence must contain each sensor number 1 through 6 exactly once")
    ranks = np.empty(6, dtype=float)
    for rank, sensor in enumerate(sequence):
        ranks[sensor - 1] = rank
    return ranks


def load_artifact(model_path: Path):
    artifact = joblib.load(model_path)
    if artifact.get("method") != "sequence_8class":
        raise ValueError(f"Not an eight-class sequence model: {model_path}")
    return artifact


def predict_sequence(artifact: dict, sequence_text: str) -> dict:
    ranks = sequence_to_ranks(sequence_text)
    model = artifact["model"]
    probabilities = model.predict_proba([ranks])[0]
    classes = np.asarray(model.classes_, dtype=int)
    best = int(np.argmax(probabilities))
    return {
        "response_sequence": [int(sensor) for sensor in np.argsort(ranks, kind="stable") + 1],
        "response_ranks_by_sensor": ranks.astype(int).tolist(),
        "predicted_angle_deg": int(classes[best]),
        "confidence": float(probabilities[best]),
        "class_probabilities": {str(int(angle)): float(probability) for angle, probability in zip(classes, probabilities)},
    }


def train(data_dir: Path, output_dir: Path):
    X, y, paths, diagnostics = collect_samples(data_dir, "sequence_8class", FeatureConfig())
    counts = {int(angle): int(np.sum(y == angle)) for angle in ANGLES}
    classifier = make_classifier()
    if len(counts) != len(ANGLES):
        missing = sorted(set(ANGLES) - set(counts))
        raise ValueError(f"All eight direction folders need at least one valid CSV; missing: {missing}")
    if min(counts.values()) < 2:
        raise ValueError("At least two valid CSV experiments are required in every direction folder for cross-validation.")
    splits = min(5, min(counts.values()))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    predicted = cross_val_predict(classifier, X, y, cv=cv)
    metrics = {
        "sample_count": int(len(y)),
        "class_counts": counts,
        "cv_folds": splits,
        "cv_accuracy": float(accuracy_score(y, predicted)),
        "cv_confusion_matrix": confusion_matrix(y, predicted, labels=list(ANGLES)).tolist(),
        "classes_deg": list(ANGLES),
    }
    classifier.fit(X, y)
    artifact = {
        "model": classifier,
        "method": "sequence_8class",
        "model_name": "eight_direction_response_sequence_classifier",
        "angles": list(ANGLES),
        "feature_names": diagnostics[0]["feature_names"],
        "feature_config": FeatureConfig().__dict__,
        "sensor_count": 6,
        "metrics": metrics,
        "source_data_dir": str(data_dir.resolve()),
        "sample_paths": paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "direction_model_sequence_8class.joblib"
    joblib.dump(artifact, model_path)
    (output_dir / "direction_metrics_sequence_8class.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_backtest(output_dir / "sequence_8class_backtest_cv.csv", paths, y, X, predicted)
    print(f"Trained eight-class sequence model from {len(y)} experiments: {model_path}")
    print(f"Cross-validation accuracy: {metrics['cv_accuracy']:.3f}")


def _write_backtest(output_path: Path, paths: list[str], labels: np.ndarray, features: np.ndarray, predicted: np.ndarray):
    """Write held-out predictions so every experiment can be audited after training."""
    rows = []
    for source, actual, ranks, estimate in zip(paths, labels, features, predicted):
        order = (np.argsort(ranks, kind="stable") + 1).astype(int).tolist()
        rows.append({
            "source_csv": source,
            "true_angle_deg": int(actual),
            "response_sequence": "->".join(map(str, order)),
            "predicted_angle_deg": int(estimate),
            "correct": int(estimate) == int(actual),
        })
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_csv", "true_angle_deg", "response_sequence", "predicted_angle_deg", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train or use a strictly eight-class response-sequence direction classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, default=base_dir.parent / "方向传感数据")
    train_parser.add_argument("--output-dir", type=Path, default=base_dir / "models")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, default=base_dir / "models" / "direction_model_sequence_8class.joblib")
    predict_parser.add_argument("--sequence", required=True, help="Six sensor IDs in response order, e.g. 1,4,3,2,5,6")
    args = parser.parse_args()
    if args.command == "train":
        train(args.data_dir, args.output_dir)
    else:
        result = predict_sequence(load_artifact(args.model), args.sequence)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
