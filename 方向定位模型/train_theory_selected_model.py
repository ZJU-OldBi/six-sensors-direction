"""Select the two most theory-consistent experiments per angle and train.

The selection compares each extracted response order with the order predicted
by the regular-hexagon sensor geometry. It never moves or edits raw CSV data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from direction_features import ANGLES, FeatureConfig, SENSOR_ANGLES_DEG, angle_from_folder, extract_features, load_csv
from train_direction_models import circular_error, make_classifier


def circular_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 360.0
    return min(difference, 360.0 - difference)


def theory_distances(angle: int) -> np.ndarray:
    return np.asarray([circular_distance(angle, sensor_angle) for sensor_angle in SENSOR_ANGLES_DEG], dtype=float)


def theory_order(angle: int) -> list[int]:
    distances = theory_distances(angle)
    return (np.argsort(distances, kind="stable") + 1).astype(int).tolist()


def order_consistency(observed_ranks: np.ndarray, angle: int) -> tuple[float, bool, float]:
    """Score a recorded order against geometry, ignoring theoretical ties."""
    distances = theory_distances(angle)
    agreements = 0
    comparisons = 0
    for left in range(len(distances)):
        for right in range(left + 1, len(distances)):
            if np.isclose(distances[left], distances[right]):
                continue
            expected_left_first = distances[left] < distances[right]
            actual_left_first = observed_ranks[left] < observed_ranks[right]
            agreements += int(expected_left_first == actual_left_first)
            comparisons += 1
    nearest = np.flatnonzero(np.isclose(distances, np.min(distances)))
    first_sensor = int(np.argmin(observed_ranks))
    first_match = bool(first_sensor in nearest)
    rank_distance = float(np.mean(np.abs(observed_ranks - np.argsort(np.argsort(distances, kind="stable"), kind="stable"))))
    return agreements / max(1, comparisons), first_match, rank_distance


def collect_candidates(data_dir: Path, config: FeatureConfig):
    candidates = []
    errors = []
    for path in sorted(data_dir.glob("**/*.csv")):
        try:
            angle = angle_from_folder(path.parent)
            samples, timestamps = load_csv(path)
            feature, diagnostics = extract_features(samples, timestamps, "order", config)
            ranks = np.asarray(diagnostics["response_rank"], dtype=int)
            consistency, first_match, rank_distance = order_consistency(ranks, angle)
            candidates.append({
                "path": path,
                "angle": angle,
                "feature": feature,
                "diagnostics": diagnostics,
                "response_sequence": diagnostics["response_sequence"],
                "consistency": consistency,
                "first_match": first_match,
                "rank_distance": rank_distance,
                "theory_order": theory_order(angle),
            })
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {error}")
    if errors:
        print(f"Skipped {len(errors)} invalid CSV file(s).")
        for error in errors[:10]:
            print(f"  - {error}")
    return candidates


def select_candidates(candidates: list[dict], per_angle: int) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["angle"]].append(candidate)
    selected = []
    for angle in ANGLES:
        group = grouped.get(angle, [])
        if len(group) < per_angle:
            raise RuntimeError(f"Angle {angle} has only {len(group)} usable CSV files; {per_angle} are required.")
        # First-response agreement is primary; then retain the greatest number
        # of correctly ordered sensor pairs, then minimize rank displacement.
        group.sort(key=lambda item: (item["first_match"], item["consistency"], -item["rank_distance"], str(item["path"])), reverse=True)
        selected.extend(group[:per_angle])
    return selected


def save_selection(path: Path, candidates: list[dict], selected: list[dict]):
    selected_paths = {item["path"] for item in selected}
    rows = []
    for item in sorted(candidates, key=lambda row: (row["angle"], str(row["path"]))):
        rows.append({
            "source_csv": str(item["path"]),
            "angle_deg": item["angle"],
            "selected": int(item["path"] in selected_paths),
            "first_response_matches_theory": int(item["first_match"]),
            "pairwise_order_consistency": round(item["consistency"], 6),
            "mean_rank_distance": round(item["rank_distance"], 6),
            "observed_response_order": "->".join(f"S{sensor}" for sensor in item["response_sequence"]),
            "theoretical_order": "->".join(f"S{sensor}" for sensor in item["theory_order"]),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_selected(selected: list[dict], output_dir: Path, data_dir: Path, config: FeatureConfig):
    X = np.asarray([item["feature"] for item in selected], dtype=float)
    y = np.asarray([item["angle"] for item in selected], dtype=int)
    classifier = make_classifier()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    predicted = cross_val_predict(classifier, X, y, cv=cv)
    metrics = {
        "sample_count": int(len(y)),
        "samples_per_angle": 2,
        "cv_folds": 2,
        "cv_accuracy": float(accuracy_score(y, predicted)),
        "cv_circular_mae_deg": float(np.mean(circular_error(predicted, y))),
        "cv_confusion_matrix": confusion_matrix(y, predicted, labels=list(ANGLES)).tolist(),
    }
    classifier.fit(X, y)
    artifact = {
        "model": classifier,
        "method": "order",
        "model_name": "theory_selected_order",
        "angles": list(ANGLES),
        "feature_names": selected[0]["diagnostics"]["feature_names"],
        "feature_config": config.__dict__,
        "sensor_count": 6,
        "metrics": metrics,
        "selection_rule": "top two per angle: theoretical-first response, pairwise geometry-order consistency, rank displacement",
        "source_data_dir": str(data_dir.resolve()),
        "selected_sample_paths": [str(item["path"]) for item in selected],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_dir / "direction_model_theory_selected_order.joblib")
    (output_dir / "direction_metrics_theory_selected_order.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Select the two geometry-consistent response-order CSV files per angle and train a model.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base_dir.parent / "方向传感数据")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "models")
    parser.add_argument("--per-angle", type=int, default=2)
    args = parser.parse_args()
    if args.per_angle != 2:
        raise ValueError("This training workflow intentionally selects exactly 2 experiments per angle.")
    config = FeatureConfig()
    candidates = collect_candidates(args.data_dir, config)
    selected = select_candidates(candidates, args.per_angle)
    selection_path = args.output_dir / "theory_selected_samples.csv"
    save_selection(selection_path, candidates, selected)
    metrics = train_selected(selected, args.output_dir, args.data_dir, config)
    print(f"Selected {len(selected)} samples (2 per angle).")
    for angle in ANGLES:
        paths = [item["path"].name for item in selected if item["angle"] == angle]
        print(f"  {angle:3d} deg: {', '.join(paths)}")
    print(f"Selection table: {selection_path}")
    print(f"2-fold CV accuracy={metrics['cv_accuracy']:.3f}, circular MAE={metrics['cv_circular_mae_deg']:.1f} deg")


if __name__ == "__main__":
    main()
