"""Feature extraction shared by training and live direction inference."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


SENSOR_COUNT = 6
ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
SENSOR_ANGLES_DEG = np.arange(0, 360, 60, dtype=float)


@dataclass
class FeatureConfig:
    # The live detector uses baseline_seconds with the sample limits below;
    # offline CSVs use the first baseline_fraction of each recording.
    baseline_fraction: float = 0.12
    baseline_seconds: float = 3.0
    min_baseline_samples: int = 20
    max_baseline_samples: int = 80
    smooth_window: int = 5
    onset_sigma: float = 5.0
    onset_fraction: float = 0.10
    # After one channel has triggered the event, a secondary channel may have
    # a smaller relative amplitude because it is farther from the plume. Its
    # rise is still required to clear the per-channel noise threshold and to
    # remain positive for sustained_samples; this lower fraction prevents a
    # high-baseline S2/S4/S6 from being incorrectly marked as silent.
    secondary_onset_fraction: float = 0.01
    secondary_slope_fraction: float = 0.005
    slope_sigma: float = 2.0
    sustained_samples: int = 3
    min_response_snr: float = 3.0
    min_active_sensors: int = 6
    sensor_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    # Live decision rule: the first reliable positive rise on any of the six
    # sensors starts primary_observe_seconds. The remaining channels are
    # ranked if they respond before the deadline; silent channels do not block
    # the decision. The field name is retained for model-file compatibility.
    primary_sensors: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    primary_observe_seconds: float = 5.0
    # first_response model: the first sensor supplies the base azimuth. The
    # next five response positions add progressively smaller corrections based
    # on their signed angular difference from that first sensor.
    first_response_rank_weights: tuple[float, ...] = (0.50, 0.25, 0.15, 0.07, 0.03)


def angle_from_folder(path: Path) -> int:
    match = re.search(r"(\d+)\s*度", path.name)
    if not match:
        raise ValueError(f"Cannot infer direction angle from folder: {path}")
    angle = int(match.group(1)) % 360
    if angle not in ANGLES:
        raise ValueError(f"Angle must be one of {ANGLES}: {path}")
    return angle


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load six sensor columns and optional timestamps from a monitor CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames if name}
        sensor_fields = []
        for index in range(1, SENSOR_COUNT + 1):
            key = f"sensor{index}"
            if key not in fields:
                raise ValueError(f"Missing {key} column")
            sensor_fields.append(fields[key])
        timestamp_field = fields.get("timestamp") or fields.get("time")
        rows: list[list[float]] = []
        timestamps: list[float] = []
        have_timestamps = timestamp_field is not None
        for row in reader:
            try:
                rows.append([float(row[field].strip()) for field in sensor_fields])
                if have_timestamps:
                    raw = (row.get(timestamp_field) or "").strip()
                    try:
                        timestamps.append(datetime.fromisoformat(raw).timestamp())
                    except ValueError:
                        have_timestamps = False
                        timestamps.clear()
            except (AttributeError, TypeError, ValueError):
                continue
    if len(rows) < 8:
        raise ValueError("CSV has fewer than 8 valid samples")
    time_axis = np.asarray(timestamps, dtype=float) if have_timestamps and len(timestamps) == len(rows) else None
    return np.asarray(rows, dtype=float), time_axis


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])])


def _first_sustained(values: np.ndarray, threshold: float, count: int) -> int:
    count = max(1, int(count))
    mask = values >= threshold
    if count == 1:
        indices = np.flatnonzero(mask)
        return int(indices[0]) if len(indices) else int(np.argmax(values))
    runs = np.convolve(mask.astype(int), np.ones(count, dtype=int), mode="valid")
    indices = np.flatnonzero(runs >= count)
    return int(indices[0]) if len(indices) else int(np.argmax(values))


def _first_sustained_crossing(values: np.ndarray, threshold: float, count: int) -> float:
    """Return a sub-sample onset time by interpolating the threshold crossing."""
    index = _first_sustained(values, threshold, count)
    if index <= 0:
        return float(index)
    previous = float(values[index - 1])
    current = float(values[index])
    if current <= previous or current <= threshold:
        return float(index)
    fraction = (float(threshold) - previous) / (current - previous)
    return float(index - 1 + np.clip(fraction, 0.0, 1.0))


def _first_sustained_rise(values: np.ndarray, amplitude_threshold: float, slope_threshold: float, count: int) -> float:
    """Return the first visible upward change-point, not a noise crossing."""
    count = max(1, int(count))
    values = np.asarray(values, dtype=float)
    mask = values > float(amplitude_threshold) * (1.0 + 1e-9)
    if len(mask) < count:
        return float(np.argmax(values))
    runs = np.convolve(mask.astype(int), np.ones(count, dtype=int), mode="valid")
    slope = np.diff(values, prepend=float(values[0]))
    indices = np.flatnonzero(runs >= count)
    for index in indices:
        end = min(len(slope), int(index) + count + 1)
        if np.max(slope[int(index):end], initial=0.0) >= float(slope_threshold):
            return _first_sustained_crossing(values, amplitude_threshold, count)
    # A gradual but sustained rise is still usable; the slope test mainly
    # prevents ordinary baseline jitter from being called a response.
    return (_first_sustained_crossing(values, amplitude_threshold, count)
            if len(indices) else float(np.argmax(values)))


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(1, len(values) - 1)


def position_order_features(response_ranks: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert the six response ranks into geometry-aware order features.

    The six rank values are the only measurements. Sensor coordinates are the
    fixed regular-hexagon positions: S1=0 deg, S2=60 deg, ..., S6=300 deg.
    No signal amplitude or response-time interval is used here.
    """
    ranks = np.asarray(response_ranks, dtype=float)
    if ranks.shape != (SENSOR_COUNT,) or sorted(ranks.astype(int).tolist()) != list(range(SENSOR_COUNT)):
        raise ValueError("response_ranks must be a permutation of 0 through 5")
    radians = np.deg2rad(SENSOR_ANGLES_DEG)
    unit_vectors = np.column_stack([np.sin(radians), np.cos(radians)])
    # Earlier ranks receive exponentially more weight, so S3 preceding S2
    # shifts the vector toward S3 while two jointly early neighbours point
    # midway between them.
    priority = np.exp(-0.70 * ranks)
    weighted_vector = np.sum(priority[:, None] * unit_vectors, axis=0)
    first_two = np.argsort(ranks, kind="stable")[:2]
    first_three = np.argsort(ranks, kind="stable")[:3]
    first_two_vector = np.mean(unit_vectors[first_two], axis=0)
    first_three_vector = np.mean(unit_vectors[first_three], axis=0)
    feature = np.concatenate([ranks, weighted_vector, first_two_vector, first_three_vector])
    angle = float(np.rad2deg(np.arctan2(weighted_vector[0], weighted_vector[1])) % 360.0)
    return feature, {
        "geometry_angle_deg": angle,
        "weighted_vector": weighted_vector.tolist(),
        "first_two_sensors": (first_two + 1).astype(int).tolist(),
        "first_three_sensors": (first_three + 1).astype(int).tolist(),
        "feature_names": ([f"response_rank_s{i}" for i in range(1, 7)] +
                          ["weighted_right", "weighted_front", "first_two_right", "first_two_front", "first_three_right", "first_three_front"]),
    }


def _t50_times(normalized: np.ndarray, onset_times: np.ndarray, active_channels: np.ndarray) -> np.ndarray:
    """Legacy feature helper kept for offline comparison methods only."""
    n = len(normalized)
    result = np.zeros(SENSOR_COUNT, dtype=float)
    for sensor in range(SENSOR_COUNT):
        if not active_channels[sensor]:
            result[sensor] = onset_times[sensor]
            continue
        onset_index = min(n - 1, max(0, int(np.floor(onset_times[sensor]))))
        after_onset = np.flatnonzero(normalized[onset_index:, sensor] >= 0.5)
        result[sensor] = onset_index + int(after_onset[0]) if len(after_onset) else int(np.argmax(normalized[:, sensor]))
    return result


def _normalize_times(times: np.ndarray, timestamps: np.ndarray | None, n: int) -> np.ndarray:
    if timestamps is not None and len(timestamps) == n and np.all(np.diff(timestamps) >= 0):
        axis = np.asarray(timestamps, dtype=float)
        axis = axis - axis[0]
        duration = float(axis[-1]) if axis[-1] > 0 else float(max(1, n - 1))
        return np.interp(times, np.arange(n), axis) / duration
    return times / float(max(1, n - 1))


def extract_features(samples: np.ndarray, timestamps: np.ndarray | None = None, method: str = "combined", config: FeatureConfig | None = None) -> tuple[np.ndarray, dict]:
    """Extract features from one six-channel experiment.

    ``order`` uses only the ordinal order in which the six channels respond.
    No absolute time, t50, peak, area, or amplitude is included in that
    feature. ``t50`` and the other feature families remain available for
    offline comparison; the integrated live models use the complete six-way
    positive-rise sequence.
    """
    config = config or FeatureConfig()
    order_only = method in ("order", "sequence_8class")
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != SENSOR_COUNT:
        raise ValueError(f"Expected an N x {SENSOR_COUNT} sensor matrix")
    n = len(values)
    baseline_count = min(n - 1, max(config.min_baseline_samples, int(n * config.baseline_fraction)))
    baseline_count = min(baseline_count, max(config.min_baseline_samples, config.max_baseline_samples))
    baseline_count = max(3, baseline_count)
    baseline = np.median(values[:baseline_count], axis=0)
    # Normalize each channel independently. Only upward departures from its
    # own baseline are considered; absolute channel amplitude is not used.
    smoothed = _smooth(np.maximum(values - baseline, 0.0), config.smooth_window)
    baseline_noise = np.maximum(np.median(np.abs(values[:baseline_count] - baseline), axis=0) * 1.4826, 1e-9)
    peaks = np.max(smoothed, axis=0)
    response_snr = peaks / baseline_noise
    active_channels = response_snr >= config.min_response_snr
    selected_active = active_channels[list(config.sensor_indices)]
    required_active = min(config.min_active_sensors, len(config.sensor_indices))
    if int(np.sum(selected_active)) < required_active:
        raise ValueError(f"Insufficient reliable selected-sensor responses: {int(np.sum(selected_active))}/{len(config.sensor_indices)} have SNR >= {config.min_response_snr}")
    normalized = smoothed / np.maximum(peaks, 1e-9)
    onset_times = np.zeros(SENSOR_COUNT, dtype=float)
    for sensor in range(SENSOR_COUNT):
        if not active_channels[sensor]:
            onset_times[sensor] = n + sensor / SENSOR_COUNT
            continue
        amplitude_threshold = max(
            config.onset_sigma * baseline_noise[sensor],
            config.onset_fraction * max(abs(float(baseline[sensor])), 1.0),
            config.onset_fraction * float(peaks[sensor]),
        )
        slope_threshold = max(
            config.slope_sigma * baseline_noise[sensor],
            0.02 * max(abs(float(baseline[sensor])), 1.0),
        )
        onset = _first_sustained_rise(
            smoothed[:, sensor], amplitude_threshold, slope_threshold, config.sustained_samples
        )
        onset_times[sensor] = onset
    if timestamps is not None and len(timestamps) == n and np.all(np.diff(timestamps) >= 0):
        axis = np.asarray(timestamps, dtype=float)
        axis = axis - axis[0]
        duration = float(axis[-1]) if axis[-1] > 0 else float(max(1, n - 1))
        onset_norm = np.interp(onset_times, np.arange(n), axis) / duration
    else:
        duration = float(max(1, n - 1))
        onset_norm = onset_times / duration
    onset_rank = _ordinal_ranks(onset_times)
    if order_only:
        # This vector is the sequence alone: rank 0 responds first and rank 5
        # responds last. Example 1->4->3->2->5->6 is [0, 3, 2, 1, 4, 5].
        selected_onsets = onset_times[list(config.sensor_indices)]
        feature = _ordinal_ranks(selected_onsets) * (len(config.sensor_indices) - 1)
        names = [f"response_rank_s{i + 1}" for i in config.sensor_indices]
    elif method == "position_order":
        response_ranks = np.full(SENSOR_COUNT, SENSOR_COUNT - 1, dtype=int)
        selected_onsets = onset_times[list(config.sensor_indices)]
        selected_ranks = (_ordinal_ranks(selected_onsets) * (len(config.sensor_indices) - 1)).astype(int)
        response_ranks[list(config.sensor_indices)] = selected_ranks
        feature, position_info = position_order_features(response_ranks)
        names = position_info["feature_names"]
    elif method == "first_response":
        first_sensor = int(np.argmin(onset_rank))
        feature = np.zeros(SENSOR_COUNT, dtype=float)
        feature[first_sensor] = 1.0
        names = [f"first_response_s{i}" for i in range(1, 7)]
    elif method == "t50":
        t50_times = _t50_times(normalized, onset_times, active_channels)
        t50_norm = _normalize_times(t50_times, timestamps, n)
        t50_rank = _ordinal_ranks(t50_times)
        feature = np.concatenate([t50_norm, t50_rank])
        names = [f"t50_time_s{i}" for i in range(1, 7)] + [f"t50_rank_s{i}" for i in range(1, 7)]
    elif method == "profile":
        t50_times = _t50_times(normalized, onset_times, active_channels)
        t50_norm = _normalize_times(t50_times, timestamps, n)
        t50_rank = _ordinal_ranks(t50_times)
        peak_share = peaks / max(float(np.sum(peaks)), 1e-9)
        # A response-centroid/area feature uses the complete rising curve,
        # instead of relying only on one threshold crossing. It is useful when
        # gas arrives gradually or when the first peak is noisy.
        sample_axis = np.linspace(0.0, 1.0, n)
        area = np.trapezoid(normalized, sample_axis, axis=0)
        centroid = np.sum(normalized * sample_axis[:, None], axis=0) / np.maximum(np.sum(normalized, axis=0), 1e-9)
        peak_position = np.argmax(normalized, axis=0) / max(1, n - 1)
        area_share = area / max(float(np.sum(area)), 1e-9)
        feature = np.concatenate([onset_norm, t50_norm, centroid, peak_position, peak_share, area_share])
        names = ([f"onset_time_s{i}" for i in range(1, 7)] + [f"t50_time_s{i}" for i in range(1, 7)] +
                 [f"response_centroid_s{i}" for i in range(1, 7)] + [f"peak_position_s{i}" for i in range(1, 7)] +
                 [f"peak_share_s{i}" for i in range(1, 7)] + [f"area_share_s{i}" for i in range(1, 7)])
    elif method == "combined":
        t50_times = _t50_times(normalized, onset_times, active_channels)
        t50_norm = _normalize_times(t50_times, timestamps, n)
        t50_rank = _ordinal_ranks(t50_times)
        peak_share = peaks / max(float(np.sum(peaks)), 1e-9)
        feature = np.concatenate([onset_norm, onset_rank, t50_norm, t50_rank, peak_share])
        names = ([f"onset_time_s{i}" for i in range(1, 7)] + [f"onset_rank_s{i}" for i in range(1, 7)] +
                 [f"t50_time_s{i}" for i in range(1, 7)] + [f"t50_rank_s{i}" for i in range(1, 7)] +
                 [f"peak_share_s{i}" for i in range(1, 7)])
    else:
        raise ValueError("method must be order, sequence_8class, position_order, first_response, t50, profile, or combined")
    diagnostics = {
        "baseline": baseline.tolist(),
        "peaks": peaks.tolist(),
        "response_snr": response_snr.tolist(),
        "active_channels": active_channels.tolist(),
        "selected_sensor_numbers": [index + 1 for index in config.sensor_indices],
        "onset_norm": onset_norm.tolist(),
        "onset_rank": onset_rank.tolist(),
        "response_rank": (onset_rank * (SENSOR_COUNT - 1)).astype(int).tolist(),
        "response_sequence": (np.argsort(onset_rank, kind="stable") + 1).astype(int).tolist(),
        "feature_names": names,
    }
    if method in ("t50", "profile", "combined"):
        diagnostics["t50_norm"] = t50_norm.tolist()
        diagnostics["t50_rank"] = t50_rank.tolist()
    if method == "position_order":
        diagnostics.update(position_info)
    return feature.astype(float), diagnostics
