"""Shared detection core for the Linux gas-direction toolset.

This module contains everything that is independent of the user interface:

- JSON result sinks (stdout / UDP / serial) for robot integration,
- ``DirectionEstimator``: loads a trained model and predicts a direction,
- ``DetectionSession``: the streaming state machine
  (baseline -> first trigger -> fixed observation window -> locked result),
- ``SerialFrameReader`` / ``ReplayReader``: data sources that feed a session.

Both the Tk GUI (``direction_inference.py``) and the headless tool
(``direction_cli.py``) use these classes, so the detection rule is identical
in graphical, headless and replay modes, and matches the Windows version.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import serial
import serial.tools.list_ports

from direction_features import (
    SENSOR_ANGLES_DEG,
    FeatureConfig,
    extract_features,
    load_csv,
    position_order_features,
)


SENSOR_COUNT = 6
SUPPORTED_MODEL_METHODS = ("order", "sequence_8class", "first_response", "position_order", "t50", "profile", "combined")
# These models are scored from the positive-rise response order.  The live
# rule starts as soon as any of the six sensors responds, then ranks every
# channel that responds during the fixed observation window. Silent channels
# are treated as "not observed" and never block the decision.
COMPLETE_SEQUENCE_METHODS = ("order", "sequence_8class", "first_response", "position_order")
ALL_SENSOR_INDICES = tuple(range(SENSOR_COUNT))


class RobotCommandSink:
    """Replace or extend this class for a machine-dog SDK/ROS 2 adapter."""

    def send_direction(self, angle: float, confidence: float, method: str):
        raise NotImplementedError

    def close(self):
        pass


class NullSink(RobotCommandSink):
    """Default sink: print one JSON line per locked direction to stdout."""

    def send_direction(self, angle, confidence, method):
        print(json.dumps({"type": "gas_direction", "angle_deg": angle, "confidence": confidence, "method": method}, ensure_ascii=False), flush=True)


class UdpJsonSink(RobotCommandSink):
    def __init__(self, host: str, port: int):
        self.target = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_direction(self, angle, confidence, method):
        payload = {"type": "gas_direction", "angle_deg": float(angle), "confidence": float(confidence), "method": method, "timestamp": time.time()}
        self.sock.sendto((json.dumps(payload) + "\n").encode("utf-8"), self.target)

    def close(self):
        self.sock.close()


class SerialJsonSink(RobotCommandSink):
    def __init__(self, device: str, baud: int):
        self.ser = serial.Serial(device, baud, timeout=0.1)

    def send_direction(self, angle, confidence, method):
        payload = {"type": "gas_direction", "angle_deg": float(angle), "confidence": float(confidence), "method": method}
        self.ser.write((json.dumps(payload) + "\n").encode("utf-8"))

    def close(self):
        self.ser.close()


def create_sink(spec: str | None, baud: int) -> RobotCommandSink:
    """Build a result sink from ``udp:HOST:PORT`` or ``serial:DEVICE``."""
    if not spec:
        return NullSink()
    if spec.startswith("udp:"):
        host, port = spec[4:].rsplit(":", 1)
        return UdpJsonSink(host, int(port))
    if spec.startswith("serial:"):
        return SerialJsonSink(spec[7:], baud)
    raise ValueError("Robot output must be udp:HOST:PORT or serial:DEVICE")


def list_serial_ports():
    """Return pyserial port info objects sorted by device name."""
    return sorted(serial.tools.list_ports.comports(), key=lambda port: port.device)


class DirectionEstimator:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]
        self.method = self.artifact["method"]
        if self.method not in SUPPORTED_MODEL_METHODS:
            raise ValueError(f"Unsupported model method: {self.method}. Supported methods: {', '.join(SUPPORTED_MODEL_METHODS)}")
        self.config = FeatureConfig(**self.artifact.get("feature_config", {}))
        # Older artifacts stored the former S1/S3/S5 trigger set.  The live
        # interface now always starts its window from the first response on
        # any of S1-S6; the command-line option can still intentionally narrow
        # this set for a special experiment.
        self.config.primary_sensors = ALL_SENSOR_INDICES

    def _predict_feature(self, feature: np.ndarray):
        probabilities = self.model.predict_proba([feature])[0]
        classes = np.asarray(self.model.classes_, dtype=float)
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        # Circular interpolation turns the eight class probabilities into a
        # continuous angle (for example 37.6 deg), including the 315/0 wrap.
        model_vector = np.sum(probabilities * np.exp(1j * np.deg2rad(classes)))
        model_angle = float(np.rad2deg(np.angle(model_vector)) % 360)
        return model_angle, confidence, float(classes[best]), probabilities, model_vector

    def _sequential_first_response_angle(self, response_sequence: list[int]) -> tuple[float, dict]:
        """Estimate an angle from the first response plus ordered corrections.

        The first sensor azimuth is the base. For each later observed sensor,
        take its shortest signed angular difference from that base, multiply it
        by the coefficient for its response rank, and add the correction to
        the base. This works with a partial five-second sequence as well as a
        complete six-sensor sequence.
        """
        observed, _ = self._complete_response_sequence(response_sequence)
        first_sensor = observed[0]
        base_angle = float(SENSOR_ANGLES_DEG[first_sensor - 1])
        configured = tuple(float(value) for value in self.config.first_response_rank_weights)
        if not configured:
            configured = (0.0,)
        weights = list(configured[:SENSOR_COUNT - 1])
        weights.extend([weights[-1]] * (SENSOR_COUNT - 1 - len(weights)))
        corrections = []
        angle = base_angle
        for rank, sensor in enumerate(observed[1:]):
            sensor_angle = float(SENSOR_ANGLES_DEG[sensor - 1])
            signed_difference = (sensor_angle - base_angle + 180.0) % 360.0 - 180.0
            correction = weights[rank] * signed_difference
            corrections.append({
                "rank": rank + 2,
                "sensor": sensor,
                "sensor_azimuth_deg": sensor_angle,
                "signed_difference_from_first_deg": signed_difference,
                "weight": weights[rank],
                "correction_deg": correction,
            })
            angle = (angle + correction) % 360.0
        return float(angle), {
            "first_response_sensor": first_sensor,
            "first_sensor_azimuth_deg": base_angle,
            "first_response_rank_weights": weights,
            "first_response_corrections": corrections,
            "sequential_angle_deg": float(angle),
        }

    def predict(self, samples: np.ndarray, timestamps: np.ndarray | None = None):
        feature, diagnostics = extract_features(samples, timestamps, self.method, self.config)
        model_angle, confidence, predicted_class, _, _ = self._predict_feature(feature)
        # sequence_8class is deliberately a discrete classifier.  Unlike the
        # older models, it must never interpolate between adjacent classes.
        if self.method == "sequence_8class":
            angle = predicted_class
        elif self.method == "first_response":
            angle, sequential = self._sequential_first_response_angle(diagnostics["response_sequence"])
            diagnostics.update(sequential)
        else:
            angle = model_angle
        diagnostics["model_angle_deg"] = model_angle
        diagnostics["predicted_class_deg"] = predicted_class
        return angle, confidence, diagnostics

    @staticmethod
    def _complete_response_sequence(response_sequence: list[int]) -> tuple[list[int], list[int]]:
        """Validate a partial onset order and append silent sensors for features.

        Live data can end with only a subset of sensors responding. The
        learned order models still require a six-element rank vector, so the
        observed order is kept intact and unobserved sensors are appended as a
        deterministic late tie-break. They are explicitly marked in the
        diagnostics and never shown as real responses in the GUI.
        """
        observed = [int(sensor) for sensor in response_sequence]
        if not observed:
            raise ValueError("response_sequence must contain at least one responding sensor")
        if any(sensor not in range(1, SENSOR_COUNT + 1) for sensor in observed):
            raise ValueError("response_sequence sensor numbers must be between 1 and 6")
        if len(set(observed)) != len(observed):
            raise ValueError("response_sequence cannot contain duplicate sensor numbers")
        missing = [sensor for sensor in range(1, SENSOR_COUNT + 1) if sensor not in observed]
        return observed, observed + missing

    def predict_from_response_sequence(self, response_sequence: list[int]):
        observed_sequence, complete_sequence = self._complete_response_sequence(response_sequence)
        unresponded = [sensor for sensor in complete_sequence if sensor not in observed_sequence]
        if self.method == "first_response":
            return self.predict_from_first_sensor(observed_sequence[0], observed_sequence)
        ranks = np.empty(SENSOR_COUNT, dtype=int)
        for rank, sensor in enumerate(complete_sequence):
            ranks[sensor - 1] = rank
        if self.method in ("order", "sequence_8class"):
            _, confidence, predicted_class, probabilities, _ = self._predict_feature(ranks.astype(float))
            diagnostics = {
                "response_rank": ranks.tolist(),
                "response_sequence": observed_sequence,
                "complete_response_sequence": complete_sequence,
                "unresponded_sensors": unresponded,
                "predicted_class_deg": predicted_class,
                "class_probabilities": {
                    str(int(angle)): float(probability)
                    for angle, probability in zip(self.model.classes_, probabilities)
                },
            }
            angle = predicted_class
            diagnostics["model_angle_deg"] = angle
            diagnostics["first_response_sensor"] = observed_sequence[0]
            diagnostics["first_sensor_azimuth_deg"] = float((observed_sequence[0] - 1) * 60)
            return angle, confidence, diagnostics
        feature, geometry = position_order_features(ranks)
        model_angle, confidence, _, _, model_vector = self._predict_feature(feature)
        geometry_angle = geometry["geometry_angle_deg"]
        # Both vectors are derived exclusively from sensor geometry and order.
        # The geometric vector is primary: a low-confidence learned model must
        # not overturn the physical S2/S3 right-side relationship. Training
        # data provides only a bounded correction once its confidence is high.
        geometric_vector = np.exp(1j * np.deg2rad(geometry_angle))
        learned_vector = model_vector / max(abs(model_vector), 1e-9)
        learned_weight = min(0.35, max(0.0, (confidence - 0.35) / 0.40))
        blended_vector = (1.0 - learned_weight) * geometric_vector + learned_weight * learned_vector
        angle = float(np.rad2deg(np.angle(blended_vector)) % 360.0)
        diagnostics = {
            **geometry,
            "response_rank": ranks.tolist(),
            "response_sequence": observed_sequence,
            "complete_response_sequence": complete_sequence,
            "unresponded_sensors": unresponded,
            "model_angle_deg": model_angle,
            "learned_correction_weight": learned_weight,
            "blended_angle_deg": angle,
        }
        return angle, confidence, diagnostics

    def predict_from_first_sensor(self, first_sensor: int, response_sequence: list[int]):
        if first_sensor not in range(1, SENSOR_COUNT + 1):
            raise ValueError("first_sensor must be in 1 through 6")
        observed_sequence, complete_sequence = self._complete_response_sequence(response_sequence)
        if observed_sequence[0] != first_sensor:
            raise ValueError("first_sensor must match the first element of response_sequence")
        feature = np.zeros(SENSOR_COUNT, dtype=float)
        feature[first_sensor - 1] = 1.0
        model_angle, confidence, predicted_class, _, _ = self._predict_feature(feature)
        observed_ranks = np.empty(SENSOR_COUNT, dtype=float)
        for rank, sensor in enumerate(complete_sequence):
            observed_ranks[sensor - 1] = rank
        profile = self.artifact.get("validation_profiles", {}).get(str(int(predicted_class)))
        if profile:
            expected_ranks = np.asarray(profile["mean_response_rank"], dtype=float)
            mean_rank_error = float(np.mean(np.abs(observed_ranks - expected_ranks)))
            validation_score = max(0.0, 1.0 - mean_rank_error / (SENSOR_COUNT - 1))
            validation = "consistent" if validation_score >= 0.55 else "needs review"
        else:
            mean_rank_error = None
            validation_score = None
            validation = "no profile"
        diagnostics = {
            "response_rank": observed_ranks.astype(int).tolist(),
            "response_sequence": observed_sequence,
            "complete_response_sequence": complete_sequence,
            "unresponded_sensors": [sensor for sensor in complete_sequence if sensor not in observed_sequence],
            "first_response_sensor": first_sensor,
            "predicted_class_deg": predicted_class,
            "model_angle_deg": model_angle,
            "first_sensor_azimuth_deg": float((first_sensor - 1) * 60),
            "validation": validation,
            "validation_score": validation_score,
            "mean_rank_error": mean_rank_error,
        }
        angle, sequential = self._sequential_first_response_angle(observed_sequence)
        diagnostics.update(sequential)
        return angle, confidence, diagnostics


class DetectionSession:
    """Streaming detector: baseline -> trigger -> observation window -> lock.

    The rule is the same as the Windows program: collect a quiet baseline,
    wait for the first reliable positive rise on any of S1-S6, then rank every
    channel that responds inside the fixed observation window.  Silent
    channels never block the decision and are reported as "unresponded".
    """

    def __init__(self, estimator: DirectionEstimator, sink: RobotCommandSink,
                 on_update=None, on_result=None, on_status=None):
        self.estimator = estimator
        self.sink = sink
        self.on_update = on_update or (lambda samples: None)
        self.on_result = on_result or (lambda angle, confidence, diagnostics: None)
        self.on_status = on_status or (lambda text: None)
        self.reset()

    def reset(self):
        self.samples: list[list[float]] = []
        self.times: list[float] = []
        self.baseline = None
        self.noise = None
        self.state = "armed"
        self.trigger_index = None
        self.baseline_started_at = None
        self.response_history: list[list[float]] = []
        self.response_history_times: list[float] = []
        self.response_onsets: dict[int, int] = {}
        self.response_sequence: list[int] = []
        self.primary_ready_at = None
        self.observe_deadline = None
        self.observation_start_index = None
        self.last_progress_text = ""
        self.on_status("Armed - collecting a few seconds of baseline")

    def _primary_sensors(self) -> list[int]:
        """Sensor numbers whose first response starts the observation window."""
        return [index + 1 for index in self.estimator.config.primary_sensors]

    def _primary_label(self) -> str:
        return "/".join(f"S{sensor}" for sensor in self._primary_sensors())

    def _responded_sequence(self) -> list[int]:
        """Only sensors that actually responded, ordered by onset time."""
        return [
            sensor for sensor, _ in sorted(self.response_onsets.items(), key=lambda item: (item[1], item[0]))
        ]

    def _report_observation(self, now: float):
        window = float(self.estimator.config.primary_observe_seconds)
        remaining = max(0.0, float(self.observe_deadline) - now)
        # Refresh the countdown at half-second granularity so the status line
        # stays readable instead of changing on every serial frame.
        remaining = round(remaining * 2.0) / 2.0
        responded = self._responded_sequence()
        order_text = " -> ".join(f"S{sensor}" for sensor in responded) or "none yet"
        text = (
            f"Sensor trigger ({self._primary_label()}) - observing {window:.0f} s "
            f"({remaining:.1f} s left); order so far: {order_text}"
        )
        if text != self.last_progress_text:
            self.last_progress_text = text
            self.on_status(text)

    def _response_candidates(self, strict: bool = True, min_onset_index: int | None = None):
        """Return channels with a sustained, clearly upward change-point.

        ``strict=True`` is used before the event starts and preserves the
        10%-of-baseline gas-trigger rule. Once a channel has triggered,
        ``strict=False`` is used for the other five channels: their own MAD
        noise plus a small normalized positive rise is enough to record a
        genuine but weaker plume response.
        """
        if self.baseline is None or self.noise is None:
            return []
        count = max(1, int(self.estimator.config.sustained_samples))
        if len(self.response_history) < count:
            return []
        history = np.asarray(self.response_history, dtype=float)
        config = self.estimator.config
        fraction = config.onset_fraction if strict else min(
            config.onset_fraction, config.secondary_onset_fraction
        )
        slope_fraction = 0.02 if strict else min(0.02, config.secondary_slope_fraction)
        thresholds = np.maximum(
            config.onset_sigma * self.noise,
            fraction * np.maximum(np.abs(self.baseline), 1.0),
        )
        slope_thresholds = np.maximum(
            config.slope_sigma * self.noise,
            slope_fraction * np.maximum(np.abs(self.baseline), 1.0),
        )
        candidates = []
        for sensor in range(SENSOR_COUNT):
            if sensor + 1 in self.response_onsets:
                continue
            strict_threshold = thresholds[sensor] * (1.0 + 1e-9)
            latest_start = len(history) - count
            onset_index = None
            first_start = max(0, int(min_onset_index or 0))
            for start in range(first_start, max(first_start, latest_start) + 1):
                window = history[start:start + count, sensor]
                if len(window) < count or not np.all(window > strict_threshold):
                    continue
                preceding = history[max(0, start - 1):start + count, sensor]
                slope = np.diff(preceding, prepend=preceding[0])
                if np.max(slope, initial=0.0) >= slope_thresholds[sensor]:
                    onset_index = start
                    break
            if onset_index is not None:
                candidates.append((sensor, onset_index))
        return candidates

    def _record_responding_sensors(self, strict: bool = False, min_onset_index: int | None = None):
        """Record a newly confirmed onset for each of the six channels.

        Detection is deliberately channel-independent: once the first
        channel starts the observation window, S1 through S6 are all tested
        with their own baseline/noise/sustained-rise thresholds. Secondary
        onsets are restricted to the window start so a small baseline drift
        from before the trigger is not reported as a gas response.
        """
        candidates = self._response_candidates(
            strict=strict,
            min_onset_index=min_onset_index,
        )
        # Sensors crossing in the same serial frame have no measurable time
        # separation. Use their fixed IDs as a deterministic tie-breaker, not
        # response amplitude, so this model remains order-and-geometry only.
        candidates.sort(key=lambda item: (item[1], item[0]))
        for sensor, onset_index in candidates:
            # At the deadline, keep only onsets that actually occurred inside
            # the five-second window. Earlier onsets are still valid (their
            # confirmation may have arrived after the first trigger).
            if self.observe_deadline is not None and 0 <= onset_index < len(self.response_history_times):
                if self.response_history_times[onset_index] > self.observe_deadline:
                    continue
            self.response_onsets[sensor + 1] = int(onset_index)
        self.response_sequence = [
            sensor for sensor, _ in sorted(self.response_onsets.items(), key=lambda item: (item[1], item[0]))
        ]

    def add_sample(self, values: list[float], timestamp: float | None = None):
        if self.state == "locked":
            return
        now = float(timestamp if timestamp is not None else time.time())
        self.samples.append(list(values))
        self.times.append(now)
        if self.baseline_started_at is None:
            self.baseline_started_at = now
        config = self.estimator.config
        if self.state == "armed" and self.baseline is None:
            elapsed = now - float(self.baseline_started_at)
            enough_time = elapsed >= float(config.baseline_seconds)
            enough_samples = len(self.samples) >= int(config.min_baseline_samples)
            if enough_time and enough_samples:
                baseline_count = min(len(self.samples), max(int(config.min_baseline_samples), int(config.max_baseline_samples)))
                base = np.asarray(self.samples[:baseline_count], dtype=float)
                self.baseline = np.median(base, axis=0)
                self.noise = np.maximum(np.median(np.abs(base - self.baseline), axis=0) * 1.4826, 1e-9)
                # Keep one quiet reference frame so the first rising window
                # can prove that it contains a positive slope.
                self.response_history.append([0.0] * SENSOR_COUNT)
                self.response_history_times.append(now)
                self.on_status("Baseline established - waiting for a clear upward change")
                return
        if self.baseline is None:
            self.on_update(np.asarray(self.samples, dtype=float))
            return
        positive_change = np.maximum(np.asarray(values, dtype=float) - self.baseline, 0.0)
        self.response_history.append(positive_change.tolist())
        self.response_history_times.append(now)
        strict_candidates = self._response_candidates(strict=True)
        if self.state == "armed" and self.estimator.method in COMPLETE_SEQUENCE_METHODS:
            # The window is keyed to the first response of any monitored
            # channel. All six sensors are in the default trigger set, and
            # every later onset is collected if it falls within the window.
            primary_candidates = [
                (sensor, onset_index) for sensor, onset_index in strict_candidates
                if sensor + 1 in self._primary_sensors()
            ]
            if primary_candidates:
                first_primary_index = min(onset_index for _, onset_index in primary_candidates)
                self.state = "observing"
                self.trigger_index = len(self.samples) - 1
                self.observation_start_index = first_primary_index
                if 0 <= first_primary_index < len(self.response_history_times):
                    self.primary_ready_at = self.response_history_times[first_primary_index]
                else:
                    self.primary_ready_at = now
                self.observe_deadline = self.primary_ready_at + float(config.primary_observe_seconds)
                self._record_responding_sensors(strict=True, min_onset_index=first_primary_index)
                self.last_progress_text = ""
                self.on_status(
                    f"Gas detected by {', '.join(f'S{s + 1}' for s, _ in primary_candidates)} "
                    f"- observing {config.primary_observe_seconds:.0f} s"
                )
        elif self.state == "armed" and strict_candidates:
            # Keep older amplitude/time models loadable.  Their historical
            # capture rule remains unchanged because they do not consume a
            # response-order sequence.
            self.state = "capturing"
            self.trigger_index = len(self.samples) - 1
            self._record_responding_sensors(strict=True)
            self.last_progress_text = ""
            self.on_status("Gas detected - capturing clear response onsets")
        if self.state == "observing":
            self._record_responding_sensors(
                strict=False,
                min_onset_index=self.observation_start_index,
            )
            all_responded = len(self.response_onsets) >= SENSOR_COUNT
            if all_responded or now >= float(self.observe_deadline):
                # Pass only genuine responses. The estimator appends silent
                # channels internally for a fixed-size model feature vector.
                self._predict_once(self._responded_sequence())
            else:
                self._report_observation(now)
        elif self.state == "capturing" and self.estimator.method not in COMPLETE_SEQUENCE_METHODS:
            # Keep older amplitude/time models loadable.  They are not the
            # recommended source-direction rule, but can still be used for
            # comparison from the captured window.
            if len(self.samples) - int(self.trigger_index or 0) >= 12:
                self._predict_once()
        if len(self.samples) > 1200:
            self.samples = self.samples[-600:]
            self.times = self.times[-600:]
            self.response_history = self.response_history[-600:]
            self.response_history_times = self.response_history_times[-600:]
        self.on_update(np.asarray(self.samples, dtype=float))
        return

    def force_finish(self):
        """Judge with whatever responses were collected so far.

        Used by replay mode when the recording ends while the observation
        window is still open.
        """
        if self.state == "observing":
            self._predict_once(self._responded_sequence())
        elif self.state == "capturing":
            self._predict_once()

    def _predict_once(self, response_sequence: list[int] | None = None, first_sensor: int | None = None):
        self.state = "locked"
        samples = np.asarray(self.samples, dtype=float)
        timestamps = np.asarray(self.times, dtype=float)
        if first_sensor is not None:
            angle, confidence, diagnostics = self.estimator.predict_from_first_sensor(first_sensor, response_sequence or [])
        elif response_sequence is None:
            angle, confidence, diagnostics = self.estimator.predict(samples, timestamps)
        else:
            self.response_sequence = list(response_sequence)
            angle, confidence, diagnostics = self.estimator.predict_from_response_sequence(response_sequence)
        responded = sorted(self.response_onsets, key=lambda sensor: (self.response_onsets[sensor], sensor))
        diagnostics["primary_sensors"] = self._primary_sensors()
        diagnostics["responded_sensors"] = responded
        diagnostics["unresponded_sensors"] = [
            sensor for sensor in range(1, SENSOR_COUNT + 1) if sensor not in self.response_onsets
        ]
        diagnostics["onset_index_s1_to_s6"] = [
            self.response_onsets.get(sensor) for sensor in range(1, SENSOR_COUNT + 1)
        ]
        observed_rank = {sensor: rank + 1 for rank, sensor in enumerate(responded)}
        diagnostics["response_rank_s1_to_s6"] = [
            observed_rank.get(sensor) for sensor in range(1, SENSOR_COUNT + 1)
        ]
        diagnostics["response_order_all"] = responded + diagnostics["unresponded_sensors"]
        diagnostics["response_order_all_text"] = " -> ".join(
            f"S{sensor}" if sensor in responded else f"S{sensor}(no response)"
            for sensor in diagnostics["response_order_all"]
        )
        self.sink.send_direction(angle, confidence, self.estimator.method)
        self.on_result(angle, confidence, diagnostics)
        angle_text = f"{int(angle)}" if self.estimator.method == "sequence_8class" else f"{angle:.1f}"
        self.on_status(f"Direction locked: {angle_text} deg; press Reset for next event")


class SerialFrameReader(threading.Thread):
    """Read sensor frames from a serial port and feed a detection session.

    The wire protocol is the same eight-field line used by the acquisition
    program: the first six comma-separated values are S1-S6.
    """

    def __init__(self, device: str, baud: int, session: DetectionSession,
                 stop_event: threading.Event, on_error=None):
        super().__init__(daemon=True)
        self.device = device
        self.baud = int(baud)
        self.session = session
        self.stop_event = stop_event
        self.on_error = on_error or (lambda message: None)

    def run(self):
        buffer = b""
        try:
            with serial.Serial(self.device, self.baud, timeout=0.1) as port:
                while not self.stop_event.is_set():
                    chunk = port.read(256)
                    if not chunk:
                        continue
                    buffer += chunk
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()
                    for raw in lines:
                        fields = raw.decode("utf-8", errors="ignore").strip().split(",")
                        if len(fields) < SENSOR_COUNT:
                            continue
                        try:
                            values = [float(value) for value in fields[:SENSOR_COUNT]]
                        except ValueError:
                            continue
                        self.session.add_sample(values)
        except (serial.SerialException, OSError) as error:
            self.on_error(str(error))


class ReplayReader(threading.Thread):
    """Feed a recorded monitor CSV into a detection session.

    The recorded timestamps are passed to the session, so the baseline and
    observation windows keep the original time base no matter how fast the
    playback runs. ``speed`` scales the wall-clock pacing used for on-screen
    rendering; ``speed <= 0`` means "as fast as possible".
    """

    def __init__(self, path: Path, session: DetectionSession, speed: float = 1.0,
                 stop_event: threading.Event | None = None, on_error=None, on_end=None):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.session = session
        self.speed = float(speed)
        self.stop_event = stop_event or threading.Event()
        self.on_error = on_error or (lambda message: None)
        self.on_end = on_end or (lambda: None)

    def run(self):
        try:
            samples, timestamps = load_csv(self.path)
        except (OSError, ValueError) as error:
            self.on_error(str(error))
            return
        if timestamps is None:
            timestamps = np.arange(len(samples), dtype=float) * 0.1
        previous = None
        for row, moment in zip(samples, timestamps):
            if self.stop_event.is_set():
                return
            if previous is not None and self.speed > 0:
                delay = max(0.0, float(moment - previous)) / self.speed
                time.sleep(min(delay, 2.0))
            previous = moment
            self.session.add_sample([float(value) for value in row], float(moment))
        self.on_end()
