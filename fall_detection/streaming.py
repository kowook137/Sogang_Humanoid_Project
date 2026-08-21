"""Constant-time streaming fall detection for on-device (Jetson) operation.

The offline path recomputes `extract_features` and `detect_fall` over the whole
buffer for every new frame. On a Jetson Orin Nano that costs 65 ms at a
120-frame window and 238 ms at 450 frames, against 35 ms for the GPU pose
inference itself, so the rule layer - not the network - sets the frame rate.

This module keeps the same features, the same `DetectorConfig` thresholds and
the same state transitions, but updates them incrementally: it stores only the
~1 second of history the drop features actually read, so per-frame cost is
constant and independent of how long the camera has been running.

Two behaviours differ from re-running the offline detector on a sliding window,
and both are deliberate:

* The baseline body height and hip position are measured once during the
  calibration window and then frozen. On a sliding window they are re-measured
  from whatever is currently oldest in the buffer, so a few seconds after a fall
  the person's collapsed pose silently becomes the "standing" reference.
* FALLEN latches for the session. On a sliding window it clears itself as soon
  as the falling motion scrolls out of the buffer, which means an unattended
  person on the floor stops being reported.

Causality: the offline helpers may look ahead (linear fill of a short gap needs
the value after the gap, `rolling_median` is centered). Live there is no future,
so a gap shorter than `max_gap` holds the last valid sample and the median is
trailing. For the newest frame - the only one a live consumer reads - the
offline median window is already `[i - radius, i]`, so the trailing median here
reproduces exactly the value the live path used before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detector import DetectorConfig, FallState
from features import (
    frame_bbox_aspect,
    frame_body_height,
    frame_head,
    frame_hip,
    frame_shoulder,
    frame_torso_angle,
    resolve_baseline_height,
)


@dataclass(frozen=True)
class StreamingUpdate:
    """Everything a caller needs for one frame: decision plus the evidence."""

    state: FallState
    fall_latched: bool
    calibrating: bool
    detected: bool
    score: float
    downward_speed: float
    drop: float
    torso_angle: float
    bbox_aspect: float
    low_hip: float


class _Ring:
    """Fixed-capacity float ring buffer with O(1) append and lookback."""

    def __init__(self, capacity: int) -> None:
        self._data = np.full(max(capacity, 1), np.nan, dtype=np.float32)
        self._count = 0

    def append(self, value: float) -> None:
        self._data[self._count % self._data.size] = value
        self._count += 1

    def lookback(self, frames: int) -> float:
        """Value `frames` frames before the newest one; NaN when out of history."""
        if frames < 0 or frames >= min(self._count, self._data.size):
            return float("nan")
        return float(self._data[(self._count - 1 - frames) % self._data.size])

    def tail(self, count: int) -> np.ndarray:
        span = min(count, self._count, self._data.size)
        if span <= 0:
            return np.empty(0, dtype=np.float32)
        indices = (self._count - span + np.arange(span)) % self._data.size
        return self._data[indices]


class _CausalSeries:
    """Short-gap hold plus trailing median, evaluated one sample at a time."""

    def __init__(self, capacity: int, max_gap: int, radius: int) -> None:
        self._max_gap = max_gap
        self._radius = radius
        self._filled = _Ring(capacity)
        self.smooth = _Ring(capacity)
        self._last_valid = float("nan")
        self._since_valid = 0

    def push(self, raw: float) -> float:
        if np.isfinite(raw):
            self._last_valid = float(raw)
            self._since_valid = 0
            filled = float(raw)
        else:
            self._since_valid += 1
            filled = self._last_valid if self._since_valid <= self._max_gap else float("nan")
        self._filled.append(filled)

        window = self._filled.tail(self._radius + 1)
        finite = window[np.isfinite(window)]
        smoothed = float(np.median(finite)) if finite.size else float("nan")
        self.smooth.append(smoothed)
        return smoothed


def _finite_max(*values: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite, default=0.0)


class StreamingFallDetector:
    """Frame-by-frame fall detection with constant cost and a session baseline.

    Args:
        fps: processing rate the feature windows are sized for. Speeds stay in
            body-heights per second, so a threshold tuned at 10 FPS keeps its
            meaning here.
        frame_height: input height, used only as the fallback body height.
        config: the same thresholds the offline detector uses.
        calibration_seconds: how long the person must be visible before a
            baseline is fixed. No fall is reported before that.
        auto_clear_seconds: 0 keeps FALLEN latched until `reset_alarm()`, which
            matches the offline semantics. A positive value clears the latch
            after the person has been continuously upright for that long.
    """

    def __init__(
        self,
        fps: float,
        frame_height: int,
        config: DetectorConfig | None = None,
        confidence_threshold: float = 0.2,
        calibration_seconds: float = 2.0,
        auto_clear_seconds: float = 0.0,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        self.fps = float(fps)
        self.frame_height = int(frame_height)
        self.config = config or DetectorConfig()
        self.confidence_threshold = confidence_threshold

        self._speed_window = max(1, round(fps * 0.4))
        self._drop_window = max(1, round(fps * 1.0))
        self._max_gap = max(1, round(fps * 0.3))
        self._radius = max(1, round(fps * 0.15))
        self._evidence_window = max(1, round(fps * self.config.evidence_window_seconds))
        self._confirm_frames = max(1, round(fps * self.config.confirmation_seconds))
        self._missing_frames = max(1, round(fps * self.config.missing_confirmation_seconds))
        self._calibration_frames = max(1, round(fps * calibration_seconds))
        self._auto_clear_frames = round(fps * auto_clear_seconds) if auto_clear_seconds > 0 else 0

        # Only the drop window is ever read back, plus a little slack.
        capacity = max(self._drop_window, self._speed_window, self._radius) + 2
        self._head = _CausalSeries(capacity, self._max_gap, self._radius)
        self._hip = _CausalSeries(capacity, self._max_gap, self._radius)
        self._angle = _CausalSeries(capacity, self._max_gap, self._radius)
        self._aspect = _CausalSeries(capacity, self._max_gap, self._radius)

        self._calibration_heights: list[float] = []
        self._calibration_hips: list[float] = []
        self.baseline_body_height = float("nan")
        self.baseline_hip_y = float("nan")

        self._index = -1
        self._active_until = -1
        self._candidate = 0
        self._missing = 0
        self._upright = 0
        self._state = FallState.UNKNOWN
        self._latched = False
        self.first_falling_frame: int | None = None
        self.first_fallen_frame: int | None = None

    @property
    def calibrated(self) -> bool:
        return np.isfinite(self.baseline_body_height)

    @property
    def state(self) -> FallState:
        return self._state

    @property
    def fall_latched(self) -> bool:
        return self._latched

    def reset_alarm(self) -> None:
        """Acknowledge a fall: drop the latch but keep the calibrated baseline."""
        self._latched = False
        self._state = FallState.NORMAL
        self._active_until = -1
        self._candidate = 0
        self._missing = 0
        self._upright = 0

    def update(self, keypoints: np.ndarray) -> StreamingUpdate:
        """Consume one `(25, 3)` BODY_25 frame and return the current decision."""
        if keypoints.shape != (25, 3):
            raise ValueError(f"Expected one frame of shape (25, 3), got {keypoints.shape}")
        self._index += 1
        threshold = self.confidence_threshold
        detected = bool(np.any(keypoints[:, 2] >= threshold))

        head_y = self._head.push(float(frame_head(keypoints, threshold)[1]))
        hip_y = self._hip.push(float(frame_hip(keypoints, threshold)[1]))
        shoulder = frame_shoulder(keypoints, threshold)
        angle = self._angle.push(frame_torso_angle(shoulder, frame_hip(keypoints, threshold)))
        aspect = self._aspect.push(frame_bbox_aspect(keypoints, threshold))

        if not self.calibrated:
            self._collect_baseline(keypoints, hip_y, threshold)
            return self._report(detected, 0.0, 0.0, angle, aspect, 0.0)

        speed = _finite_max(
            self._series_speed(self._hip), self._series_speed(self._head)
        )
        drop = _finite_max(
            self._series_drop(self._hip), self._series_drop(self._head)
        )
        low_hip = (
            (hip_y - self.baseline_hip_y) / self.baseline_body_height
            if np.isfinite(hip_y)
            else 0.0
        )
        angle_value = float(angle) if np.isfinite(angle) else 0.0
        aspect_value = float(aspect) if np.isfinite(aspect) else 0.0

        self._advance(speed, drop, angle_value, aspect_value, low_hip, detected)
        return self._report(detected, speed, drop, angle_value, aspect_value, low_hip)

    def _collect_baseline(self, keypoints: np.ndarray, hip_y: float, threshold: float) -> None:
        height = frame_body_height(keypoints, threshold)
        if np.isfinite(height):
            self._calibration_heights.append(float(height))
        if np.isfinite(hip_y):
            self._calibration_hips.append(float(hip_y))
        if self._index + 1 < self._calibration_frames:
            return
        # Require a person to have been seen; otherwise keep waiting rather than
        # freezing a fallback baseline nothing was measured against.
        if not self._calibration_heights:
            return
        self.baseline_body_height = resolve_baseline_height(
            self._calibration_heights, self.frame_height
        )
        self.baseline_hip_y = (
            float(np.median(self._calibration_hips))
            if self._calibration_hips
            else self.frame_height * 0.5
        )
        self._state = FallState.NORMAL

    def _series_speed(self, series: _CausalSeries) -> float:
        change = self._series_change(series, self._speed_window)
        return change * self.fps / self._speed_window

    def _series_drop(self, series: _CausalSeries) -> float:
        return self._series_change(series, self._drop_window)

    def _series_change(self, series: _CausalSeries, window: int) -> float:
        now = series.smooth.lookback(0)
        before = series.smooth.lookback(window)
        if not (np.isfinite(now) and np.isfinite(before)):
            return float("nan")
        return (now - before) / self.baseline_body_height

    def _advance(
        self,
        speed: float,
        drop: float,
        angle: float,
        aspect: float,
        low_hip: float,
        detected: bool,
    ) -> None:
        """Same transitions as `detector.detect_fall`, carried across frames."""
        cfg = self.config
        index = self._index
        motion = speed >= cfg.min_downward_speed and drop >= cfg.min_drop
        horizontal = angle >= cfg.min_torso_angle or aspect >= cfg.min_bbox_aspect
        low = low_hip >= cfg.min_low_hip

        if motion:
            self._active_until = index + self._evidence_window
            if self.first_falling_frame is None:
                self.first_falling_frame = index
            if self._state != FallState.FALLEN:
                self._state = FallState.FALLING

        has_recent_motion = index <= self._active_until
        if has_recent_motion and detected:
            self._missing = 0
            if horizontal and low:
                self._candidate += 1
                if self._state != FallState.FALLEN:
                    self._state = FallState.FALL_CANDIDATE
            elif self._state != FallState.FALLEN:
                self._candidate = max(0, self._candidate - 1)
        elif has_recent_motion and not detected:
            self._missing += 1
        elif self._state != FallState.FALLEN:
            self._state = FallState.NORMAL if detected else FallState.UNKNOWN
            self._candidate = 0
            self._missing = 0

        confirmed = (
            self._candidate >= self._confirm_frames or self._missing >= self._missing_frames
        )
        if has_recent_motion and confirmed:
            self._state = FallState.FALLEN
            self._latched = True
            if self.first_fallen_frame is None:
                self.first_fallen_frame = index

        if self._auto_clear_frames and self._latched:
            if detected and not horizontal and not low:
                self._upright += 1
                if self._upright >= self._auto_clear_frames:
                    self.reset_alarm()
            else:
                self._upright = 0

    def _score(
        self, speed: float, drop: float, angle: float, aspect: float, low_hip: float
    ) -> float:
        cfg = self.config
        return float(
            min(speed / cfg.min_downward_speed, 2.0) * 0.35
            + min(drop / cfg.min_drop, 2.0) * 0.25
            + min(angle / cfg.min_torso_angle, 2.0) * 0.15
            + min(aspect / cfg.min_bbox_aspect, 2.0) * 0.10
            + min(max(low_hip, 0.0) / cfg.min_low_hip, 2.0) * 0.15
        )

    def _report(
        self,
        detected: bool,
        speed: float,
        drop: float,
        angle: float,
        aspect: float,
        low_hip: float,
    ) -> StreamingUpdate:
        angle_value = float(angle) if np.isfinite(angle) else 0.0
        aspect_value = float(aspect) if np.isfinite(aspect) else 0.0
        return StreamingUpdate(
            state=self._state,
            fall_latched=self._latched,
            calibrating=not self.calibrated,
            detected=detected,
            score=self._score(speed, drop, angle_value, aspect_value, low_hip)
            if self.calibrated
            else 0.0,
            downward_speed=speed,
            drop=drop,
            torso_angle=angle_value,
            bbox_aspect=aspect_value,
            low_hip=low_hip,
        )
