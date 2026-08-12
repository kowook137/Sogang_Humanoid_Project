"""Interpretable state-machine baseline for fall detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from features import PoseFeatures


class FallState(IntEnum):
    NORMAL = 0
    FALLING = 1
    FALL_CANDIDATE = 2
    FALLEN = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class DetectorConfig:
    min_downward_speed: float = 0.55
    min_drop: float = 0.30
    min_torso_angle: float = 48.0
    min_bbox_aspect: float = 0.85
    min_low_hip: float = 0.28
    evidence_window_seconds: float = 1.5
    confirmation_seconds: float = 0.5
    missing_confirmation_seconds: float = 0.5


@dataclass(frozen=True)
class DetectionResult:
    states: np.ndarray
    fall_score: np.ndarray
    fall_detected: bool
    first_falling_frame: int | None
    first_fallen_frame: int | None


def _finite_max(*values: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite, default=0.0)


def detect_fall(features: PoseFeatures, config: DetectorConfig | None = None) -> DetectionResult:
    cfg = config or DetectorConfig()
    count = len(features.detected)
    states = np.full(count, FallState.NORMAL, dtype=np.int8)
    scores = np.zeros(count, dtype=np.float32)
    evidence_window = max(1, round(features.fps * cfg.evidence_window_seconds))
    confirmation_frames = max(1, round(features.fps * cfg.confirmation_seconds))
    missing_frames_required = max(1, round(features.fps * cfg.missing_confirmation_seconds))

    active_until = -1
    candidate_count = 0
    missing_count = 0
    first_falling: int | None = None
    first_fallen: int | None = None
    current = FallState.NORMAL

    for index in range(count):
        speed = _finite_max(features.hip_speed[index], features.head_speed[index])
        drop = _finite_max(features.hip_drop[index], features.head_drop[index])
        angle = float(features.torso_angle[index]) if np.isfinite(features.torso_angle[index]) else 0.0
        aspect = float(features.bbox_aspect[index]) if np.isfinite(features.bbox_aspect[index]) else 0.0
        low_hip = (
            (features.hip_y[index] - features.baseline_hip_y) / features.baseline_body_height
            if np.isfinite(features.hip_y[index])
            else 0.0
        )

        motion = speed >= cfg.min_downward_speed and drop >= cfg.min_drop
        horizontal = angle >= cfg.min_torso_angle or aspect >= cfg.min_bbox_aspect
        low = low_hip >= cfg.min_low_hip
        scores[index] = (
            min(speed / cfg.min_downward_speed, 2.0) * 0.35
            + min(drop / cfg.min_drop, 2.0) * 0.25
            + min(angle / cfg.min_torso_angle, 2.0) * 0.15
            + min(aspect / cfg.min_bbox_aspect, 2.0) * 0.10
            + min(max(low_hip, 0.0) / cfg.min_low_hip, 2.0) * 0.15
        )

        if motion:
            active_until = index + evidence_window
            if first_falling is None:
                first_falling = index
            if current != FallState.FALLEN:
                current = FallState.FALLING

        has_recent_motion = index <= active_until
        if has_recent_motion and features.detected[index]:
            missing_count = 0
            if horizontal and low:
                candidate_count += 1
                if current != FallState.FALLEN:
                    current = FallState.FALL_CANDIDATE
            elif current != FallState.FALLEN:
                candidate_count = max(0, candidate_count - 1)
        elif has_recent_motion and not features.detected[index]:
            missing_count += 1
        elif current != FallState.FALLEN:
            current = FallState.NORMAL if features.detected[index] else FallState.UNKNOWN
            candidate_count = 0
            missing_count = 0

        confirmed_by_pose = candidate_count >= confirmation_frames
        confirmed_by_loss = missing_count >= missing_frames_required
        if has_recent_motion and (confirmed_by_pose or confirmed_by_loss):
            current = FallState.FALLEN
            if first_fallen is None:
                first_fallen = index

        states[index] = int(current)

    return DetectionResult(
        states=states,
        fall_score=scores,
        fall_detected=first_fallen is not None,
        first_falling_frame=first_falling,
        first_fallen_frame=first_fallen,
    )
