"""Feature extraction for BODY_25 fall detection.

All displacement and velocity features are normalized by the person's initial
body height. Velocities are expressed per second, so thresholds do not depend
on the input frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BODY_JOINTS = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
NOSE = 0
NECK = 1
R_SHOULDER = 2
L_SHOULDER = 5
MID_HIP = 8
R_HIP = 9
L_HIP = 12


@dataclass(frozen=True)
class PoseFeatures:
    fps: float
    frame_width: int
    frame_height: int
    detected: np.ndarray
    head_y: np.ndarray
    hip_y: np.ndarray
    torso_angle: np.ndarray
    bbox_aspect: np.ndarray
    hip_drop: np.ndarray
    head_drop: np.ndarray
    hip_speed: np.ndarray
    head_speed: np.ndarray
    baseline_body_height: float
    baseline_hip_y: float


def _weighted_point(frame: np.ndarray, indices: tuple[int, ...], threshold: float) -> np.ndarray:
    selected = frame[list(indices)]
    valid = np.isfinite(selected[:, 0]) & np.isfinite(selected[:, 1]) & (selected[:, 2] >= threshold)
    if not np.any(valid):
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    weights = selected[valid, 2]
    return np.average(selected[valid, :2], axis=0, weights=weights).astype(np.float32)


def _point_series(keypoints: np.ndarray, primary: tuple[int, ...], fallback: tuple[int, ...], threshold: float) -> np.ndarray:
    result = np.full((len(keypoints), 2), np.nan, dtype=np.float32)
    for index, frame in enumerate(keypoints):
        point = _weighted_point(frame, primary, threshold)
        if not np.all(np.isfinite(point)):
            point = _weighted_point(frame, fallback, threshold)
        result[index] = point
    return result


def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill only gaps bounded by valid values on both sides."""
    result = values.astype(np.float32, copy=True)
    valid_indices = np.flatnonzero(np.isfinite(result))
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            result[left + 1 : right] = np.linspace(result[left], result[right], gap + 2)[1:-1]
    return result


def rolling_median(values: np.ndarray, radius: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float32)
    for index in range(len(values)):
        chunk = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            result[index] = np.median(finite)
    return result


def trailing_change(values: np.ndarray, frames: int, scale: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float32)
    for index in range(frames, len(values)):
        if np.isfinite(values[index]) and np.isfinite(values[index - frames]):
            result[index] = (values[index] - values[index - frames]) / scale
    return result


def downward_speed(values: np.ndarray, fps: float, window: int, scale: float) -> np.ndarray:
    change = trailing_change(values, window, scale)
    return change * fps / window


def extract_features(
    keypoints: np.ndarray,
    fps: float,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float = 0.2,
) -> PoseFeatures:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (25, 3):
        raise ValueError(f"Expected keypoints shape (frames, 25, 3), got {keypoints.shape}")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    detected = np.any(keypoints[:, :, 2] >= confidence_threshold, axis=1)
    head = _point_series(keypoints, (NOSE, NECK), (R_SHOULDER, L_SHOULDER), confidence_threshold)
    hip = _point_series(keypoints, (MID_HIP,), (R_HIP, L_HIP), confidence_threshold)
    shoulder = _point_series(keypoints, (R_SHOULDER, L_SHOULDER), (NECK,), confidence_threshold)

    max_gap = max(1, round(fps * 0.3))
    radius = max(1, round(fps * 0.15))
    head_y = rolling_median(interpolate_short_gaps(head[:, 1], max_gap), radius)
    hip_y = rolling_median(interpolate_short_gaps(hip[:, 1], max_gap), radius)

    torso_angle = np.full(len(keypoints), np.nan, dtype=np.float32)
    vector = shoulder - hip
    valid_vector = np.all(np.isfinite(vector), axis=1)
    torso_angle[valid_vector] = np.degrees(
        np.arctan2(np.abs(vector[valid_vector, 0]), np.abs(vector[valid_vector, 1]))
    )
    torso_angle = rolling_median(interpolate_short_gaps(torso_angle, max_gap), radius)

    bbox_aspect = np.full(len(keypoints), np.nan, dtype=np.float32)
    body = keypoints[:, BODY_JOINTS]
    for index, frame in enumerate(body):
        valid = np.isfinite(frame[:, 0]) & np.isfinite(frame[:, 1]) & (frame[:, 2] >= confidence_threshold)
        if np.count_nonzero(valid) < 5:
            continue
        width = float(np.ptp(frame[valid, 0]))
        height = float(np.ptp(frame[valid, 1]))
        if height > 1:
            bbox_aspect[index] = width / height
    bbox_aspect = rolling_median(interpolate_short_gaps(bbox_aspect, max_gap), radius)

    baseline_count = min(len(keypoints), max(round(fps * 2.0), 1))
    baseline_indices = slice(0, baseline_count)
    initial_heights: list[float] = []
    for frame in body[baseline_indices]:
        valid = np.isfinite(frame[:, 1]) & (frame[:, 2] >= confidence_threshold)
        if np.count_nonzero(valid) >= 5:
            height = float(np.ptp(frame[valid, 1]))
            if height > 1:
                initial_heights.append(height)
    baseline_body_height = float(np.median(initial_heights)) if initial_heights else frame_height * 0.5
    baseline_body_height = max(baseline_body_height, frame_height * 0.15)

    initial_hip = hip_y[baseline_indices]
    initial_hip = initial_hip[np.isfinite(initial_hip)]
    baseline_hip_y = float(np.median(initial_hip)) if initial_hip.size else frame_height * 0.5

    speed_window = max(1, round(fps * 0.4))
    drop_window = max(1, round(fps * 1.0))
    return PoseFeatures(
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        detected=detected,
        head_y=head_y,
        hip_y=hip_y,
        torso_angle=torso_angle,
        bbox_aspect=bbox_aspect,
        hip_drop=trailing_change(hip_y, drop_window, baseline_body_height),
        head_drop=trailing_change(head_y, drop_window, baseline_body_height),
        hip_speed=downward_speed(hip_y, fps, speed_window, baseline_body_height),
        head_speed=downward_speed(head_y, fps, speed_window, baseline_body_height),
        baseline_body_height=baseline_body_height,
        baseline_hip_y=baseline_hip_y,
    )
