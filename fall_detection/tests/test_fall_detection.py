from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from detector import FallState, detect_fall  # noqa: E402
from features import extract_features  # noqa: E402
from openpose_runtime import find_openpose  # noqa: E402
from process_video import select_person  # noqa: E402
from evaluate import finite_max  # noqa: E402
from streaming import StreamingFallDetector  # noqa: E402
from train_gmdcsa24 import resample  # noqa: E402
from yolo_pose import coco_to_body25, resolve_device  # noqa: E402


def falling_frame(index: int, fall_start: int, fall_frames: int) -> np.ndarray:
    """One BODY_25 frame of a person standing, then toppling to the floor."""
    progress = min(max((index - fall_start) / fall_frames, 0.0), 1.0)
    frame = np.full((25, 3), np.nan, dtype=np.float32)
    frame[:, 2] = 0.0
    hip = np.asarray([320, 260 + 130 * progress])
    shoulder = np.asarray([320 + 100 * progress, 150 + 210 * progress])
    nose = np.asarray([320 + 130 * progress, 80 + 290 * progress])
    points = {
        0: nose, 1: shoulder, 2: shoulder + (-20, 0), 5: shoulder + (20, 0),
        8: hip, 9: hip + (-20, 0), 12: hip + (20, 0),
        10: hip + (-20, 70 * (1 - progress) + 10),
        13: hip + (20, 70 * (1 - progress) + 10),
        11: hip + (-20, 140 * (1 - progress) + 15),
        14: hip + (20, 140 * (1 - progress) + 15),
    }
    for joint, xy in points.items():
        frame[joint, :2] = xy
        frame[joint, 2] = 1.0
    return frame


class OpenPoseRuntimeTests(unittest.TestCase):
    def test_detects_linux_source_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "openpose/build/examples/openpose/openpose.bin"
            model = root / "openpose/models/pose/body_25/pose_iter_584000.caffemodel"
            executable.parent.mkdir(parents=True)
            model.parent.mkdir(parents=True)
            executable.touch()
            model.touch()

            runtime = find_openpose(root)

            self.assertEqual(runtime.executable, executable)
            self.assertEqual(runtime.model_dir, root / "openpose/models")

    def test_reports_searched_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "openpose.bin"):
                find_openpose(Path(directory))


class KeypointTests(unittest.TestCase):
    def test_resample_rejects_empty_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty sequence"):
            resample(np.empty((0, 3), dtype=np.float32), frames=10)

    def test_resample_rejects_nonpositive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            resample(np.ones((2, 3), dtype=np.float32), frames=0)

    def test_missing_person_uses_nan_coordinates(self) -> None:
        frame = select_person([])
        self.assertEqual(frame.shape, (25, 3))
        self.assertTrue(np.isnan(frame[:, :2]).all())
        self.assertTrue((frame[:, 2] == 0).all())

    def test_static_pose_has_zero_drop_after_warmup(self) -> None:
        keypoints = np.zeros((40, 25, 3), dtype=np.float32)
        keypoints[:, :, 0] = np.linspace(100, 200, 25)
        keypoints[:, :, 1] = np.linspace(50, 350, 25)
        keypoints[:, :, 2] = 1.0

        features = extract_features(keypoints, fps=10.0, frame_width=640, frame_height=480)

        self.assertAlmostEqual(float(np.nanmax(np.abs(features.hip_speed))), 0.0)
        self.assertAlmostEqual(float(np.nanmax(np.abs(features.head_speed))), 0.0)

    def test_feature_baseline_can_be_frozen_for_streaming(self) -> None:
        keypoints = np.zeros((30, 25, 3), dtype=np.float32)
        keypoints[:, :, 0] = np.linspace(100, 200, 25)
        keypoints[:, :, 1] = np.linspace(50, 350, 25)
        keypoints[:, :, 2] = 1.0

        features = extract_features(
            keypoints,
            fps=10.0,
            frame_width=640,
            frame_height=480,
            baseline_body_height=321.0,
            baseline_hip_y=222.0,
        )

        self.assertEqual(features.baseline_body_height, 321.0)
        self.assertEqual(features.baseline_hip_y, 222.0)

    def test_coco_pose_maps_centers_to_body25(self) -> None:
        xy = np.arange(34, dtype=np.float32).reshape(17, 2)
        confidence = np.ones(17, dtype=np.float32)
        body25 = coco_to_body25(xy, confidence)

        np.testing.assert_allclose(body25[1, :2], np.mean(xy[[5, 6]], axis=0))
        np.testing.assert_allclose(body25[8, :2], np.mean(xy[[11, 12]], axis=0))
        np.testing.assert_allclose(body25[0, :2], xy[0])

    def test_explicit_cpu_device(self) -> None:
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Device must be"):
            resolve_device("gpu")

    def test_synthetic_fall_reaches_fallen_state(self) -> None:
        keypoints = np.full((60, 25, 3), np.nan, dtype=np.float32)
        keypoints[:, :, 2] = 0.0
        for frame_index in range(60):
            progress = min(max((frame_index - 20) / 10, 0.0), 1.0)
            hip = np.asarray([320, 260 + 130 * progress])
            shoulder = np.asarray([320 + 100 * progress, 150 + 210 * progress])
            nose = np.asarray([320 + 130 * progress, 80 + 290 * progress])
            points = {
                0: nose,
                1: shoulder,
                2: shoulder + (-20, 0),
                5: shoulder + (20, 0),
                8: hip,
                9: hip + (-20, 0),
                12: hip + (20, 0),
                10: hip + (-20, 70 * (1 - progress) + 10),
                13: hip + (20, 70 * (1 - progress) + 10),
                11: hip + (-20, 140 * (1 - progress) + 15),
                14: hip + (20, 140 * (1 - progress) + 15),
            }
            for joint, xy in points.items():
                keypoints[frame_index, joint, :2] = xy
                keypoints[frame_index, joint, 2] = 1.0

        result = detect_fall(extract_features(keypoints, 10.0, 640, 480))

        self.assertTrue(result.fall_detected)
        self.assertEqual(FallState(int(result.states[-1])), FallState.FALLEN)


class EvaluationTests(unittest.TestCase):
    def test_finite_max_survives_a_video_with_no_detections(self) -> None:
        self.assertEqual(finite_max(np.full(5, np.nan, dtype=np.float32)), 0.0)

    def test_finite_max_ignores_nan(self) -> None:
        self.assertAlmostEqual(finite_max(np.asarray([np.nan, 0.3, 1.7])), 1.7)


class StreamingDetectorTests(unittest.TestCase):
    def test_synthetic_fall_reaches_fallen_state(self) -> None:
        detector = StreamingFallDetector(fps=10.0, frame_height=480)
        for index in range(60):
            detector.update(falling_frame(index, fall_start=20, fall_frames=10))

        self.assertEqual(detector.state, FallState.FALLEN)
        self.assertTrue(detector.fall_latched)

    def test_alarm_survives_the_fall_leaving_the_feature_window(self) -> None:
        """A motionless person on the floor must stay reported as FALLEN.

        Re-running the offline detector over a sliding window used to clear the
        alarm as soon as the falling motion scrolled out of the buffer.
        """
        detector = StreamingFallDetector(fps=15.0, frame_height=480)
        for index in range(600):  # 40 s: fall at 5 s, motionless afterwards
            detector.update(falling_frame(index, fall_start=75, fall_frames=15))

        self.assertEqual(detector.state, FallState.FALLEN)
        self.assertLess(detector.first_fallen_frame, 150)

    def test_baseline_is_frozen_after_calibration(self) -> None:
        """The standing reference must not drift onto the collapsed pose."""
        detector = StreamingFallDetector(fps=15.0, frame_height=480)
        for index in range(40):
            detector.update(falling_frame(index, fall_start=1000, fall_frames=15))
        calibrated_height = detector.baseline_body_height
        calibrated_hip = detector.baseline_hip_y
        for index in range(40, 600):
            detector.update(falling_frame(index, fall_start=75, fall_frames=15))

        self.assertEqual(detector.baseline_body_height, calibrated_height)
        self.assertEqual(detector.baseline_hip_y, calibrated_hip)

    def test_reset_alarm_clears_the_latch(self) -> None:
        detector = StreamingFallDetector(fps=10.0, frame_height=480)
        for index in range(60):
            detector.update(falling_frame(index, fall_start=20, fall_frames=10))
        detector.reset_alarm()

        self.assertFalse(detector.fall_latched)
        self.assertEqual(detector.state, FallState.NORMAL)

    def test_no_fall_is_reported_before_calibration_completes(self) -> None:
        detector = StreamingFallDetector(fps=10.0, frame_height=480)
        update = detector.update(falling_frame(0, fall_start=20, fall_frames=10))

        self.assertTrue(update.calibrating)
        self.assertFalse(update.fall_latched)

    def test_history_stays_bounded_over_a_long_session(self) -> None:
        detector = StreamingFallDetector(fps=30.0, frame_height=480)
        standing = falling_frame(0, fall_start=10_000, fall_frames=15)
        for _ in range(5_000):
            detector.update(standing)

        # Only the drop window is ever read back, so memory must not track runtime.
        self.assertLessEqual(detector._hip.smooth._data.size, 64)

    def test_fall_hold_clears_the_latch_after_the_configured_time(self) -> None:
        """Time-based release, for a consumer that must eventually see it end."""
        detector = StreamingFallDetector(fps=10.0, frame_height=480, fall_hold_seconds=5.0)
        for index in range(60):  # fall at 2 s, confirmed well before the hold ends
            detector.update(falling_frame(index, fall_start=20, fall_frames=10))
        self.assertTrue(detector.fall_latched)
        for index in range(60, 200):  # motionless on the floor past the 5 s hold
            detector.update(falling_frame(index, fall_start=20, fall_frames=10))

        self.assertFalse(detector.fall_latched)

    def test_fall_hold_of_zero_keeps_the_alarm_for_the_session(self) -> None:
        detector = StreamingFallDetector(fps=10.0, frame_height=480, fall_hold_seconds=0.0)
        for index in range(600):
            detector.update(falling_frame(index, fall_start=20, fall_frames=10))

        self.assertTrue(detector.fall_latched)
        self.assertEqual(detector.state, FallState.FALLEN)

    def test_force_fallen_raises_the_alarm_through_the_same_latch(self) -> None:
        """The classifier path must not report FALLEN behind the detector's back."""
        detector = StreamingFallDetector(fps=10.0, frame_height=480)
        standing = falling_frame(0, fall_start=10_000, fall_frames=10)
        for _ in range(40):
            detector.update(standing)
        self.assertEqual(detector.state, FallState.NORMAL)
        detector.force_fallen()

        self.assertEqual(detector.state, FallState.FALLEN)
        self.assertTrue(detector.fall_latched)
        self.assertEqual(detector.update(standing).state, FallState.FALLEN)

    def test_rejects_a_batch_instead_of_a_single_frame(self) -> None:
        detector = StreamingFallDetector(fps=10.0, frame_height=480)
        with self.assertRaisesRegex(ValueError, r"\(25, 3\)"):
            detector.update(np.zeros((5, 25, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
