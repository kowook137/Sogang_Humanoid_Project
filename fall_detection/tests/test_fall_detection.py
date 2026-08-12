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
from yolo_pose import coco_to_body25, resolve_device  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
