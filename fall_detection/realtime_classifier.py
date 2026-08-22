"""Small GRU classifier adapter for a rolling real-time pose window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from .train_gmdcsa24 import (
        FallGRU,
        engineered_features_from_keypoints,
        pose_features_from_keypoints,
        resample,
    )
except ImportError:
    from train_gmdcsa24 import (
        FallGRU,
        engineered_features_from_keypoints,
        pose_features_from_keypoints,
        resample,
    )


class RealtimeFallClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.feature_mode = str(checkpoint.get("feature_mode", "pose"))
        expected_size = 13 if self.feature_mode == "engineered" else 75
        if int(checkpoint["input_size"]) != expected_size:
            raise ValueError("Classifier checkpoint feature metadata is inconsistent")
        self.frames = int(checkpoint["frames"])
        self.model = FallGRU(input_size=expected_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def probability(self, keypoints: np.ndarray) -> float:
        sequence = (
            engineered_features_from_keypoints(keypoints)
            if self.feature_mode == "engineered"
            else pose_features_from_keypoints(keypoints)
        )
        inputs = torch.from_numpy(resample(sequence, self.frames)).unsqueeze(0).to(self.device)
        return float(torch.softmax(self.model(inputs), dim=1)[0, 1])
