"""Machine-readable status output shared by live fall-detection runners."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from .detector import FallState
except ImportError:
    from detector import FallState


def build_status_payload(
    state: FallState,
    frame: int,
    fps: float,
    inference_ms: float,
    reason: str,
    fall_probability: float | None = None,
    latched: bool = False,
    calibrating: bool = False,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build the stable JSON schema without performing any I/O."""
    return {
        "schema": 1,
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "state": state.name,
        "fall_detected": state == FallState.FALLEN or latched,
        "fall_latched": latched,
        "calibrating": calibrating,
        "frame": int(frame),
        "fps": round(float(fps), 3),
        "inference_ms": round(float(inference_ms), 3),
        "reason": reason,
        "fall_probability": (
            round(float(fall_probability), 4) if fall_probability is not None else None
        ),
    }


def publish_status(
    path: Path | None,
    state: FallState,
    frame: int,
    fps: float,
    inference_ms: float,
    reason: str,
    fall_probability: float | None = None,
    latched: bool = False,
    calibrating: bool = False,
) -> dict[str, Any]:
    """Print status and atomically replace the optional JSON status file."""
    payload = build_status_payload(
        state=state,
        frame=frame,
        fps=fps,
        inference_ms=inference_ms,
        reason=reason,
        fall_probability=fall_probability,
        latched=latched,
        calibrating=calibrating,
    )
    compact = json.dumps(payload, separators=(",", ":"))
    print("FALL_STATUS " + compact, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    return payload
