"""Shared on-screen presentation for the live fall-detection paths.

Kept apart from `detector.py` and `streaming.py` on purpose: those decide what
is happening, this only decides how it looks. `yolo_pose.py` draws over the
camera frame, `live_detect.py` draws a standalone dashboard, and both read the
same colours from here so one state never shows up green in one window and red
in the other.
"""

from __future__ import annotations

import cv2
import numpy as np

from detector import FallState


# BGR, dark-background friendly.
STATE_COLORS: dict[FallState, tuple[int, int, int]] = {
    FallState.NORMAL: (0, 200, 0),
    FallState.FALLING: (0, 200, 255),
    FallState.FALL_CANDIDATE: (0, 120, 255),
    FallState.FALLEN: (0, 0, 255),
    FallState.UNKNOWN: (160, 160, 160),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CALIBRATING_COLOR = (200, 200, 200)


def state_color(state: FallState, latched: bool = False) -> tuple[int, int, int]:
    """Colour for a state; a latched alarm stays red whatever the state reads."""
    if latched:
        return STATE_COLORS[FallState.FALLEN]
    return STATE_COLORS.get(state, STATE_COLORS[FallState.UNKNOWN])


def gui_available() -> bool:
    """Whether this OpenCV build can open a window at all.

    The `opencv-python-headless` wheel - the sensible default for a robot, and
    what JetPack setups often end up with - raises from `imshow` instead of
    saying so up front. Ask the build info rather than finding out mid-stream.
    """
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip().upper() not in {"NONE", ""}
    return False


def draw_status_overlay(
    frame: np.ndarray,
    state: FallState,
    latched: bool,
    calibrating: bool,
    fps: float,
    inference_ms: float,
    fall_probability: float | None = None,
) -> np.ndarray:
    """Draw the live status panel onto `frame` in place and return it.

    Readable from across a room: the state name is the largest element, and a
    latched alarm also paints a border so it is visible without reading text.
    """
    height, width = frame.shape[:2]
    label = "CALIBRATING" if calibrating else state.name
    color = _CALIBRATING_COLOR if calibrating else state_color(state, latched)

    panel_height = 78
    panel = frame[0:panel_height, 0:width]
    cv2.addWeighted(np.zeros_like(panel), 0.45, panel, 0.55, 0, panel)

    cv2.putText(frame, label, (18, 46), _FONT, 1.1, color, 2, cv2.LINE_AA)
    if latched:
        cv2.putText(frame, "FALL", (width - 118, 46), _FONT, 1.1, color, 2, cv2.LINE_AA)

    details = f"{fps:.1f} FPS   inference {inference_ms:.0f} ms"
    if fall_probability is not None:
        details += f"   p(fall) {fall_probability:.2f}"
    cv2.putText(frame, details, (18, 68), _FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q / ESC: quit    R: reset alarm", (18, height - 14), _FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)

    if latched:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 4)
    return frame
