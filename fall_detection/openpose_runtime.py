"""Locate and launch either a Linux build or Windows portable OpenPose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenPoseRuntime:
    executable: Path
    model_dir: Path
    working_dir: Path


def _candidate_roots(project_root: Path, requested_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if requested_root is not None:
        roots.append(requested_root.expanduser().resolve())
    roots.extend(
        (
            project_root / "openpose",
            project_root / "openpose" / "openpose-portable" / "openpose",
        )
    )
    return list(dict.fromkeys(path.resolve() for path in roots))


def find_openpose(project_root: Path, requested_root: Path | None = None) -> OpenPoseRuntime:
    """Find a repository Linux build or a Windows portable installation."""
    searched: list[Path] = []
    for root in _candidate_roots(project_root, requested_root):
        candidates = (
            (root / "build" / "examples" / "openpose" / "openpose.bin", root / "models", root),
            (root / "examples" / "openpose" / "openpose.bin", root.parent / "models", root.parent),
            (root / "bin" / "OpenPoseDemo.exe", root / "models", root),
        )
        for executable, model_dir, working_dir in candidates:
            searched.append(executable)
            if not executable.is_file():
                continue
            body_model = model_dir / "pose" / "body_25" / "pose_iter_584000.caffemodel"
            if not body_model.is_file():
                raise FileNotFoundError(f"BODY_25 model not found: {body_model}")
            return OpenPoseRuntime(executable, model_dir, working_dir)

    locations = "\n  - ".join(str(path) for path in searched)
    raise FileNotFoundError(f"OpenPose executable not found. Searched:\n  - {locations}")
