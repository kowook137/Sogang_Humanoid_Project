"""Download and extract the 70 camera-0 RGB sequences from the official UR Fall site."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = project_root() / "data/datasets/urfall"
    archives = root / "archives"
    sequences = root / "rgb_sequences"
    archives.mkdir(parents=True, exist_ok=True)
    sequences.mkdir(parents=True, exist_ok=True)
    names = [f"fall-{index:02d}-cam0-rgb" for index in range(1, 31)]
    names += [f"adl-{index:02d}-cam0-rgb" for index in range(1, 41)]
    def fetch(name: str) -> str:
        archive = archives / f"{name}.zip"
        if archive.is_file() and subprocess.run(
            ["unzip", "-tq", str(archive)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0:
            return name
        subprocess.run(
            ["curl", "-fsSL", "--continue-at", "-", "--output", str(archive), f"{BASE_URL}/{name}.zip"],
            check=True,
        )
        subprocess.run(["unzip", "-tq", str(archive)], check=True, stdout=subprocess.DEVNULL)
        return name

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, name): name for name in names}
        for completed, future in enumerate(as_completed(futures), start=1):
            print(f"downloaded={completed}/70 {future.result()}", flush=True)

    for index, name in enumerate(names, start=1):
        archive = archives / f"{name}.zip"
        destination = sequences / name
        if not destination.is_dir() or not any(destination.rglob("*.png")):
            print(f"[{index:02d}/70] extract {name}", flush=True)
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(destination)], check=True)
    image_count = sum(1 for path in sequences.rglob("*.png"))
    print(f"done sequences={len(names)} images={image_count} output={sequences}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
