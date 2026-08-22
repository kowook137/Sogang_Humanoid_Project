"""Download and extract all FallVision keypoint CSV archives from Harvard Dataverse."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


DOI = "doi:10.7910/DVN/75QPKK"
API = "https://dataverse.harvard.edu/api"
USER_AGENT = "Sogang-Humanoid-Fall-Research/1.0"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request) as response:
        if existing and response.status != 206:
            existing = 0
        mode = "ab" if existing else "wb"
        with destination.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(source: Path, destination: Path) -> None:
    executable = shutil.which("7z") or shutil.which("7zz")
    if executable:
        subprocess.run(
            [executable, "x", "-y", "-aoa", f"-o{destination}", str(source)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError("Install 7z or py7zr to extract FallVision archives") from error
    destination.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(source, mode="r") as archive:
        archive.extractall(path=destination)


def main() -> int:
    root = project_root() / "data/datasets/fallvision"
    archive_dir = root / "keypoints_archives"
    output_dir = root / "keypoints"
    metadata_url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    metadata_request = urllib.request.Request(metadata_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(metadata_request) as response:
        payload = json.load(response)
    version = payload["data"]["latestVersion"]
    files = []
    for entry in version["files"]:
        data = entry["dataFile"]
        name = data["filename"].lower()
        if "keypoints_csv" in name or "ketpoints_csv" in name:
            files.append(data)
    if len(files) != 20:
        raise RuntimeError(f"Expected 20 keypoint archives, found {len(files)}")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "dataverse_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    for index, data in enumerate(sorted(files, key=lambda item: item["filename"]), start=1):
        destination = archive_dir / data["filename"]
        expected_size = int(data["filesize"])
        if not destination.is_file() or destination.stat().st_size != expected_size:
            print(f"[{index:02d}/20] download {data['filename']} ({expected_size} bytes)", flush=True)
            download(f"{API}/access/datafile/{data['id']}", destination)
        checksum = data.get("md5") or data.get("checksum", {}).get("value")
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"Size mismatch: {destination}")
        if checksum and md5(destination).lower() != checksum.lower():
            raise RuntimeError(f"MD5 mismatch: {destination}")
        print(f"[{index:02d}/20] extract {data['filename']}", flush=True)
        extract_archive(destination, output_dir)
    csv_count = sum(1 for path in output_dir.rglob("*.csv") if path.stat().st_size)
    print(f"done archives={len(files)} csv_files={csv_count} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
