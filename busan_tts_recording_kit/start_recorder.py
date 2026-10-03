#!/usr/bin/env python3
"""Launch the dependency-free Busan recording tool in the default browser."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/prompts.json":
            with (ROOT / "recording_manifest.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = [
                    {"id": row["id"], "text": row["text"]}
                    for row in csv.DictReader(handle)
                ]
            payload = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path in {"/", "/index.html"}:
            self.path = "/recorder.html"
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        return


def open_browser(url: str) -> None:
    """Open the Windows browser from WSL, otherwise use the platform default."""
    if os.environ.get("WSL_DISTRO_NAME"):
        try:
            subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Start-Process", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    if not webbrowser.open(url):
        print(f"브라우저에서 직접 여세요: {url}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"녹음 도구: {url}")
    print("종료: 이 터미널에서 Ctrl+C")
    threading.Timer(0.5, open_browser, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료했습니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
