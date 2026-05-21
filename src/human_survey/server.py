#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
RESPONSES_ROOT = ROOT / "data" / "responses"


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in str(value or "").strip())
    return cleaned.strip("-._") or "anonymous"


class SurveyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/survey":
            suffix = f"?{parsed.query}" if parsed.query else ""
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", f"/survey/{suffix}")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/save", "/survey/api/save"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid JSON"})
            return
        participant_id = _safe_name(
            str(payload.get("prolific_pid") or payload.get("participant_id") or payload.get("userId") or "")
        )
        RESPONSES_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = RESPONSES_ROOT / f"user_{participant_id}.json"
        existing = []
        if out_path.is_file():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        payload["server_received_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        payload["remote_address"] = self.client_address[0]
        payload["user_agent"] = self.headers.get("User-Agent")
        existing.append(payload)
        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        self._send_json(HTTPStatus.OK, {"success": True, "filename": str(out_path.relative_to(ROOT))})

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def guess_type(self, path: str) -> str:
        guessed = super().guess_type(path)
        if guessed == "application/octet-stream":
            if path.endswith(".json"):
                return "application/json"
        return guessed

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        mount_prefix = "/survey/"
        if parsed.path.startswith(mount_prefix):
            stripped_path = "/" + parsed.path[len(mount_prefix) :].lstrip("/")
            if stripped_path == "/":
                stripped_path = "/index.html"
            rewritten = stripped_path
            if parsed.query:
                rewritten = f"{rewritten}?{parsed.query}"
            return super().translate_path(rewritten)
        return super().translate_path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SurveyHandler)
    print(f"Serving survey from {ROOT}")
    print(f"Visit: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
