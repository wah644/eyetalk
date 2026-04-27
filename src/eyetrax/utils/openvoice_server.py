from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyetrax.utils.openvoice_engine import OpenVoiceEngine


ENGINE: OpenVoiceEngine | None = None


class OpenVoiceHandler(BaseHTTPRequestHandler):
    server_version = "OpenVoiceServer/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        self._send_json(200, {"ok": True, "status": "ready"})

    def do_POST(self) -> None:
        global ENGINE
        if ENGINE is None:
            self._send_json(503, {"ok": False, "error": "engine not initialized"})
            return

        try:
            data = self._read_json()
            if self.path == "/warm":
                ENGINE.warm(
                    language=str(data.get("language", "en")).lower(),
                    speaker_wav=data.get("speaker_wav"),
                )
                self._send_json(200, {"ok": True, "status": "warmed"})
                return

            if self.path == "/synthesize":
                output_path = ENGINE.synthesize(
                    text=str(data["text"]),
                    language=str(data.get("language", "en")).lower(),
                    speaker_wav=data["speaker_wav"],
                    output_path=data["output_path"],
                    refresh_speaker=bool(data.get("refresh_speaker", False)),
                    play_audio=bool(data.get("play_audio", False)),
                )
                self._send_json(200, {"ok": True, "output_path": str(output_path)})
                return

            self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        sys.stdout.write(f"[openvoice-server] {format % args}\n")
        sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a warm OpenVoice HTTP server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--openvoice-repo", required=True, help="Path to OpenVoice repo")
    parser.add_argument("--work-dir", required=True, help="Runtime/cache directory")
    parser.add_argument("--language", default="en", help="Warm this language on startup")
    parser.add_argument("--speaker-wav", help="Warm this speaker embedding on startup")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--skip-warm", action="store_true", help="Start server without warming models")
    return parser


def main() -> int:
    global ENGINE
    args = build_parser().parse_args()
    ENGINE = OpenVoiceEngine(
        openvoice_repo=Path(args.openvoice_repo),
        work_dir=Path(args.work_dir),
        use_gpu=bool(args.use_gpu),
    )
    if not args.skip_warm:
        ENGINE.warm(language=args.language, speaker_wav=args.speaker_wav)
        print("[openvoice-server] Warmed models and speaker cache.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), OpenVoiceHandler)
    print(f"[openvoice-server] Listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[openvoice-server] Shutting down.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
