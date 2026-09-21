"""HTTP service for the night planner (Python standard library only).

Endpoints
---------
* ``GET  /health``  -- liveness/readiness probe
* ``POST /plan``    -- plan a night; body is the JSON request described in
                       README.md, response is the canonical plan plus the
                       required/optional/excluded classification

Configuration via environment variables
---------------------------------------
* ``API_HOST`` -- bind address inside the container (default ``0.0.0.0``)
* ``API_PORT`` -- bind port inside the container (default ``8080``)
* ``API_MAX_BODY_BYTES`` -- maximum request body size (default 1 MiB)
"""

from __future__ import annotations

import json
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # package import (tests)
    from .planner import PlanError, plan
except ImportError:  # run as a script with /app on sys.path (container)
    from planner import PlanError, plan

START_TIME = time.time()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


class Handler(BaseHTTPRequestHandler):
    server_version = "NightPlanner/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path.split("?", 1)[0] == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "uptime_seconds": round(time.time() - START_TIME, 3)},
            )
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if self.path.split("?", 1)[0] != "/plan":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        limit = _int_env("API_MAX_BODY_BYTES", 1 << 20)
        length = self.headers.get("Content-Length")
        if length is None:
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"}
            )
            return
        try:
            length = int(length)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return
        if length < 0 or length > limit:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "body_too_large", "limit_bytes": limit},
            )
            return

        raw = self.rfile.read(length)
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        try:
            result = plan(request)
        except PlanError as exc:
            # Illegal input is rejected as a whole; no partial plan exists.
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request",
                                                      "detail": str(exc)})
            return

        self._send_json(HTTPStatus.OK, result)

    def log_message(self, fmt, *args):  # quiet by default; count if present
        count = getattr(self.server, "request_count", None)
        if count is not None:
            self.server.request_count = count + 1


def main() -> None:
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = _int_env("API_PORT", 8080)
    server = ThreadingHTTPServer((host, port), Handler)
    server.request_count = 0
    print(f"night-planner listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
