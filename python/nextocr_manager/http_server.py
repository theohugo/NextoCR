# SPDX-License-Identifier: Apache-2.0
"""Hardened loopback HTTP API and static dashboard server."""

from __future__ import annotations

import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import re
import secrets
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .service import ManagerError, ManagerService


MAX_BODY_BYTES = 64 * 1024
RUN_ACTIONS = frozenset({"checkpoint", "pause", "resume", "versions", "replays"})
APP_NAME = "NextoCR Manager"
APP_SIGNATURE = "nextocr-manager-v1"


class NextoCRHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, service: ManagerService):
        if service.config.host != "127.0.0.1":
            raise ValueError("NextoCR Manager must bind exactly to 127.0.0.1")
        self.service = service
        super().__init__((service.config.host, service.config.port), NextoCRRequestHandler)


class NextoCRRequestHandler(BaseHTTPRequestHandler):
    server: NextoCRHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        try:
            self._validate_host()
            parsed = urlsplit(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api_get(parsed.path, parse_qs(parsed.query))
            else:
                self._serve_static(parsed.path)
        except Exception as exc:
            self._send_exception(exc)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        try:
            self._validate_host()
            self._validate_mutation_request()
            body = self._read_json_body()
            self._handle_api_post(urlsplit(self.path).path, body)
        except Exception as exc:
            self._send_exception(exc)

    def do_OPTIONS(self) -> None:  # noqa: N802 - deliberately no CORS preflight
        self._send_error_envelope(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "CORS is disabled")

    def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        service = self.server.service
        if path == "/api/health":
            self._send_json({"ok": True, "data": {"status": "ready"}})
            return
        if path == "/api/session":
            session = dict(service.session())
            session.update(appName=APP_NAME, signature=APP_SIGNATURE)
            self._send_json({"ok": True, "data": session})
            return
        if path == "/api/state":
            self._send_json({"ok": True, "data": service.state()})
            return
        if path == "/api/runs":
            self._send_json({"ok": True, "data": service.runs()})
            return

        replay_match = re.fullmatch(r"/api/runs/(.+)/replays/([^/]+)", path)
        if replay_match:
            run_id = self._decode_run_id(replay_match.group(1))
            replay_id = unquote(replay_match.group(2))
            self._send_json({"ok": True, "data": service.replay(run_id, replay_id)})
            return

        route = self._parse_run_suffix(path, {"metrics", "logs"})
        if route is not None:
            run_id, suffix = route
            if suffix == "metrics":
                self._send_json({"ok": True, "data": service.metrics(run_id)})
            else:
                raw_tail = query.get("tail", ["200"])[0]
                try:
                    tail = int(raw_tail)
                except ValueError as exc:
                    raise ManagerError("invalid_tail", "tail must be an integer") from exc
                self._send_json({"ok": True, "data": service.logs(run_id, tail=tail)})
            return

        prefix = "/api/runs/"
        if path.startswith(prefix):
            run_id = self._decode_run_id(path[len(prefix):])
            self._send_json({"ok": True, "data": service.run_detail(run_id)})
            return
        raise ManagerError("not_found", "API endpoint not found", 404)

    def _handle_api_post(self, path: str, body: dict[str, Any]) -> None:
        service = self.server.service
        if path == "/api/runs":
            data = service.create_and_start_run(body)
            self._send_json({"ok": True, "data": data}, status=HTTPStatus.CREATED)
            return
        route = self._parse_run_suffix(path, RUN_ACTIONS)
        if route is None:
            raise ManagerError("not_found", "API endpoint not found", 404)
        run_id, action = route
        if action == "checkpoint":
            data = service.checkpoint(run_id)
            status = HTTPStatus.OK
        elif action == "pause":
            data = service.pause(run_id)
            status = HTTPStatus.OK
        elif action == "resume":
            data = service.resume(run_id)
            status = HTTPStatus.OK
        elif action == "versions":
            data = service.create_version(run_id, body)
            status = HTTPStatus.CREATED
        elif action == "replays":
            data = service.create_replay(run_id, body)
            status = HTTPStatus.ACCEPTED
        else:  # pragma: no cover - guarded by the suffix set
            raise ManagerError("not_found", "API endpoint not found", 404)
        self._send_json({"ok": True, "data": data}, status=status)

    def _parse_run_suffix(self, path: str, suffixes: set[str] | frozenset[str]) -> tuple[str, str] | None:
        prefix = "/api/runs/"
        if not path.startswith(prefix):
            return None
        remainder = path[len(prefix):]
        for suffix in sorted(suffixes, key=len, reverse=True):
            marker = f"/{suffix}"
            if remainder.endswith(marker):
                return self._decode_run_id(remainder[:-len(marker)]), suffix
        return None

    @staticmethod
    def _decode_run_id(raw: str) -> str:
        try:
            value = unquote(raw, errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ManagerError("invalid_run_id", "invalid runId") from exc
        if not value or "?" in value or "#" in value or "\x00" in value or "\\" in value:
            raise ManagerError("invalid_run_id", "invalid runId")
        return value

    def _validate_host(self) -> None:
        host = self.headers.get("Host")
        allowed = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if host not in allowed:
            if self.command == "POST":
                self.close_connection = True
            raise ManagerError("invalid_host", "request Host must be the local dashboard", 403)

    def _validate_mutation_request(self) -> None:
        if not urlsplit(self.path).path.startswith("/api/"):
            raise ManagerError("not_found", "POST is supported only by the local API", 404)
        token = self.headers.get("X-NextoCR-Token", "")
        # Hash first so compare_digest always receives fixed-size inputs.
        supplied = hashlib.sha256(token.encode("utf-8", errors="replace")).digest()
        expected = hashlib.sha256(
            self.server.service.session_token.encode("utf-8", errors="replace")
        ).digest()
        if not secrets.compare_digest(supplied, expected):
            self.close_connection = True
            raise ManagerError("invalid_session", "missing or invalid manager session token", 403)
        origin = self.headers.get("Origin")
        if origin:
            allowed = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if origin not in allowed:
                self.close_connection = True
                raise ManagerError("invalid_origin", "mutation Origin must be the local dashboard", 403)

    def _read_json_body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self.close_connection = True
            raise ManagerError("invalid_content_type", "Content-Type must be application/json", 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            self.close_connection = True
            raise ManagerError("invalid_body", "invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            raise ManagerError("body_too_large", "JSON body exceeds 64 KiB", 413)
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManagerError("invalid_json", "request body is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ManagerError("invalid_json", "request body must be a JSON object")
        return value

    def _serve_static(self, raw_path: str) -> None:
        static_root = self.server.service.config.static_root.resolve()
        try:
            path = unquote(raw_path, errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ManagerError("not_found", "static asset not found", 404) from exc
        if "\x00" in path or "\\" in path:
            raise ManagerError("not_found", "static asset not found", 404)
        if path in ("", "/"):
            relative = PurePosixPath("index.html")
        else:
            relative = PurePosixPath(path.lstrip("/"))
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ManagerError("not_found", "static asset not found", 404)
        candidate = (static_root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError as exc:
            raise ManagerError("not_found", "static asset not found", 404) from exc
        if not candidate.is_file():
            if relative.suffix:
                raise ManagerError("not_found", "static asset not found", 404)
            candidate = static_root / "index.html"
        if not candidate.is_file():
            self._send_html(
                "<!doctype html><meta charset=utf-8><title>NextoCR</title>"
                "<h1>NextoCR Manager</h1><p>Interface assets are not built yet.</p>",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        content = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, value: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, value: str, *, status: int) -> None:
        body = value.encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )

    def _send_exception(self, exc: Exception) -> None:
        if isinstance(exc, ManagerError):
            self._send_error_envelope(exc.status, exc.code, str(exc))
        elif isinstance(exc, FileNotFoundError):
            self._send_error_envelope(HTTPStatus.NOT_FOUND, "not_found", str(exc))
        elif isinstance(exc, (ValueError, TypeError)):
            self._send_error_envelope(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        else:
            self.log_error("manager service error: %s", exc)
            self._send_error_envelope(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "the manager could not complete the request",
            )

    def _send_error_envelope(self, status: int, code: str, message: str) -> None:
        self._send_json(
            {"ok": False, "error": {"code": code, "message": message}},
            status=status,
        )

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal output concise; run logs are exposed through the API.
        if self.server.service._closing:
            return
        super().log_message(format, *args)
