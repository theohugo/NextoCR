# SPDX-License-Identifier: Apache-2.0
"""Real-loopback tests for the local NextoCR Manager HTTP boundary."""

from __future__ import annotations

from contextlib import contextmanager
import http.client
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any, Iterator
from urllib.parse import quote
from unittest.mock import patch

import pytest

from nextocr_manager.__main__ import _probe_port, main
from nextocr_manager.http_server import APP_SIGNATURE, MAX_BODY_BYTES, NextoCRHTTPServer
from nextocr_manager.service import ManagerError


RUN_ID = "nextocr/mortar-v001-study"


class FakeManagerService:
    def __init__(self, static_root: Path, *, host: str = "127.0.0.1") -> None:
        self.config = SimpleNamespace(host=host, port=0, static_root=static_root)
        self.session_token = "test-session-token"
        self._closing = False
        self.calls: list[tuple[Any, ...]] = []

    def session(self) -> dict[str, Any]:
        return {"token": self.session_token, "header": "X-NextoCR-Token"}

    def state(self) -> dict[str, Any]:
        return {"selectedRunId": RUN_ID}

    def runs(self) -> list[dict[str, Any]]:
        return [{"runId": RUN_ID}]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        if run_id == "explode":
            raise RuntimeError("private implementation detail")
        return {"runId": run_id, "status": "paused"}

    def metrics(self, run_id: str) -> dict[str, Any]:
        return {"runId": run_id, "training": {"lastTimesteps": 123}}

    def logs(self, run_id: str, *, tail: int) -> dict[str, Any]:
        return {"runId": run_id, "tail": tail, "logs": {}}

    def replay(self, run_id: str, replay_id: str) -> dict[str, Any]:
        return {"runId": run_id, "replayId": replay_id, "frames": []}

    def create_and_start_run(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create", body))
        return {"runId": RUN_ID}

    def checkpoint(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("checkpoint", run_id))
        return {"command": "checkpoint"}

    def pause(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("pause", run_id))
        return {"status": "paused"}

    def resume(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("resume", run_id))
        return {"status": "running"}

    def create_version(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("versions", run_id, body))
        return {"runId": f"{run_id}-v2"}

    def create_replay(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("replays", run_id, body))
        return {"replayId": "replay-12345678", "status": "queued"}


@contextmanager
def running_server(tmp_path: Path) -> Iterator[tuple[FakeManagerService, NextoCRHTTPServer]]:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><h1>NextoCR dashboard</h1>", encoding="utf-8")
    (static / "assets" / "app.js").write_text("globalThis.nextocr = true;", encoding="utf-8")
    service = FakeManagerService(static)
    server = NextoCRHTTPServer(service)  # port 0 selects a real ephemeral localhost port
    thread = threading.Thread(target=server.serve_forever, name="test-nextocr-http")
    thread.start()
    try:
        yield service, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def request(
    server: NextoCRHTTPServer,
    method: str,
    target: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(method, target, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, {key.lower(): value for key, value in response.getheaders()}, response.read()
    finally:
        connection.close()


def json_request(
    server: NextoCRHTTPServer,
    method: str,
    target: str,
    value: dict[str, Any] | None = None,
    *,
    token: str | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    payload = json.dumps(value or {}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-NextoCR-Token"] = token
    status, response_headers, body = request(server, method, target, body=payload, headers=headers)
    return status, response_headers, json.loads(body)


def test_server_refuses_non_loopback_bind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        NextoCRHTTPServer(FakeManagerService(tmp_path, host="0.0.0.0"))


def test_session_static_spa_and_security_headers(tmp_path: Path) -> None:
    with running_server(tmp_path) as (_, server):
        status, headers, payload = request(server, "GET", "/api/session")
        data = json.loads(payload)["data"]
        assert status == 200
        assert data["appName"] == "NextoCR Manager"
        assert data["signature"] == APP_SIGNATURE
        assert data["token"] == "test-session-token"
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["cross-origin-resource-policy"] == "same-origin"
        assert "access-control-allow-origin" not in headers

        status, _, payload = request(server, "GET", "/training/current")
        assert status == 200
        assert b"NextoCR dashboard" in payload

        status, headers, payload = request(server, "GET", "/assets/app.js")
        assert status == 200
        assert headers["content-type"].startswith(("text/javascript", "application/javascript"))
        assert b"globalThis.nextocr" in payload


def test_static_traversal_and_missing_assets_are_blocked(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("must-not-leak", encoding="utf-8")
    with running_server(tmp_path) as (_, server):
        for target in ("/%2e%2e/secret.txt", "/..%5csecret.txt", "/assets/missing.js"):
            status, _, payload = request(server, "GET", target)
            assert status == 404
            assert b"must-not-leak" not in payload


def test_all_read_routes_use_success_envelopes_and_encoded_run_ids(tmp_path: Path) -> None:
    encoded = quote(RUN_ID, safe="")
    routes = {
        "/api/state": "selectedRunId",
        "/api/runs": None,
        f"/api/runs/{encoded}": "runId",
        f"/api/runs/{encoded}/metrics": "training",
        f"/api/runs/{encoded}/logs?tail=37": "tail",
        f"/api/runs/{encoded}/replays/replay-12345678": "replayId",
    }
    with running_server(tmp_path) as (_, server):
        for target, expected_key in routes.items():
            status, _, raw = request(server, "GET", target)
            payload = json.loads(raw)
            assert status == 200, target
            assert payload["ok"] is True
            if expected_key is not None:
                assert expected_key in payload["data"]


def test_mutations_require_token_and_dispatch_plural_routes(tmp_path: Path) -> None:
    encoded = quote(RUN_ID, safe="")
    with running_server(tmp_path) as (service, server):
        status, _, payload = json_request(server, "POST", "/api/runs", {"label": "study"})
        assert status == 403
        assert payload == {
            "ok": False,
            "error": {"code": "invalid_session", "message": "missing or invalid manager session token"},
        }
        assert service.calls == []

        status, _, payload = json_request(
            server,
            "POST",
            "/api/runs",
            {"label": "study"},
            token=service.session_token,
        )
        assert status == 201 and payload["ok"] is True

        expected = (
            ("checkpoint", {}, 200),
            ("pause", {}, 200),
            ("resume", {}, 200),
            ("versions", {"label": "v2", "targetTotalTimesteps": 20_000}, 201),
            ("replays", {"checkpoint": "latest", "opponent": "league"}, 202),
        )
        for action, body, expected_status in expected:
            status, _, payload = json_request(
                server,
                "POST",
                f"/api/runs/{encoded}/{action}",
                body,
                token=service.session_token,
            )
            assert status == expected_status, action
            assert payload["ok"] is True

        assert [call[0] for call in service.calls] == [
            "create", "checkpoint", "pause", "resume", "versions", "replays"
        ]
        assert service.calls[1][1] == RUN_ID


def test_json_limits_shape_content_type_and_errors(tmp_path: Path) -> None:
    with running_server(tmp_path) as (service, server):
        headers = {
            "Content-Type": "application/json",
            "X-NextoCR-Token": service.session_token,
            "Content-Length": str(MAX_BODY_BYTES + 1),
        }
        status, _, payload = request(
            server,
            "POST",
            "/api/runs",
            body=b"{}",
            headers=headers,
        )
        assert status == 413
        assert json.loads(payload)["error"]["code"] == "body_too_large"

        status, _, payload = request(
            server,
            "POST",
            "/api/runs",
            body=b"[]",
            headers={
                "Content-Type": "application/json",
                "X-NextoCR-Token": service.session_token,
            },
        )
        assert status == 400
        assert json.loads(payload)["error"]["code"] == "invalid_json"

        status, _, payload = request(
            server,
            "POST",
            "/api/runs",
            body=b"label=study",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-NextoCR-Token": service.session_token,
            },
        )
        assert status == 415
        assert json.loads(payload)["error"]["code"] == "invalid_content_type"

        status, headers, payload = request(server, "OPTIONS", "/api/runs")
        assert status == 405
        assert "access-control-allow-origin" not in headers
        assert json.loads(payload)["ok"] is False


def test_internal_errors_are_not_disclosed(tmp_path: Path) -> None:
    with running_server(tmp_path) as (_, server):
        status, _, raw = request(server, "GET", "/api/runs/explode")
        payload = json.loads(raw)
        assert status == 500
        assert payload["error"]["code"] == "internal_error"
        assert "private implementation detail" not in payload["error"]["message"]


def test_single_launch_probe_recognizes_existing_manager(tmp_path: Path) -> None:
    with running_server(tmp_path) as (_, server):
        assert _probe_port(server.server_port) == "nextocr"
        with patch("nextocr_manager.__main__.webbrowser.open") as browser:
            assert main(["--port", str(server.server_port), "--open-browser"]) == 0
            browser.assert_called_once()
