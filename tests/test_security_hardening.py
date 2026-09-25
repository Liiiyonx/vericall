# -*- coding: utf-8 -*-
"""安全加固回归：代理边界、限流、票据、上传与家庭隔离。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import api.family_api as family
import api.server as server
from fusion.voiceprint_channel import VoiceprintChannel
from server import ws_api
from server.ws_api import register_stream_ws


def _http_request(host: str = "127.0.0.1", *, headers=None,
                  scheme: str = "http"):
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers=headers or {},
        url=SimpleNamespace(scheme=scheme),
    )


def _write_accounts(path, token: str | None = None,
                    family_id: str = "family-a") -> None:
    tokens = {}
    if token:
        tokens[family._token_key(token)] = {
            "username": "u",
            "family_id": family_id,
            "created": time.time(),
        }
    path.write_text(json.dumps({
        "users": {
            "u": {
                "display_name": "U",
                "salt": "00" * 16,
                "hash": "x",
                "family_id": family_id,
            },
        },
        "tokens": tokens,
    }), encoding="utf-8")


def _wait_for_stream_slots(expected: int = 0, timeout_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ws_api._active_stream_connections == expected:
            return
        time.sleep(0.01)
    assert ws_api._active_stream_connections == expected


def test_login_failure_limit_returns_retry_after(tmp_path, monkeypatch):
    accounts = tmp_path / "accounts.json"
    salt = "00" * 16
    accounts.write_text(json.dumps({
        "users": {
            "u": {
                "display_name": "U",
                "salt": salt,
                "hash": family._hash_pw("correct123", salt),
                "family_id": "family-a",
            },
        },
        "tokens": {},
    }), encoding="utf-8")
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)
    monkeypatch.setattr(family, "_login_failures", {})
    monkeypatch.setattr(family, "_login_lock", threading.Lock())
    request = _http_request("203.0.113.8")

    for _ in range(family.LOGIN_MAX_FAILURES):
        response = family.family_login(
            family._Auth(username="u", password="wrong-pass"), request)
        assert response.status_code == 401

    blocked = family.family_login(
        family._Auth(username="u", password="correct123"), request)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_login_failure_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(family, "_login_failures", {})
    monkeypatch.setattr(family, "_login_lock", threading.Lock())
    monkeypatch.setenv("VERICALL_LOGIN_MAX_TRACKED_IDENTITIES", "8")

    for idx in range(40):
        family._record_login_failure((f"ip:{idx}", f"user:u{idx}"))

    assert len(family._login_failures) <= 8


def test_proxy_headers_cannot_claim_loopback_with_trust_enabled(monkeypatch):
    forwarded = {
        "host": "127.0.0.1:8000",
        "x-forwarded-for": "203.0.113.9",
    }
    monkeypatch.delenv("VERICALL_TRUST_PROXY_HEADERS", raising=False)
    assert family.is_local_request(
        _http_request(headers=forwarded)) is False

    monkeypatch.setenv("VERICALL_TRUST_PROXY_HEADERS", "1")
    assert family.is_local_request(
        _http_request(headers=forwarded)) is False
    assert family.is_local_request(
        _http_request(headers={"host": "127.0.0.1:8000"})) is True


def test_local_browser_path_rejects_cross_origin(monkeypatch):
    monkeypatch.delenv("VERICALL_TRUST_PROXY_HEADERS", raising=False)
    cross_origin = {
        "host": "127.0.0.1:8000",
        "origin": "https://attacker.example",
    }
    same_origin = {
        "host": "127.0.0.1:8000",
        "origin": "http://127.0.0.1:8000",
    }
    assert family.is_local_request(
        _http_request(headers=cross_origin)) is False
    assert family.is_local_request(
        _http_request(headers=same_origin)) is True


class _Proc:
    def __init__(self, sample_rate: int, family_id: str, observed: dict):
        self.sr = sample_rate
        self.family_id = family_id
        self.observed = observed

    def push_chunk(self, samples):
        return [{"type": "window", "samples": len(samples)}]

    def close(self):
        self.observed["closed"] = True


def test_stream_ws_rejects_remote_anonymous_but_accepts_token(
        tmp_path, monkeypatch):
    token = "token-family-a"
    accounts = tmp_path / "accounts.json"
    _write_accounts(accounts, token)
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)
    observed = {}
    app = FastAPI()
    register_stream_ws(
        app, lambda sr, family_id: _Proc(sr, family_id, observed))
    remote = TestClient(
        app, client=("203.0.113.10", 12000), raise_server_exceptions=True)

    with pytest.raises(WebSocketDisconnect) as exc:
        with remote.websocket_connect("/ws/stream"):
            pass
    assert exc.value.code == 4401

    with remote.websocket_connect(
            "/ws/stream",
            headers={"Authorization": f"Bearer {token}"}) as ws:
        ws.send_bytes(b"\x00\x00")
        assert ws.receive_json()["type"] == "window"
    assert observed["closed"] is True
    _wait_for_stream_slots()


def test_stream_ticket_is_single_use_and_purpose_bound(monkeypatch):
    app = FastAPI()
    register_stream_ws(app, lambda sr, family_id: _Proc(sr, family_id, {}))
    remote = TestClient(app, client=("203.0.113.11", 12000))
    monkeypatch.setenv(
        family.ELDER_ACCESS_ENV, "elder-key-0123456789")
    ticket = family.stream_ws_ticket(
        request=_http_request(
            "203.0.113.11",
            headers={family.ELDER_ACCESS_HEADER: "elder-key-0123456789"},
        ))["ticket"]

    with remote.websocket_connect(f"/ws/stream?ticket={ticket}"):
        pass
    with pytest.raises(WebSocketDisconnect) as exc:
        with remote.websocket_connect(f"/ws/stream?ticket={ticket}"):
            pass
    assert exc.value.code == 4401

    call_ticket = family._issue_ws_ticket(
        {"family_id": "family-a"}, "room-1")
    with pytest.raises(WebSocketDisconnect) as exc:
        with remote.websocket_connect(f"/ws/stream?ticket={call_ticket}"):
            pass
    assert exc.value.code == 4401
    _wait_for_stream_slots()


def test_elder_access_key_is_scoped_to_approved_endpoints(monkeypatch):
    key = "elder-key-0123456789"
    monkeypatch.setenv(family.ELDER_ACCESS_ENV, key)
    request = _http_request(
        "203.0.113.12",
        headers={family.ELDER_ACCESS_HEADER: key},
    )

    assert family.request_family_scope(
        request, None, required=True, allow_elder_key=True,
    ) == (family.DEMO_FAMILY_ID, "elder_key")
    with pytest.raises(HTTPException) as exc:
        family.request_family_scope(request, None, required=True)
    assert exc.value.status_code == 401

    wrong = _http_request(
        "203.0.113.12",
        headers={family.ELDER_ACCESS_HEADER: key + "-wrong"},
    )
    with pytest.raises(HTTPException) as exc:
        family.request_family_scope(
            wrong, None, required=True, allow_elder_key=True)
    assert exc.value.status_code == 401


def test_elder_access_key_can_issue_stream_ticket(monkeypatch):
    key = "elder-key-0123456789"
    monkeypatch.setenv(family.ELDER_ACCESS_ENV, key)
    request = _http_request(
        "203.0.113.13",
        headers={family.ELDER_ACCESS_HEADER: key},
    )

    payload = family.stream_ws_ticket(request)
    assert family._consume_ws_ticket(
        payload["ticket"], purpose="stream") == family.DEMO_FAMILY_ID
    assert family._consume_ws_ticket(
        payload["ticket"], purpose="stream") is None

    with pytest.raises(HTTPException) as exc:
        family.stream_ws_ticket(_http_request("203.0.113.13"))
    assert exc.value.status_code == 401


def test_elder_call_invite_issues_room_bound_elder_ticket(monkeypatch):
    key = "elder-key-0123456789"
    monkeypatch.setenv(family.ELDER_ACCESS_ENV, key)
    monkeypatch.setattr(family, "_rooms", {})
    request = _http_request(
        "203.0.113.14",
        headers={family.ELDER_ACCESS_HEADER: key},
    )

    payload = asyncio.run(family.call_invite(
        {"reason": "test", "level": "red"}, request))
    assert payload["ok"] is True
    assert family._consume_ws_ticket(
        payload["ws_ticket"],
        purpose="call_elder",
        room_id=payload["room_id"],
    ) == family.DEMO_FAMILY_ID
    assert family._consume_ws_ticket(
        payload["ws_ticket"],
        purpose="call_elder",
        room_id=payload["room_id"],
    ) is None

    denied = asyncio.run(family.call_invite(
        {"reason": "test"}, _http_request("203.0.113.14")))
    assert denied.status_code == 403


def test_elder_routes_do_not_open_child_data(monkeypatch):
    key = "elder-key-0123456789"
    headers = {family.ELDER_ACCESS_HEADER: key}
    monkeypatch.setenv(family.ELDER_ACCESS_ENV, key)
    monkeypatch.setattr(family, "_rooms", {})

    with TestClient(
            server.app, client=("203.0.113.15", 12000)) as client:
        status = client.get("/api/status", headers=headers)
        assert status.status_code == 200
        assert status.json()["access_mode"] == "elder_key"
        assert status.json()["family_scope"] == family.DEMO_FAMILY_ID

        history = client.get("/api/history", headers=headers)
        assert history.status_code == 401
        alerts = client.get("/api/family/alerts", headers=headers)
        assert alerts.status_code == 401

        ticket = client.post("/api/stream/ws-ticket", headers=headers)
        assert ticket.status_code == 200
        assert ticket.json()["ticket"]

        invite = client.post(
            "/api/call/invite",
            headers=headers,
            json={"reason": "route test", "level": "red"},
        )
        assert invite.status_code == 200
        assert invite.json()["ws_ticket"]


def test_stream_ws_enforces_connection_and_frame_limits(monkeypatch):
    observed = {}
    app = FastAPI()
    register_stream_ws(
        app, lambda sr, family_id: _Proc(sr, family_id, observed))
    monkeypatch.setenv("VERICALL_MAX_STREAM_CONNECTIONS", "1")
    monkeypatch.setattr(ws_api, "_active_stream_connections", 0)
    monkeypatch.setenv("VERICALL_MAX_STREAM_FRAME_MB", "1")

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/ws/stream"):
                    pass
            assert exc.value.code == 4429
            ws.send_bytes(b"\x00" * (1024 * 1024 + 1))
            assert ws.receive_json()["code"] == "audio_frame_too_large"

    _wait_for_stream_slots()


def test_invalid_audio_content_is_rejected_before_pipeline_load(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("invalid uploads must not load the model pipeline")

    monkeypatch.setattr(server, "get_pipe", fail_if_loaded)
    # 防止其他离线回归用例遗留 VERICALL_OFFLINE=1，使校验顺序难以复现。
    monkeypatch.setattr(server, "is_offline", lambda: False)
    with TestClient(server.app, client=("127.0.0.1", 12000)) as client:
        response = client.post(
            "/api/analyze",
            files={"file": ("fake.wav", b"not-an-audio-file", "audio/wav")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_audio_content"


def test_oversized_content_length_is_rejected_before_multipart_parse(
        monkeypatch):
    monkeypatch.setenv("VERICALL_MAX_UPLOAD_MB", "1")
    declared = 4 * 1024 * 1024
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/analyze"),
        headers={"content-length": str(declared)},
    )
    assert server._content_length_exceeds_upload_limit(request) is True

    with TestClient(server.app, client=("127.0.0.1", 12000)) as client:
        response = client.post(
            "/api/analyze",
            headers={"content-length": str(declared)},
            content=b"",
        )
    assert response.status_code == 413
    assert response.json()["error"] == "request_too_large"


def test_security_headers_and_api_docs_are_locked_down():
    with TestClient(server.app) as client:
        response = client.get("/healthz")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in response.headers[
            "content-security-policy"]
        assert response.headers[
            "content-security-policy"].endswith("connect-src 'self'; "
                                                 "font-src 'self' data:")
        assert response.headers[
            "cross-origin-resource-policy"] == "same-origin"
        assert response.headers[
            "cross-origin-opener-policy"] == "same-origin"
        assert response.headers[
            "x-permitted-cross-domain-policies"] == "none"
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_challenge_is_family_scoped_and_limited(monkeypatch):
    monkeypatch.setattr(server, "_challenges", {})
    monkeypatch.setattr(server, "_challenges_lock", threading.Lock())
    request = _http_request()

    response = server.challenge_begin(request, authorization=None)
    assert response["ok"] is True
    code = response["code"]
    assert server._claim_challenge(code, "family-b") is None
    assert server._claim_challenge(
        code, family.DEMO_FAMILY_ID) == response["digits"]

    now = time.time()
    for idx in range(server._CHALLENGE_MAX_PER_FAMILY):
        server._challenges[f"{idx:08x}"] = {
            "digits": "1234",
            "expires": now + server._CHALLENGE_TTL_S,
            "family_id": family.DEMO_FAMILY_ID,
        }
    limited = server.challenge_begin(request, authorization=None)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_voiceprint_verification_is_limited_to_allowed_family(
        monkeypatch):
    channel = VoiceprintChannel.__new__(VoiceprintChannel)
    channel.profiles = {
        "family-a": np.asarray([1.0, 0.0], dtype=np.float32),
        "family-b": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    monkeypatch.setattr(
        channel, "_embed",
        lambda _path: np.asarray([1.0, 0.0], dtype=np.float32))

    best, similarity, error = channel.similarity(
        "probe.wav", allowed_names={"family-b"})
    assert error is None
    assert best == "family-b"
    assert similarity == pytest.approx(0.0)
    assert "family-a" not in channel.verify(
        "probe.wav", allowed_names={"family-b"}).detail
