# -*- coding: utf-8 -*-
"""家庭接口安全与并发边界回归。"""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import api.family_api as family


def test_expired_token_rejected_even_without_new_login(tmp_path, monkeypatch):
    accounts = tmp_path / "accounts.json"
    accounts.write_text(json.dumps({
        "users": {"u": {"display_name": "U", "salt": "00", "hash": "x"}},
        "tokens": {"old": {"username": "u", "created": time.time() - family.TOKEN_TTL_S - 1}},
    }), encoding="utf-8")
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)

    with pytest.raises(HTTPException) as exc:
        family._user_from_token("old")
    assert exc.value.detail == "token_expired"
    assert "old" not in json.loads(accounts.read_text(encoding="utf-8"))["tokens"]


def test_legacy_plain_token_is_hashed_and_migrates_family_id(tmp_path, monkeypatch):
    accounts = tmp_path / "accounts.json"
    old_token = "legacy-token"
    accounts.write_text(json.dumps({
        "users": {"u": {"display_name": "U", "salt": "00", "hash": "x"}},
        "tokens": {old_token: {"username": "u", "created": time.time()}},
    }), encoding="utf-8")
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)

    user = family._user_from_token(old_token)
    assert user["family_id"] == family.DEMO_FAMILY_ID
    saved = json.loads(accounts.read_text(encoding="utf-8"))
    assert old_token not in saved["tokens"]
    assert family._token_key(old_token) in saved["tokens"]
    assert saved["users"]["u"]["family_id"] == family.DEMO_FAMILY_ID


def test_concurrent_ack_does_not_lose_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(family, "ACK_FILE", tmp_path / "acks.json")
    monkeypatch.setattr(family, "ACK_EVENT_FILE", tmp_path / "ack-events.jsonl")
    monkeypatch.setattr(family, "_store_lock", threading.Lock())
    barrier = threading.Barrier(16)

    def ack(i):
        barrier.wait()
        family.family_alert_ack(f"a{i}", {})

    threads = [threading.Thread(target=ack, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw = json.loads(family.ACK_FILE.read_text(encoding="utf-8"))
    saved = set(raw[family.DEMO_FAMILY_ID])
    assert saved == {f"a{i}" for i in range(16)}
    events = [
        json.loads(line) for line in
        family.ACK_EVENT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {event["alert_id"] for event in events} == saved


def test_alerts_expose_demo_family_id(tmp_path, monkeypatch):
    hist = tmp_path / "history.jsonl"
    hist.write_text(json.dumps({
        "id": "alert1", "final": "block", "name": "demo",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(family, "HIST_FILE", hist)
    monkeypatch.setattr(family, "ACK_FILE", tmp_path / "acks.json")

    result = family.family_alerts(user={})
    assert result["family_id"] == "demo"
    assert result["alerts"][0]["family_id"] == "demo"


def test_alerts_and_ack_are_family_scoped(tmp_path, monkeypatch):
    hist = tmp_path / "history.jsonl"
    rows = [
        {"id": "demo-a", "final": "block", "name": "legacy"},
        {"id": "a2", "family_id": "family-a", "final": "caution", "name": "A"},
        {"id": "b2", "family_id": "family-b", "final": "block", "name": "B"},
    ]
    hist.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")
    monkeypatch.setattr(family, "HIST_FILE", hist)
    monkeypatch.setattr(family, "ACK_FILE", tmp_path / "acks.json")

    a = family.family_alerts(user={"family_id": "family-a"})
    assert [x["id"] for x in a["alerts"]] == ["a2"]
    family.family_alert_ack("a2", {"family_id": "family-a"})
    assert family.family_alerts(
        user={"family_id": "family-a"})["alerts"][0]["ack"] is True
    assert family.family_alerts(
        user={"family_id": "family-b"})["alerts"][0]["ack"] is False


def _make_auth_app(monkeypatch, tmp_path, family_id: str):
    accounts = tmp_path / f"accounts-{family_id}.json"
    token = f"token-{family_id}"
    accounts.write_text(json.dumps({
        "users": {
            "u": {
                "display_name": "U",
                "salt": "00",
                "hash": "x",
                "family_id": family_id,
            },
        },
        "tokens": {
            family._token_key(token): {
                "username": "u",
                "family_id": family_id,
                "created": time.time(),
            },
        },
    }), encoding="utf-8")
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)
    app = FastAPI()
    app.include_router(family.router)
    return app, token


def test_child_call_ws_requires_ticket_or_authorization_header(
        tmp_path, monkeypatch):
    app, token = _make_auth_app(monkeypatch, tmp_path, "family-a")
    room = family._CallRoom(
        "r1", "risk", "red", family_id="family-a")
    monkeypatch.setattr(family, "_rooms", {"r1": room})

    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/call/r1?role=child") as ws:
                ws.receive_json()
        # 长期 token 不再允许放进 URL，防止日志与浏览器历史泄露。
        with pytest.raises(Exception):
            with client.websocket_connect(
                    f"/ws/call/r1?role=child&token={token}") as ws:
                ws.receive_json()
        with client.websocket_connect(
                "/ws/call/r1?role=child",
                headers={"Authorization": f"Bearer {token}"}) as ws:
            joined = ws.receive_json()
            assert joined["type"] == "joined"
            assert joined["role"] == "child"


def test_child_call_ws_ticket_is_single_use(tmp_path, monkeypatch):
    app, token = _make_auth_app(monkeypatch, tmp_path, "family-a")
    room = family._CallRoom(
        "r1", "risk", "red", family_id="family-a")
    monkeypatch.setattr(family, "_rooms", {"r1": room})

    with TestClient(app) as client:
        response = client.post(
            "/api/call/ws-ticket",
            headers={"Authorization": f"Bearer {token}"},
            json={"room_id": "r1"},
        )
        assert response.status_code == 200
        ticket = response.json()["ticket"]
        assert response.json()["expires_in"] == family.WS_TICKET_TTL_S

        with client.websocket_connect(
                f"/ws/call/r1?role=child&ticket={ticket}") as ws:
            assert ws.receive_json()["type"] == "joined"

        monkeypatch.setattr(family, "_rooms", {
            "r1": family._CallRoom(
                "r1", "risk", "red", family_id="family-a"),
        })
        with pytest.raises(Exception):
            with client.websocket_connect(
                    f"/ws/call/r1?role=child&ticket={ticket}") as ws:
                ws.receive_json()


def test_child_call_ws_rejects_cross_family_ticket(tmp_path, monkeypatch):
    app, token = _make_auth_app(monkeypatch, tmp_path, "family-a")
    room = family._CallRoom(
        "r1", "risk", "red", family_id="family-a")
    monkeypatch.setattr(family, "_rooms", {"r1": room})
    ticket = family._issue_ws_ticket({"family_id": "family-b"}, "r1")

    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect(
                    f"/ws/call/r1?role=child&ticket={ticket}") as ws:
                ws.receive_json()


def test_pending_call_is_visible_only_to_same_family(tmp_path, monkeypatch):
    monkeypatch.setattr(family, "_rooms", {
        "r1": family._CallRoom("r1", "risk", "red", family_id="family-a"),
    })
    assert family.call_pending(
        user={"family_id": "family-a"})["ringing"] is True
    assert family.call_pending(
        user={"family_id": "family-b"})["ringing"] is False


def test_family_number_report_writes_only_private_library(
        tmp_path, monkeypatch):
    from fusion.number_channel import NumberChannel

    public = tmp_path / "public.txt"
    public.write_text("# 公开库为空\n", encoding="utf-8")
    private = tmp_path / "family_blocklist.jsonl"
    monkeypatch.setenv("VERICALL_NUMBER_BLOCKLIST", str(public))
    monkeypatch.setenv("VERICALL_FAMILY_NUMBER_BLOCKLIST", str(private))
    monkeypatch.setenv("VERICALL_NUMBER_HEURISTIC", "0")

    result = family.family_number_report(
        {"number": "17012345678", "remark": "子女端举报"},
        {"family_id": "family-a"},
    )
    assert result["ok"] is True
    assert "仅对当前家庭" in result["message"]
    assert "17012345678" not in public.read_text(encoding="utf-8")

    nc = NumberChannel()
    assert nc.check(
        "17012345678", family_id="family-a").source == "family_blocklist"
    assert not nc.check("17012345678", family_id="family-b").matched


def test_call_ws_enforces_connection_limit_and_releases_slot(
        tmp_path, monkeypatch):
    app, _ = _make_auth_app(monkeypatch, tmp_path, "family-a")
    room = family._CallRoom("r1", "risk", "red", family_id="family-a")
    monkeypatch.setattr(family, "_rooms", {"r1": room})
    monkeypatch.setattr(family, "_active_call_connections", 0)
    monkeypatch.setenv("VERICALL_MAX_CALL_CONNECTIONS", "1")

    with TestClient(app) as client:
        with client.websocket_connect("/ws/call/r1?role=elder") as first:
            assert first.receive_json()["type"] == "joined"
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect(
                        "/ws/call/r1?role=elder") as second:
                    second.receive_json()
            assert exc.value.code == 4429

    assert family._active_call_connections == 0
    assert "r1" not in family._rooms


@pytest.mark.parametrize("payload", [
    pytest.param(b"\x00", id="odd-byte-length"),
    pytest.param(b"\x00" * (1024 * 1024 + 2), id="oversized"),
])
def test_call_ws_rejects_invalid_or_oversized_audio_frame(
        payload, tmp_path, monkeypatch):
    app, _ = _make_auth_app(monkeypatch, tmp_path, "family-a")
    room = family._CallRoom("r1", "risk", "red", family_id="family-a")
    monkeypatch.setattr(family, "_rooms", {"r1": room})
    monkeypatch.setattr(family, "_active_call_connections", 0)
    monkeypatch.setenv("VERICALL_MAX_CALL_FRAME_MB", "1")

    with TestClient(app) as client:
        with client.websocket_connect("/ws/call/r1?role=elder") as ws:
            assert ws.receive_json()["type"] == "joined"
            ws.send_bytes(payload)
            assert ws.receive_json()["code"] == "invalid_audio_frame"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1009

    assert family._active_call_connections == 0


def test_call_ws_rejects_oversized_control_frame(tmp_path, monkeypatch):
    app, _ = _make_auth_app(monkeypatch, tmp_path, "family-a")
    room = family._CallRoom("r1", "risk", "red", family_id="family-a")
    monkeypatch.setattr(family, "_rooms", {"r1": room})
    monkeypatch.setattr(family, "_active_call_connections", 0)
    monkeypatch.setenv("VERICALL_MAX_CALL_CONTROL_KB", "1")

    with TestClient(app) as client:
        with client.websocket_connect("/ws/call/r1?role=elder") as ws:
            assert ws.receive_json()["type"] == "joined"
            ws.send_text("x" * 1025)
            assert ws.receive_json()["code"] == "control_frame_too_large"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1009

    assert family._active_call_connections == 0


def test_public_config_defaults_and_runtime_switches(monkeypatch):
    monkeypatch.delenv("VERICALL_ALLOW_REGISTRATION", raising=False)
    monkeypatch.delenv("VERICALL_DEMO_ACCOUNT", raising=False)
    monkeypatch.delenv(family.EXPOSE_DEMO_CREDENTIALS_ENV, raising=False)
    monkeypatch.delenv("VERICALL_DEMO_USERNAME", raising=False)
    monkeypatch.delenv("VERICALL_DEMO_PASSWORD", raising=False)
    monkeypatch.delenv(family.ELDER_ACCESS_ENV, raising=False)
    monkeypatch.delenv("VERICALL_REQUIRE_FAMILY_INVITE", raising=False)
    monkeypatch.delenv("VERICALL_MAX_UPLOAD_MB", raising=False)

    config = family.public_config()
    assert config == {
        "allow_registration": True,
        "demo_account": True,
        "invite_required": False,
        "max_upload_mb": 25,
        "challenge_ttl_s": 60,
        "call_rooms_in_memory": True,
    }

    monkeypatch.setenv("VERICALL_ALLOW_REGISTRATION", "0")
    monkeypatch.setenv("VERICALL_DEMO_ACCOUNT", "false")
    monkeypatch.setenv("VERICALL_MAX_UPLOAD_MB", "not-a-number")
    config = family.public_config()
    assert config["allow_registration"] is False
    assert config["demo_account"] is False
    assert config["max_upload_mb"] == 25
    assert "demo_credentials" not in config


def test_public_config_exposes_demo_elder_key_only_in_demo_mode(monkeypatch):
    key = "demo-elder-key-0123456789"
    monkeypatch.setenv("VERICALL_DEMO_ACCOUNT", "1")
    monkeypatch.setenv(family.EXPOSE_DEMO_CREDENTIALS_ENV, "1")
    monkeypatch.setenv("VERICALL_DEMO_PASSWORD", "demo-password-for-test")
    monkeypatch.setenv(family.ELDER_ACCESS_ENV, key)

    config = family.public_config()
    assert config["demo_credentials"]["elder_key"] == key

    monkeypatch.setenv("VERICALL_DEMO_ACCOUNT", "0")
    config = family.public_config()
    assert "demo_credentials" not in config


def test_public_config_hides_demo_credentials_from_remote_requests(monkeypatch):
    monkeypatch.setenv("VERICALL_DEMO_ACCOUNT", "1")
    monkeypatch.delenv(family.EXPOSE_DEMO_CREDENTIALS_ENV, raising=False)
    monkeypatch.delenv("VERICALL_TRUST_PROXY_HEADERS", raising=False)
    request = family.Request({
        "type": "http",
        "method": "GET",
        "path": "/api/config",
        "headers": [(b"host", b"vericall.example.com")],
        "client": ("203.0.113.9", 45678),
        "server": ("127.0.0.1", 8000),
        "scheme": "https",
    })

    assert "demo_credentials" not in family.public_config(request)


def test_public_config_exposes_demo_credentials_to_local_requests(monkeypatch):
    monkeypatch.setenv("VERICALL_DEMO_ACCOUNT", "1")
    monkeypatch.setenv("VERICALL_DEMO_PASSWORD", "demo-password-for-test")
    monkeypatch.delenv(family.EXPOSE_DEMO_CREDENTIALS_ENV, raising=False)
    monkeypatch.delenv("VERICALL_TRUST_PROXY_HEADERS", raising=False)
    request = family.Request({
        "type": "http",
        "method": "GET",
        "path": "/api/config",
        "headers": [(b"host", b"127.0.0.1:8000")],
        "client": ("127.0.0.1", 45678),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    })

    assert family.public_config(request)["demo_credentials"]["child_username"]


def test_public_config_omits_demo_credentials_without_runtime_password(
        monkeypatch):
    monkeypatch.setenv("VERICALL_DEMO_ACCOUNT", "1")
    monkeypatch.setenv(family.EXPOSE_DEMO_CREDENTIALS_ENV, "1")
    monkeypatch.delenv("VERICALL_DEMO_PASSWORD", raising=False)

    assert "demo_credentials" not in family.public_config()


def test_registration_can_be_disabled_without_touching_existing_accounts(
        tmp_path, monkeypatch):
    monkeypatch.setattr(family, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setenv("VERICALL_ALLOW_REGISTRATION", "0")

    response = family.family_register(
        family._Auth(username="new-user", password="secret123"))
    assert response.status_code == 403
    assert json.loads(response.body)["error"] == "registration_disabled"
    assert not family.ACCOUNTS_FILE.exists()


def test_invitation_registration_binds_family_and_consumes_use(
        tmp_path, monkeypatch):
    monkeypatch.setattr(family, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setattr(family, "INVITES_FILE", tmp_path / "invites.json")
    monkeypatch.setattr(family, "_store_lock", threading.Lock())
    monkeypatch.setenv("VERICALL_REQUIRE_FAMILY_INVITE", "1")
    family._save_invites({"invites": {
        "pilot-01-abc123": {
            "family_id": "pilot-01",
            "max_uses": 2,
            "used": 0,
            "active": True,
            "expires_at": time.time() + 3600,
        },
    }})

    response = family.family_register(family._Auth(
        username="child-01",
        password="secret123",
        invitation="pilot-01-abc123",
    ))
    assert response == {"ok": True, "message": "注册成功，请登录"}
    saved = json.loads(family.ACCOUNTS_FILE.read_text(encoding="utf-8"))
    assert saved["users"]["child-01"]["family_id"] == "pilot-01"
    invites = json.loads(family.INVITES_FILE.read_text(encoding="utf-8"))
    assert invites["invites"]["pilot-01-abc123"]["used"] == 1


def test_invitation_required_rejects_missing_or_exhausted_code(
        tmp_path, monkeypatch):
    monkeypatch.setattr(family, "ACCOUNTS_FILE", tmp_path / "accounts.json")
    monkeypatch.setattr(family, "INVITES_FILE", tmp_path / "invites.json")
    monkeypatch.setattr(family, "_store_lock", threading.Lock())
    monkeypatch.setenv("VERICALL_REQUIRE_FAMILY_INVITE", "1")

    missing = family.family_register(family._Auth(
        username="child-02", password="secret123"))
    assert missing.status_code == 400
    assert json.loads(missing.body)["error"] == "invitation_required"

    family._save_invites({"invites": {
        "pilot-02-abc123": {
            "family_id": "pilot-02",
            "max_uses": 1,
            "used": 1,
            "active": True,
        },
    }})
    exhausted = family.family_register(family._Auth(
        username="child-03", password="secret123",
        invitation="pilot-02-abc123"))
    assert exhausted.status_code == 403
    assert json.loads(exhausted.body)["error"] == "invitation_exhausted"


def test_request_scope_allows_local_parent_but_rejects_remote_anonymous():
    local = type("Request", (), {
        "client": type("Client", (), {"host": "127.0.0.1"})(),
    })()
    remote = type("Request", (), {
        "client": type("Client", (), {"host": "203.0.113.8"})(),
    })()

    assert family.request_family_scope(
        local, None, required=True) == (family.DEMO_FAMILY_ID, "local")
    with pytest.raises(HTTPException) as exc:
        family.request_family_scope(remote, None, required=True)
    assert exc.value.status_code == 401
    assert family.request_family_scope(
        remote, None, required=False) == (None, "public")


def test_voice_owner_claim_prevents_cross_family_name_collision(
        tmp_path, monkeypatch):
    monkeypatch.setattr(family, "VOICE_OWNERS_FILE", tmp_path / "owners.json")
    monkeypatch.setattr(family, "_store_lock", threading.Lock())

    assert family.claim_voice_owner("妈妈", "family-a") is True
    assert family.voice_owner("妈妈") == "family-a"
    assert family.claim_voice_owner("妈妈", "family-b") is False
    assert family.claim_voice_owner("妈妈", "family-a") is True
