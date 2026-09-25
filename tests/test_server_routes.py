# -*- coding: utf-8 -*-
"""真实 app 路由顺序回归：/ws/stream 不能被 StaticFiles 的根挂载吞掉。"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException


def test_ws_stream_reachable_and_static_http_still_works(monkeypatch):
    import api.server as server

    monkeypatch.setattr(server, "get_pipe", lambda: SimpleNamespace())
    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/manifest.json").status_code == 200
        with client.websocket_connect("/ws/stream"):
            pass


def test_root_serves_entry_page_and_role_pages_stay_available():
    import api.server as server

    with TestClient(server.app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "选择要进入的守护端" in root.text
        assert "api/family/login" in root.text
        assert "child.html" in root.text
        assert "elder.html" in root.text
        assert "index.html" in root.text

        assert client.get("/index.html").status_code == 200
        assert client.get("/child.html").status_code == 200
        assert client.get("/elder.html").status_code == 200
        assert client.get("/login.html").status_code == 200


def test_healthz_does_not_load_model(monkeypatch):
    import api.server as server

    def fail_if_loaded():
        raise AssertionError("healthz must not load the inference pipeline")

    monkeypatch.setattr(server, "get_pipe", fail_if_loaded)
    with TestClient(server.app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["service"] == "vericall"


def _remote_request(host: str = "203.0.113.9"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_history_requires_remote_authentication(monkeypatch, tmp_path):
    import api.server as server

    monkeypatch.setattr(server, "HIST_FILE", tmp_path / "history.jsonl")
    with pytest.raises(HTTPException) as exc:
        server.history(request=_remote_request(), authorization=None, limit=50)
    assert exc.value.status_code == 401


def test_history_is_filtered_by_token_family(monkeypatch, tmp_path):
    import api.family_api as family
    import api.history_store as history_store
    import api.server as server

    history = tmp_path / "history.jsonl"
    rows = [
        {"id": "a", "family_id": "family-a", "final": "block"},
        {"id": "b", "family_id": "family-b", "final": "allow"},
        {"id": "legacy", "final": "caution"},
    ]
    history.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8")
    history_store.append_history({
        "id": "a-shard",
        "family_id": "family-a",
        "final": "block",
    }, history)
    token = "family-a-token"
    accounts = tmp_path / "accounts.json"
    accounts.write_text(json.dumps({
        "users": {"u": {
            "display_name": "U", "salt": "00", "hash": "x",
            "family_id": "family-a",
        }},
        "tokens": {family._token_key(token): {
            "username": "u", "family_id": "family-a",
            "created": time.time(),
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)
    monkeypatch.setattr(server, "HIST_FILE", history)

    result = server.history(
        request=_remote_request(),
        authorization=f"Bearer {token}",
        limit=50,
    )
    assert result["family_id"] == "family-a"
    assert result["access_mode"] == "token"
    assert [record["id"] for record in result["records"]] == [
        "a-shard", "a"]


def test_public_status_does_not_leak_voice_names(monkeypatch, tmp_path):
    import api.family_api as family
    import api.server as server

    voice_dir = tmp_path / "voices"
    (voice_dir / "妈妈").mkdir(parents=True)
    (voice_dir / "妈妈" / "sample.wav").write_bytes(b"x")
    monkeypatch.setattr(server, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(server, "OLLAMA_HOST", "http://127.0.0.1:1")
    monkeypatch.setattr(family, "VOICE_OWNERS_FILE", tmp_path / "owners.json")

    status = server.status(
        request=_remote_request(), authorization=None)
    assert status["family_scope"] == "public"
    assert status["access_mode"] == "public"
    assert status["voices"] == []
    assert status["voice_count"] == 0
    assert status["acoustic_active"] == "xlsr_cn:wide"


def test_voice_list_is_filtered_by_token_family(monkeypatch, tmp_path):
    import api.family_api as family
    import api.server as server

    accounts = tmp_path / "accounts.json"
    tokens = {"family-a": "token-a", "family-b": "token-b"}
    accounts.write_text(json.dumps({
        "users": {
            "a": {"display_name": "A", "salt": "00", "hash": "x",
                  "family_id": "family-a"},
            "b": {"display_name": "B", "salt": "00", "hash": "x",
                  "family_id": "family-b"},
        },
        "tokens": {
            family._token_key(token): {
                "username": username,
                "family_id": family_id,
                "created": time.time(),
            }
            for family_id, token in tokens.items()
            for username in (family_id[-1],)
        },
    }), encoding="utf-8")
    monkeypatch.setattr(family, "ACCOUNTS_FILE", accounts)
    monkeypatch.setattr(
        server, "_model_status",
        lambda family_id=None: {"voices": [{"name": f"voice-{family_id}"}]},
    )
    monkeypatch.setattr(server, "_enrolled", {
        "voice-family-a", "voice-family-b",
    })

    a = server.voices(
        request=_remote_request(),
        authorization="Bearer token-a",
    )
    b = server.voices(
        request=_remote_request(),
        authorization="Bearer token-b",
    )

    assert a["family_id"] == "family-a"
    assert a["voices"] == [{"name": "voice-family-a"}]
    assert b["family_id"] == "family-b"
    assert b["voices"] == [{"name": "voice-family-b"}]


def test_stream_factory_passes_family_voice_scope(monkeypatch):
    import api.server as server
    from fusion.fusion_orchestrator import ChannelVerdict
    from server.scheduler import make_mock_backends

    observed = {}

    def fake_make_real_backends(pipe, **kwargs):
        observed.update(kwargs)
        stub = lambda _w: ChannelVerdict(
            "stub", 0.0, "stub", "未接入通道", 0.0)
        return make_mock_backends(
            stub, stub, lambda _w, _prev: (stub(_w), ""))

    monkeypatch.setattr(server, "get_pipe", lambda: SimpleNamespace())
    monkeypatch.setattr(
        server, "_allowed_voice_names", lambda _pipe, _family: ["妈妈"])
    monkeypatch.setattr(
        server, "make_real_backends", fake_make_real_backends)

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_bytes(b"\x00\x00" * 48000)
            assert ws.receive_json()["type"] == "window"

    assert observed["allowed_names"] == ["妈妈"]
