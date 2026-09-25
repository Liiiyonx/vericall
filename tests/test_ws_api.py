# -*- coding: utf-8 -*-
"""WebSocket 流式接口回归：线程池隔离、config 控制帧和路由可达性。"""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.family_api as family
import api.history_store as history_store
from server.ws_api import register_stream_ws


class _Proc:
    def __init__(self, sample_rate: int, observed: dict,
                 events: list[dict] | None = None):
        try:
            asyncio.get_running_loop()
            observed["factory_on_event_loop"] = True
        except RuntimeError:
            observed["factory_on_event_loop"] = False
        self.sr = sample_rate
        self.observed = observed
        self.events = events

    def push_chunk(self, samples):
        try:
            asyncio.get_running_loop()
            self.observed["on_event_loop"] = True
        except RuntimeError:
            self.observed["on_event_loop"] = False
        if self.events is not None:
            return self.events
        return [{"type": "window", "samples": len(samples)}]

    def reset(self):
        self.observed["reset"] = True

    def close(self):
        self.observed["closed"] = True


def test_ws_stream_offloads_blocking_processor():
    observed = {}
    app = FastAPI()
    register_stream_ws(app, lambda sr, family_id: _Proc(sr, observed))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_bytes(b"\x00\x00")
            assert ws.receive_json()["type"] == "window"

    assert observed["on_event_loop"] is False
    assert observed["factory_on_event_loop"] is False
    assert observed["closed"] is True


def test_ws_config_frame_controls_sample_rate():
    observed = {}
    app = FastAPI()
    register_stream_ws(app, lambda sr, family_id: _Proc(sr, observed))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text('{"type":"config","sr":8000}')
            ws.send_bytes(b"\x00\x00")
            assert ws.receive_json()["type"] == "window"
            ws.send_text('{"type":"reset"}')

    assert observed["reset"] is True


def test_ws_red_alert_is_written_as_family_alert(tmp_path, monkeypatch):
    observed = {}
    hist = tmp_path / "history.jsonl"
    monkeypatch.setattr(history_store, "HIST_FILE", hist)
    monkeypatch.setattr(family, "HIST_FILE", hist)
    monkeypatch.setattr(family, "ACK_FILE", tmp_path / "acks.json")
    events = [
        {
            "type": "window",
            "idx": 3,
            "channels": {
                "acoustic": {"score": 0.91, "label": "spoof"},
                "voiceprint": {"score": 0.22, "label": "mismatch"},
                "semantic": {"score": 0.88, "label": "money_request"},
            },
            "fused": 0.88,
            "state": "RED",
        },
        {
            "type": "alert",
            "state": "RED",
            "rationale": "声学伪造与转账话术同时命中",
            "advice": "建议先挂断，拨打家人平时的号码确认一下。",
        },
    ]
    app = FastAPI()
    register_stream_ws(
        app, lambda sr, family_id: _Proc(sr, observed, events=events))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_bytes(b"\x00\x00")
            assert ws.receive_json()["type"] == "window"
            assert ws.receive_json()["type"] == "alert"

    records = list(history_store.iter_history(
        hist, family_id=family.DEMO_FAMILY_ID))
    assert len(records) == 1
    rec = records[0]
    assert rec["source"] == "stream"
    assert rec["final"] == "block"
    assert rec["score"] == 0.88
    assert [c["name"] for c in rec["channels"]] == [
        "acoustic", "voiceprint", "semantic"]
    assert rec["rationale"] == "声学伪造与转账话术同时命中"
    alerts = family.family_alerts(user={})["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["source"] == "stream"
    assert alerts[0]["final"] == "block"
