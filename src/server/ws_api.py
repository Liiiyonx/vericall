# -*- coding: utf-8 -*-
"""
WebSocket 流式接口：/ws/stream
================================
协议：
  客户端 → 服务端：二进制帧 = 16-bit 小端 PCM 单声道 @16kHz（前端录音器分片送）。
                 首帧可先发 JSON 控制消息 {"type":"config","sr":16000}。
  服务端 → 客户端：JSON 事件，逐窗推送：
        {"type":"window","idx":7,"t_rel":21.0,
         "channels":{"acoustic":{"score":.91,"label":"spoof"},
                     "voiceprint":{"score":.12,"label":"match"},
                     "semantic":{"score":.83,"label":"money_request","transcript":"…急用五万…"}},
         "fused":0.74,"state":"YELLOW"}
        {"type":"alert","state":"RED","rationale":"…","advice":"建议挂断后回拨确认"}
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from api.history_store import append_history, next_id
from api.family_api import authorize_websocket
from paths import DEFAULT_FAMILY_ID

from .audio_util import pcm16_to_float32

_stream_connections_lock = threading.Lock()
_active_stream_connections = 0


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _acquire_stream_slot() -> bool:
    global _active_stream_connections
    limit = _env_positive_int("VERICALL_MAX_STREAM_CONNECTIONS", 4)
    with _stream_connections_lock:
        if _active_stream_connections >= limit:
            return False
        _active_stream_connections += 1
        return True


def _release_stream_slot() -> None:
    global _active_stream_connections
    with _stream_connections_lock:
        _active_stream_connections = max(0, _active_stream_connections - 1)


def _channel_list(channels) -> list[dict]:
    """把增量事件的 channel 字典规范成 /api/analyze 同构的列表。"""
    if isinstance(channels, list):
        return channels
    if isinstance(channels, dict):
        return [{"name": name, **value} for name, value in channels.items()]
    return []


def _stream_alert_record(ev: dict, last_window: dict | None = None,
                         family_id: str = DEFAULT_FAMILY_ID) -> dict:
    """把流式 RED 事件映射成子女端可消费的历史告警。"""
    window = last_window or {}
    return {
        "id": next_id(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": "常开守护（流式）",
        "source": "stream",
        "family_id": family_id,
        "final": "block",  # RED -> block：与 family_alerts 的既有口径对齐
        "score": ev.get("fused", window.get("fused")),
        "rationale": ev.get("rationale", ""),
        "advice": ev.get("advice", ""),
        "channels": _channel_list(
            ev.get("channels", window.get("channels", []))),
    }


def register_stream_ws(app, processor_factory,
                       family_id: str = DEFAULT_FAMILY_ID):
    """把 /ws/stream 挂到 FastAPI app 上。

    processor_factory: 接收 ``(采样率, family_id)`` 的工厂，返回一个新的
        StreamProcessor（每连接独立状态）。远端客户端必须使用 Bearer 或
        家庭绑定的一次性票据；仅本机父母端保留免登录兼容路径。
    """

    @app.websocket("/ws/stream")
    async def stream_ws(websocket: WebSocket):  # noqa: N805  (FastAPI websocket endpoint)
        try:
            resolved_family_id, _ = authorize_websocket(
                websocket, purpose="stream")
        except HTTPException:
            await websocket.close(code=4401)
            return
        if not _acquire_stream_slot():
            await websocket.close(code=4429)
            return
        await websocket.accept()
        proc = None
        sample_rate = 16000
        last_window = None
        processing_failed = False
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                # 文本帧 = 控制消息（JSON）
                if msg.get("text") is not None:
                    if len(msg["text"].encode("utf-8")) > 16 * 1024:
                        await websocket.send_json(
                            {"type": "error", "code": "control_frame_too_large"})
                        await websocket.close(code=1009)
                        break
                    try:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "config":
                            try:
                                new_sr = int(ctrl.get("sr", 16000))
                            except (TypeError, ValueError):
                                new_sr = 0
                            if not 8000 <= new_sr <= 48000:
                                await websocket.send_json(
                                    {"type": "error", "code": "bad_sample_rate"})
                                continue
                            if proc is not None and getattr(proc, "sr", sample_rate) != new_sr:
                                await websocket.send_json(
                                    {"type": "error", "code": "config_after_audio"})
                                continue
                            sample_rate = new_sr
                        elif ctrl.get("type") == "reset" and proc is not None:
                            proc.reset()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                # 二进制帧 = PCM 音频
                data = msg.get("bytes")
                if not data:
                    continue
                max_frame = _env_positive_int(
                    "VERICALL_MAX_STREAM_FRAME_MB", 2) * 1024 * 1024
                if len(data) > max_frame:
                    await websocket.send_json(
                        {"type": "error", "code": "audio_frame_too_large",
                         "message": "音频分片过大，请缩短单次发送时长"})
                    await websocket.close(code=1009)
                    break
                if proc is None:
                    # 首次连接可能加载模型/权重，同样不能占用事件循环。
                    proc = await run_in_threadpool(
                        processor_factory, sample_rate,
                        resolved_family_id or family_id)
                samples = pcm16_to_float32(data)
                # 推理链包含同步 HTTP/GPU/ASR，不能阻塞 asyncio 事件循环。
                events = await run_in_threadpool(proc.push_chunk, samples)
                for ev in events:
                    if ev.get("type") == "window":
                        last_window = ev
                    elif ev.get("type") == "alert":
                        try:
                            await run_in_threadpool(
                                append_history,
                                _stream_alert_record(
                                    ev, last_window,
                                    family_id=resolved_family_id or family_id),
                            )
                        except Exception as e:  # noqa: BLE001
                            # 历史盘不可写不应打断正在进行的常开守护。
                            print(f"[WS] 流式告警落盘失败: {e}")
                    await websocket.send_json(ev)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001  (客户端断开等)
            processing_failed = True
            print(f"[WS] 流式处理失败: {type(exc).__name__}: {exc}")
            try:
                await websocket.send_json(
                    {"type": "error", "code": "processing_failed",
                     "message": "流式分析暂时不可用，请稍后重试"})
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                if proc is not None:
                    close = getattr(proc, "close", None)
                    if callable(close):
                        try:
                            await run_in_threadpool(close)
                        except Exception:  # noqa: BLE001
                            pass
                try:
                    await websocket.close(
                        code=1011 if processing_failed else 1000)
                except Exception:  # noqa: BLE001
                    pass
            finally:
                _release_stream_slot()
