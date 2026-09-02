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

from .audio_util import pcm16_to_float32


def register_stream_ws(app, processor_factory):
    """把 /ws/stream 挂到 FastAPI app 上。

    processor_factory: 无参工厂，返回一个新的 StreamProcessor（每连接独立状态）。
    """

    @app.websocket("/ws/stream")
    async def stream_ws(websocket):  # noqa: N805  (FastAPI websocket endpoint)
        await websocket.accept()
        proc = processor_factory()
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                # 文本帧 = 控制消息（JSON）
                if msg.get("text") is not None:
                    try:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "reset":
                            proc.reset()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                # 二进制帧 = PCM 音频
                data = msg.get("bytes")
                if not data:
                    continue
                samples = pcm16_to_float32(data)
                events = proc.push_chunk(samples)
                for ev in events:
                    await websocket.send_json(ev)
        except Exception:  # noqa: BLE001  (客户端断开等)
            pass
        finally:
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass
