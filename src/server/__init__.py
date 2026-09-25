# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 流式实时推理服务（P1-1 核心）
============================================
把"整段文件分析"升级为"边说边判"的流式管线：

    浏览器/文件音频流 ──▶ WindowBuffer(3s 窗/1s 步进)
                          └─▶ Scheduler(常驻串行跑三通道)
                               └─▶ IncrementalFusion(EMA 平滑 + 迟滞状态机)
                                    └─▶ WS / 回放 → 实时绿/黄/红事件

设计原则（与 P0-4 一致）：所有重依赖（torch / funasr / Ollama）均懒加载，
模块在纯 numpy 环境下即可导入与单测；无 GPU/模型时走规则/缓存降级。
"""
from __future__ import annotations

from .window_buffer import WindowBuffer, Window
from .incremental_fusion import IncrementalFusion
from .scheduler import make_mock_backends
from .stream_pipeline import StreamProcessor
from .ws_api import register_stream_ws

__all__ = [
    "WindowBuffer", "Window", "IncrementalFusion",
    "make_mock_backends", "StreamProcessor",
    "register_stream_ws",
]
