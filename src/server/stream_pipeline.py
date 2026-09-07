# -*- coding: utf-8 -*-
"""
流式处理器：把"音频流 → 窗口 → 三通道 → 增量融合"串成一条线。
============================================================
- push_chunk(samples): 喂一坨 PCM，返回本次产生的事件列表（window / alert）。
- process_file(path): 回放模式，切窗后跑完整时间线，返回事件列表（供测试/绘图）。
- 仅依赖 numpy 即可导入；真实通道后端在 make_real_backends 内懒加载。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from fusion.fusion_orchestrator import FusionOrchestrator
from .window_buffer import WindowBuffer
from .incremental_fusion import IncrementalFusion
from .scheduler import make_mock_backends


class StreamProcessor:
    def __init__(self, backends: dict | None = None,
                 orchestrator: Optional[FusionOrchestrator] = None,
                 sample_rate: int = 16000,
                 window_s: float = 3.0, step_s: float = 1.0,
                 family_name: str = "家人"):
        self.sr = sample_rate
        self.winbuf = WindowBuffer(sample_rate, window_s, step_s)
        self.fusion = IncrementalFusion(orchestrator, family_name=family_name)
        self.backends = backends or self._default_offline_backends()
        self.transcript = ""
        self.transcript_max = 1500  # 转写上下文上限（滚动截断，防长流内存泄漏 09-07）

    def _default_offline_backends(self) -> dict:
        """无任何后端时：全 stub（不误拦），保证导入即可跑通空流。"""
        from fusion.fusion_orchestrator import FusionOrchestrator, ChannelVerdict

        def stub(_w):
            return ChannelVerdict("stub", 0.0, "stub", "未接入通道", 0.0)
        return make_mock_backends(stub, stub, lambda w, p: (stub(w), ""))

    def set_backends(self, backends: dict) -> None:
        self.backends = backends

    def reset(self) -> None:
        self.winbuf.reset()
        self.fusion.reset()
        self.transcript = ""

    # ------------------------------------------------------------------ #
    def push_chunk(self, samples: np.ndarray) -> list[dict]:
        windows = self.winbuf.push(samples)
        events: list[dict] = []
        for w in windows:
            events.extend(self._process_window(w))
        return events

    def flush(self) -> list[dict]:
        windows = self.winbuf.flush()
        events: list[dict] = []
        for w in windows:
            events.extend(self._process_window(w))
        return events

    def _process_window(self, w) -> list[dict]:
        ac = self.backends["acoustic"](w)
        vp = self.backends["voiceprint"](w)
        sem, txt = self.backends["semantic"](w, self.transcript)
        if txt:
            self.transcript = (self.transcript + " " + txt).strip()
            if len(self.transcript) > self.transcript_max:
                # 滚动保留最近 transcript_max 字（语义只需近期上下文）
                self.transcript = self.transcript[-self.transcript_max:].lstrip()
        return self.fusion.update(ac, vp, sem, w.t_rel, idx=w.idx)

    # ------------------------------------------------------------------ #
    def process_file(self, path: str) -> list[dict]:
        """回放模式：切窗 → 跑完整时间线。"""
        self.reset()
        windows = WindowBuffer.from_file(path, self.sr,
                                         self.winbuf.win / self.sr,
                                         self.winbuf.step / self.sr)
        events: list[dict] = []
        for w in windows:
            events.extend(self._process_window(w))
        return events

    def process_signal(self, samples: np.ndarray) -> list[dict]:
        """内存信号（float32 @sr）直接跑完整时间线（CI / 合成回归用）。"""
        self.reset()
        windows = self.winbuf.push(np.asarray(samples, dtype=np.float32))
        windows += self.winbuf.flush()
        events: list[dict] = []
        for w in windows:
            events.extend(self._process_window(w))
        return events
