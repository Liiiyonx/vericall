# -*- coding: utf-8 -*-
"""test_streaming_regression.py — 流式回归护栏（2026-09-07）

1) transcript 滚动截断：修复长流 MemoryError（压测发现）的回归测试——语义后端
   每窗返回长文本时，累积串不得超过上限 1500 字；
2) 离线降级 A/B/C 三场景保底（VERICALL_OFFLINE=1，用 demo_cache，零 GPU）。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from fusion.fusion_orchestrator import ChannelVerdict
from server.stream_pipeline import StreamProcessor


class TestTranscriptCap:
    def _proc(self):
        p = StreamProcessor.__new__(StreamProcessor)
        p.sr = 16000
        p.transcript = ""
        p.transcript_max = 1500
        return p

    def test_cap_holds(self):
        p = self._proc()
        w = SimpleNamespace(t_rel=1.0, idx=0)
        verdict = ChannelVerdict("acoustic", 0.1, "normal", "x", 1.0)
        p.backends = {
            "acoustic": lambda w: verdict,
            "voiceprint": lambda w: verdict,
            "semantic": lambda w, tr: (verdict, "长" * 400),
        }
        # 语义通道需被 fusion.update 消费——用最小桩替代
        seen = []
        class _FakeFusion:
            def update(self, ac, vp, sem, t_rel, idx):
                seen.append(1)
                return [{"type": "window", "t_rel": t_rel, "fused": 0.0,
                         "state": "GREEN", "channels": {}}]
        p.fusion = _FakeFusion()
        for _ in range(20):
            p._process_window(w)
        assert len(seen) == 20
        assert len(p.transcript) <= 1500, f"超上限: {len(p.transcript)}"
        # 截断后保留尾部
        assert p.transcript.endswith("长")


class TestOfflineScenarios:
    def test_abc_offline(self):
        os.environ["VERICALL_OFFLINE"] = "1"
        from fusion.pipeline import VeriCallPipeline
        p = VeriCallPipeline()
        expect = {"A": "allow", "B": "block", "C": "caution"}
        for scen, want in expect.items():
            r = p.analyze("", scenario=scen)
            assert r.offline is True
            assert r.final == want, f"场景{scen}: 期望{want} 实得{r.final}"
        del os.environ["VERICALL_OFFLINE"]
