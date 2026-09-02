# -*- coding: utf-8 -*-
"""增量融合状态机：EMA 平滑 + 迟滞 绿/黄/红 + 告警冷却（纯 numpy）。"""
import numpy as np
import pytest

from fusion.fusion_orchestrator import ChannelVerdict
from server.incremental_fusion import IncrementalFusion, GREEN, YELLOW, RED


def _cv(name, score, label="x", conf=0.9):
    return ChannelVerdict(name, score, label, "detail", conf)


def _feed(f, ac, vp, sem, t=0.0):
    return f.update(_cv("acoustic", ac), _cv("voiceprint", vp), _cv("semantic", sem), t)


def test_green_stays_green_on_low():
    f = IncrementalFusion()
    for i in range(6):
        _feed(f, 0.05, 0.05, 0.05, t=i)
    assert f.state == GREEN


def test_green_to_yellow_needs_two_consecutive():
    f = IncrementalFusion()
    _feed(f, 0.6, 0.1, 0.1, t=0)    # 单通道中危 → YELLOW 需求，但需连续 2 窗
    assert f.state == GREEN
    _feed(f, 0.6, 0.1, 0.1, t=1)
    assert f.state == YELLOW


def test_hard_hit_immediate_red():
    f = IncrementalFusion()
    evs = _feed(f, 0.95, 0.1, 0.1, t=0)   # 声学硬命中 → 直接 RED
    assert f.state == RED
    assert any(e["type"] == "alert" for e in evs)


def test_yellow_to_red_via_fused():
    f = IncrementalFusion()
    _feed(f, 0.6, 0.1, 0.1, t=0)
    _feed(f, 0.6, 0.1, 0.1, t=1)         # → YELLOW
    assert f.state == YELLOW
    # 持续高危（无单通道>=0.85，靠 fused>=0.7 升级）
    _feed(f, 0.75, 0.75, 0.75, t=2)
    _feed(f, 0.78, 0.78, 0.78, t=3)
    _feed(f, 0.80, 0.80, 0.80, t=4)
    assert f.state == RED


def test_red_recovers_after_five_clean():
    f = IncrementalFusion()
    _feed(f, 0.95, 0.1, 0.1, t=0)        # RED
    for i in range(1, 6):
        _feed(f, 0.05, 0.05, 0.05, t=i)  # 5 窗干净
    assert f.state == GREEN


def test_alert_cooldown():
    f = IncrementalFusion(cooldown_s=10.0)
    evs1 = _feed(f, 0.95, 0.1, 0.1, t=0)      # 首次 RED → 告警
    assert any(e["type"] == "alert" for e in evs1)
    for i in range(1, 6):
        _feed(f, 0.05, 0.05, 0.05, t=i)       # 回到 GREEN
    evs2 = _feed(f, 0.95, 0.1, 0.1, t=7)       # 冷却期内再 RED → 不重复告警
    assert f.state == RED
    assert not any(e["type"] == "alert" for e in evs2)
    evs3 = _feed(f, 0.95, 0.1, 0.1, t=20)      # 冷却结束 → 再告警
    assert any(e["type"] == "alert" for e in evs3)


def test_ema_smoothing_dampens_single_spike():
    f = IncrementalFusion()
    _feed(f, 0.05, 0.05, 0.05, t=0)
    evs = _feed(f, 0.9, 0.05, 0.05, t=1)       # 单窗尖峰
    win = [e for e in evs if e["type"] == "window"][0]
    assert win["fused"] < 0.9                    # EMA 把尖峰压下来
    assert win["fused"] > 0.05
