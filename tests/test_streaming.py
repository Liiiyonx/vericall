# -*- coding: utf-8 -*-
"""P1 流式端到端回归：绿→黄→红 时间线 + 零误拦（纯 numpy，可离线跑）。

对应任务书 P1-1e 验收：
  - 60s 诈骗信号（前 30s 正常 / 后 30s 诈骗）：进入要钱段后连续 2 窗黄-demand
    → 2 窗内升级黄（迟滞窗口数），再 2 窗内升级红（共 4 窗内红）；
  - 60s 正常家人闲聊：全程绿，零误拦（无 YELLOW/RED，无 alert）；
  - 每窗端到端处理延迟（mock 后端）P90 ≤ 2s（实时性代理指标）。

EMA 平滑（alpha=0.6）会让首个诈骗窗的融合分被前序绿窗稀释，因此"进入要钱段"
的首窗通常是绿-demand，第 2 窗起黄-demand 累积、第 2 窗（迟滞）翻黄、第 4 窗翻红，
与任务书口径一致（"2 个窗"= 2 窗黄-demand 迟滞，"4 个窗"= 第 4 窗红）。
"""
import numpy as np
import pytest

from fusion.fusion_orchestrator import ChannelVerdict
from server.scheduler import make_mock_backends
from server.stream_pipeline import StreamProcessor
from server.incremental_fusion import GREEN, YELLOW, RED

SR = 16000
SCAM_ONSET = 30.0  # 后端在第 30s 切换为高危（要钱段起点）


def _synth_signal(sr, dur=60.0):
    """合成演示信号：前 30s 家人闲聊（低危），后 30s 叠加高频异常（诈骗痕迹）。"""
    t = np.arange(int(sr * dur)) / sr
    sig = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)
    seg = int(sr * SCAM_ONSET)
    sig[seg:] += (0.2 * np.sin(2 * np.pi * 3000 * t[seg:])).astype(np.float32)
    return sig


def _scam_backends():
    """后 30s 诈骗：检测分数逐步升高（中危→硬命中），模拟真实要钱段的渐进取证。

    第 30s 中危(0.5) → 31s 中危(0.6，触发单通道升级) → 32s 起硬命中(0.9/0.85)。
    这样能完整跑出 绿→黄(迟滞 2 窗)→红 的迟滞状态机，而非首窗直接硬拦。
    """
    def score(t):
        if t < SCAM_ONSET:
            return 0.05
        if t == SCAM_ONSET:
            return 0.5
        if t in (SCAM_ONSET + 1, SCAM_ONSET + 2):
            return 0.6
        return 0.9
    def sem_score(t):
        if t < SCAM_ONSET:
            return 0.05
        if t == SCAM_ONSET:
            return 0.5
        if t in (SCAM_ONSET + 1, SCAM_ONSET + 2):
            return 0.6
        return 0.85
    def ac(w):
        t = round(w.t_rel)
        s = score(t)
        return ChannelVerdict("acoustic", s, "spoof" if s >= 0.5 else "bonafide",
                              "AASIST", 0.95)
    def vp(w):
        return ChannelVerdict("voiceprint", 0.1, "match", "声纹一致", 0.9)
    def sem(w, prev):
        t = round(w.t_rel)
        s = sem_score(t)
        if s >= 0.5:
            return ChannelVerdict("semantic", s, "money_request",
                                  "冒充熟人要钱", 0.9), "妈我是同学号急用五万块"
        return ChannelVerdict("semantic", 0.05, "normal", "正常闲聊", 0.9), ""
    return make_mock_backends(ac, vp, sem)


def _normal_backends():
    """全程低危：正常家人闲聊，三通道均正常。"""
    def ac(w):
        return ChannelVerdict("acoustic", 0.05, "bonafide", "真实人声", 0.9)
    def vp(w):
        return ChannelVerdict("voiceprint", 0.1, "match", "声纹一致", 0.9)
    def sem(w, prev):
        return ChannelVerdict("semantic", 0.05, "normal", "正常闲聊", 0.9), ""
    return make_mock_backends(ac, vp, sem)


def _split(events):
    windows = [e for e in events if e["type"] == "window"]
    alerts = [e for e in events if e["type"] == "alert"]
    return windows, alerts


def test_scam_timeline_yellow_then_red():
    """诈骗信号：进入要钱段 2 窗内升级黄、4 窗内升级红，且提前无误判。"""
    proc = StreamProcessor(backends=_scam_backends(), family_name="妈")
    events = proc.process_signal(_synth_signal(SR))
    windows, alerts = _split(events)
    assert windows, "必须产生窗口事件"

    onset_idx = next(i for i, e in enumerate(windows) if e["t_rel"] >= SCAM_ONSET)
    yellow_idxs = [i for i, e in enumerate(windows) if e["state"] == YELLOW]
    red_idxs = [i for i, e in enumerate(windows) if e["state"] == RED]
    assert yellow_idxs, "必须出现 YELLOW 状态"
    assert red_idxs, "必须出现 RED 状态"

    # 2 窗内升级黄：首个 YELLOW 不晚于 onset+2
    assert yellow_idxs[0] <= onset_idx + 2, \
        f"黄态过晚: yellow@{yellow_idxs[0]} onset@{onset_idx}"
    # 4 窗内升级红：首个 RED 不晚于 onset+3（第 4 个窗）
    assert red_idxs[0] <= onset_idx + 3, \
        f"红态过晚: red@{red_idxs[0]} onset@{onset_idx}"
    # 进入要钱段之前不得提前误判
    assert all(e["state"] == GREEN for e in windows[:onset_idx]), \
        "要钱段之前不应出现非绿态"
    # 必须发出至少一次适老告警
    assert alerts, "必须发出 RED 告警"
    assert any("挂断" in a.get("advice", "") for a in alerts), \
        "告警需含适老建议（挂断确认）"


def test_normal_chat_all_green_no_false_block():
    """正常家人闲聊：全程绿、零误拦（无 YELLOW/RED，无告警）。"""
    proc = StreamProcessor(backends=_normal_backends(), family_name="妈")
    events = proc.process_signal(_synth_signal(SR))
    windows, alerts = _split(events)
    states = [e["state"] for e in windows]
    assert states.count(GREEN) == len(states), "正常闲聊必须全程绿"
    assert not alerts, "正常闲聊零告警（零误拦）"
    assert YELLOW not in states and RED not in states


def test_per_window_latency_within_budget():
    """每窗端到端延迟代理：mock 后端下每窗处理时间 P90 ≤ 2s。"""
    import time
    proc = StreamProcessor(backends=_scam_backends(), family_name="妈")
    sig = _synth_signal(SR)
    chunk = SR  # 1s 流式分块
    per_window = []
    for i in range(0, len(sig), chunk):
        t0 = time.process_time()
        evs = proc.push_chunk(sig[i:i + chunk])
        dt = time.process_time() - t0
        for _ in evs:
            per_window.append(dt)
    proc.flush()
    assert per_window, "应产生窗口事件"
    ordered = sorted(per_window)
    p90 = ordered[int(0.9 * (len(ordered) - 1))]
    assert p90 <= 2.0, f"每窗延迟 P90={p90:.4f}s 超出 2s 预算"


def test_stream_reset_clears_state():
    """reset() 后状态机归零：重新从绿起步，再次走入红。"""
    proc = StreamProcessor(backends=_scam_backends(), family_name="妈")
    proc.process_signal(_synth_signal(SR))
    assert proc.fusion.current_state == RED
    proc.reset()
    assert proc.fusion.current_state == GREEN
    # 重新跑同一诈骗信号，首窗应为绿、全程仍能走到红
    evs = proc.process_signal(_synth_signal(SR))
    windows, _ = _split(evs)
    assert windows[0]["state"] == GREEN
    assert any(e["state"] == RED for e in windows)
