# -*- coding: utf-8 -*-
"""流式调度器三项修复的回归测试（2026-09-02）。

修复1：尾窗采样率透传——不再从 样本数/时长 反推 sr（2.5s 尾窗修复前会
        被写成 12800Hz，三通道收到变速音频）；
修复2：LLM 节流——规则器每窗快筛（微秒级），深评由 interval/trigger 双
        条件门控，不再每窗（1s 步进）都调 LLM；
修复3：AASIST 常驻——ac_b 以 unload_after=False 调用，不再每窗重载权重。
"""
import wave

import numpy as np

from fusion.fusion_orchestrator import ChannelVerdict
from server.scheduler import make_real_backends
from server.stream_pipeline import StreamProcessor
from server.window_buffer import Window

SR = 16000
# 规则评分：money_request(五万/卡号=2) + urgency(来不及=1) → 0.15*2+0.25*2=0.8
SCAM_TEXT = "你先转五万到这个卡号，别告诉我爸，要不来不及了"


class _StubAc:
    def __init__(self):
        self.calls = []          # [(path, unload_after)]

    def analyze(self, path, unload_after=True):
        self.calls.append((path, unload_after))
        return ChannelVerdict("acoustic", 0.05, "bonafide", "stub", 0.9)


class _StubVp:
    def verify(self, path):
        return ChannelVerdict("voiceprint", 0.1, "match", "stub", 0.9)


class _StubAsr:
    def __init__(self, text):
        self._text = text

    def transcribe(self, path):
        return self._text


class _StubSem:
    """在线语义通道桩：规则/兜底用真实实现，analyze 只计数不发 HTTP。"""

    def __init__(self, text):
        from fusion.semantic_channel import SemanticChannel
        self._real = SemanticChannel()
        self.asr = _StubAsr(text)
        self.analyze_calls = 0

    def _rule_score(self, text):
        return self._real._rule_score(text)

    def _fallback(self, why, t0=None):
        return self._real._fallback(why, t0)

    def analyze(self, text):
        self.analyze_calls += 1
        return self._real._rule_score(text)


class _StubPipe:
    def __init__(self, asr_text=""):
        self.ac = _StubAc()
        self.vp = _StubVp()
        self.sem = _StubSem(asr_text)


def _win(idx, seconds):
    n = int(SR * seconds)
    return Window(idx=idx, t_rel=float(idx),
                  samples=np.zeros(n, dtype=np.float32), duration=seconds)


def _n_windows(events):
    return len([e for e in events if e["type"] == "window"])


# ---------------------------------------------------------------------- #
def test_tail_window_written_with_true_sample_rate():
    """修复1回归：非整秒尾窗（2.5s）写出的 wav 头必须是 16000Hz。"""
    pipe = _StubPipe()
    backends = make_real_backends(pipe, offline=False, sr=SR)
    backends["acoustic"](_win(0, 2.5))
    path = pipe.ac.calls[-1][0]
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SR, \
            f"尾窗采样率错误: {wf.getframerate()}（修复前为 24000//2=12000 类错误值）"


def test_stream_acoustic_resident_no_reload():
    """修复3回归：流式声学通道以 unload_after=False 调用（常驻不重载）。"""
    pipe = _StubPipe()
    backends = make_real_backends(pipe, offline=False, sr=SR)
    backends["acoustic"](_win(0, 3.0))
    assert pipe.ac.calls, "声学后端未被调用"
    assert all(ua is False for _, ua in pipe.ac.calls), \
        "流式声学通道必须 unload_after=False（常驻）"


# ---------------------------------------------------------------------- #
def test_llm_throttled_by_interval_and_gap():
    """修复2回归：interval 未到期且 trigger gap 未过时，LLM 全程仅首窗 1 次。"""
    pipe = _StubPipe(SCAM_TEXT)
    backends = make_real_backends(pipe, offline=False, sr=SR,
                                  llm_interval_s=3600.0,   # interval 永不触发
                                  llm_trigger_risk=0.5,
                                  llm_trigger_gap_s=2.0)   # 处理过快,gap 内不重触发
    proc = StreamProcessor(backends=backends)
    events = proc.process_signal(np.zeros(int(SR * 10.0), dtype=np.float32))
    assert _n_windows(events) >= 8, "应产生 ≥8 个窗口"
    assert pipe.sem.analyze_calls == 1, \
        f"LLM 深评次数={pipe.sem.analyze_calls}，期望 1（修复前 = 窗口数）"


def test_llm_trigger_path_fires_on_high_rule_risk():
    """修复2触发路径：规则高危且 gap=0 时，每窗都允许深评（不丢突发高危）。"""
    pipe = _StubPipe(SCAM_TEXT)
    backends = make_real_backends(pipe, offline=False, sr=SR,
                                  llm_interval_s=3600.0,
                                  llm_trigger_risk=0.5,
                                  llm_trigger_gap_s=0.0)   # 高危即触发
    proc = StreamProcessor(backends=backends)
    events = proc.process_signal(np.zeros(int(SR * 6.0), dtype=np.float32))
    n = _n_windows(events)
    assert n >= 4
    assert pipe.sem.analyze_calls == n, \
        f"触发路径下深评次数={pipe.sem.analyze_calls} 应等于窗口数 {n}"
