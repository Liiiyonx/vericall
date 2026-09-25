# -*- coding: utf-8 -*-
"""
流式调度器：把每个窗口派发给三个通道后端，串行产出 ChannelVerdict。
================================================================
后端解耦：测试用 make_mock_backends 注入确定性函数；线上用 make_real_backends
包装现有 VeriCallPipeline 的三通道（写临时 WAV → channel.analyze，懒加载、不卸载）。
"""
from __future__ import annotations

from typing import Callable, Iterable, Tuple
from pathlib import Path
import os
import shutil
import tempfile
import time

import numpy as np

from fusion.fusion_orchestrator import ChannelVerdict
from .window_buffer import Window

# 声学/声纹后端：给窗口，返回该窗的 ChannelVerdict
Backend = Callable[[Window], ChannelVerdict]
# 语义后端：给窗口 + 累积转写，返回 (ChannelVerdict, 本窗新转写文本)
SemanticBackend = Callable[[Window, str], Tuple[ChannelVerdict, str]]


def _cleanup_stale_stream_artifacts(root: Path, max_age_s: float = 3600.0) -> None:
    """清理上次异常退出遗留的旧 WAV 与连接目录。"""
    now = time.time()
    for pattern, is_dir in (("win_*.wav", False), ("stream_*", True)):
        for stale in root.glob(pattern):
            try:
                if now - stale.stat().st_mtime <= max_age_s:
                    continue
                if is_dir:
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink(missing_ok=True)
            except OSError:
                pass


def make_mock_backends(ac_fn: Callable[[Window], ChannelVerdict],
                        vp_fn: Callable[[Window], ChannelVerdict],
                        sem_fn: Callable[[Window, str], Tuple[ChannelVerdict, str]]
                        ) -> dict:
    """测试用：注入确定性后端。"""
    return {
        "acoustic": ac_fn,
        "voiceprint": vp_fn,
        "semantic": sem_fn,
    }


def make_real_backends(pipe, offline: bool = False, sr: int = 16000,
                       llm_interval_s: float = 5.0,
                       llm_trigger_risk: float = 0.5,
                       llm_trigger_gap_s: float = 2.0,
                       allowed_names: Iterable[str] | None = None) -> dict:
    """线上用：包装 VeriCallPipeline 的真实三通道。每窗写临时 WAV 后推理。

    pipe: VeriCallPipeline 实例（含 .ac / .vp / .sem / .orch）
    offline: True 时语义通道走规则评分器，跳过 Ollama/SenseVoice 实时调用。
    sr: 窗口采样率——必须与 WindowBuffer 的 sample_rate 一致（默认 16000）。
        不要从 窗长/时长 反推：flush() 尾窗时长非整秒时会得到错误 sr
        （如 2.9s 窗算出 23200Hz），三通道会收到变速音频。
    llm_interval_s: LLM 深评最小间隔（任务书 P1-1 节流：非每窗都调 r1）。
    llm_trigger_risk / llm_trigger_gap_s: 规则快筛 risk 达阈值时，距上次
        LLM 超过 gap 秒即可提前触发一次深评（不放过突发高危话术）。
    allowed_names: 当前家庭允许参与声纹比对的家人名称；None 仅用于单机库调用，
        服务端请求必须传入家庭范围，避免跨家庭误匹配。
    """
    import sys
    import time as _time
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from .audio_util import write_wav

    _tmp = _P(tempfile.gettempdir()) / "vericall_stream"
    _tmp.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_stream_artifacts(_tmp)

    session_dir = _P(tempfile.mkdtemp(prefix="stream_", dir=str(_tmp)))
    paths = {
        "acoustic": session_dir / "acoustic.wav",
        "voiceprint": session_dir / "voiceprint.wav",
        "semantic": session_dir / "semantic.wav",
    }
    _llm = {"last": -1e9}   # 上次 LLM 深评的 monotonic 时间戳

    def _with_temp_wav(w: Window, channel: str,
                       fn: Callable[[str], ChannelVerdict]) -> ChannelVerdict:
        path = paths[channel]
        write_wav(str(path), w.samples, sr=sr)
        try:
            # 让新连接启动时不会把仍在写入的长连接目录误判为陈旧。
            os.utime(session_dir, None)
        except OSError:
            pass
        return fn(str(path))

    def ac_b(w: Window) -> ChannelVerdict:
        return _with_temp_wav(
            w, "acoustic",
            lambda path: pipe.ac.analyze(path, unload_after=False))  # AASIST 常驻

    def vp_b(w: Window) -> ChannelVerdict:
        if allowed_names is None:
            return _with_temp_wav(w, "voiceprint", pipe.vp.verify)
        return _with_temp_wav(
            w, "voiceprint",
            lambda path: pipe.vp.verify(path, allowed_names=allowed_names))

    def _sem_analyze(path: str, prev: str) -> Tuple[ChannelVerdict, str]:
        if offline:
            from fusion.transcript_cache import get as cache_get, put as cache_put
            try:
                txt = pipe.sem.asr.transcribe(path) if hasattr(pipe.sem, "asr") else ""
            except Exception:  # noqa: BLE001
                txt = ""
            if not txt:
                txt = cache_get(path) or ""
            else:
                cache_put(path, txt)
            res = pipe.sem._rule_score(txt)
            cv = ChannelVerdict("semantic", res.risk, res.category,
                                f"{res.reason} | 转写: {txt[:40]}", 0.9)
            return cv, txt
        # 在线：ASR 本窗 → 累积转写 → 规则快筛 →（按节流）LLM 深评
        try:
            txt = pipe.sem.asr.transcribe(path)
        except Exception:  # noqa: BLE001
            from fusion.transcript_cache import get as cache_get
            txt = cache_get(path) or ""
        new_text = (prev + " " + txt).strip() if txt else prev
        if not new_text:
            res = pipe.sem._fallback("ASR 转写为空", __import__("time").time())
            cv = ChannelVerdict("semantic", res.risk, res.category,
                                res.reason, 0.3)
            return cv, ""

        # 1) 规则快筛：每窗必跑（微秒级），保证任何情况下本窗都有语义分
        rule = pipe.sem._rule_score(new_text)
        # 2) LLM 节流判定：到点（interval）或规则触发（高危+最小间隔）
        now = _time.monotonic()
        due = (now - _llm["last"] >= llm_interval_s) or (
            rule.risk >= llm_trigger_risk
            and now - _llm["last"] >= llm_trigger_gap_s)
        if due:
            _llm["last"] = now
            res = pipe.sem.analyze(new_text)   # 内部已处理降级链
            if res.category == "unknown" and rule.risk >= 0.2:
                # LLM 失败/无法解析而规则有命中：用规则结果，不丢真实风险信号
                res = rule
            cv = ChannelVerdict("semantic", res.risk, res.category,
                                f"{res.reason} | 转写: {new_text[:40]}", 0.9)
        else:
            cv = ChannelVerdict("semantic", rule.risk, rule.category,
                                f"[规则快筛] {rule.reason} | 转写: {new_text[:40]}",
                                0.6)
        return cv, new_text

    def sem_b(w: Window, prev: str) -> Tuple[ChannelVerdict, str]:
        return _with_temp_wav(
            w, "semantic", lambda path: _sem_analyze(path, prev))

    def _cleanup() -> None:
        shutil.rmtree(session_dir, ignore_errors=True)

    return {
        "acoustic": ac_b,
        "voiceprint": vp_b,
        "semantic": sem_b,
        "_cleanup": _cleanup,
    }
