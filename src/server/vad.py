# -*- coding: utf-8 -*-
"""
可选 VAD 分段（复用 funasr 的 fsmn-vad，零新增依赖）。
================================================
不可用时返回 None，调用方回退到固定 3s 滑窗。保留接口以便后续做"说话人停顿切句"。
"""
from __future__ import annotations

import numpy as np

_MODEL = None


def _get_model(sample_rate: int = 16000):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from funasr import AutoModel
    except Exception:  # noqa: BLE001
        return None
    _MODEL = AutoModel(
        model="fsmn-vad",
        model_revision="v2.0.4",
        disable_update=True,
        device="cpu",
    )
    return _MODEL


def vad_segments(samples: np.ndarray, sample_rate: int = 16000,
                 min_dur_s: float = 1.0) -> list[tuple[float, float]] | None:
    """返回 [(start_s, end_s), ...]；不可用返回 None。"""
    model = _get_model(sample_rate)
    if model is None:
        return None
    try:
        import torch
        with torch.no_grad():
            res = model.generate(input=np.asarray(samples, dtype=np.float32),
                                 cache={}, is_final=True,
                                 chunk_size=16000, encoder_chunk_look_back=4,
                                 decoder_chunk_look_back=1)
    except Exception:  # noqa: BLE001
        return None
    segs: list[tuple[float, float]] = []
    for r in res or []:
        for s, e in r.get("value", []):
            s_s, e_s = s / 1000.0, e / 1000.0
            if e_s - s_s >= min_dur_s:
                segs.append((s_s, e_s))
    return segs or None
