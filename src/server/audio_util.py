# -*- coding: utf-8 -*-
"""音频格式互转工具（懒加载 soundfile，纯 numpy 可用核心）。"""
from __future__ import annotations

import numpy as np


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """16-bit 小端 PCM 字节 → float32 [-1,1]（前端 WS 传输格式）。"""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 [-1,1] → 16-bit 小端 PCM 字节。"""
    s = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (s * 32767.0).astype("<i2").tobytes()


def resample(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """重采样；优先 scipy.signal.resample_poly，缺失则线性插值兜底。"""
    if from_sr == to_sr:
        return np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
        return np.asarray(resample_poly(x, to_sr, from_sr), dtype=np.float32)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * to_sr / from_sr))
        return np.interp(np.linspace(0, len(x) - 1, max(n, 1)),
                         np.arange(len(x)), x).astype(np.float32)


def write_wav(path, samples: np.ndarray, sr: int = 16000) -> None:
    """写 16k 单声道 WAV（soundfile 懒加载）。"""
    import soundfile as sf
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sr)


def read_wav(path, sr: int = 16000) -> np.ndarray:
    """读任意采样率 WAV → 单声道 float32 @sr。"""
    import soundfile as sf
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    return resample(data, file_sr, sr)
