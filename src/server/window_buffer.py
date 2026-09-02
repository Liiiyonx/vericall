# -*- coding: utf-8 -*-
"""
音频窗口缓冲：3 秒滑窗 / 1 秒步进重叠组装。
============================================
AASIST 输入 nb_samp=48000 = 16kHz 下整 3 秒，故 3s 滑窗与模型输入天然对齐，零改造。
前端每 3s 切一段（含 1s 重叠由服务端组装），服务端累积后按此规则吐窗。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    idx: int          # 第几窗（从 0 起）
    t_rel: float      # 相对起始的时间戳（秒）
    samples: np.ndarray  # float32, 单声道, 长度 = window_samples
    duration: float   # 秒


class WindowBuffer:
    def __init__(self, sample_rate: int = 16000,
                 window_s: float = 3.0, step_s: float = 1.0):
        self.sr = sample_rate
        self.win = int(round(window_s * sample_rate))
        self.step = int(round(step_s * sample_rate))
        if self.step <= 0 or self.step > self.win:
            self.step = self.win  # 退化：不重叠
        self.buf = np.zeros(0, dtype=np.float32)
        self._idx = 0

    def push(self, chunk: np.ndarray) -> list[Window]:
        """塞入一坨 PCM 样本，返回本次可吐出的完整窗口列表。"""
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        if chunk.size == 0:
            return []
        self.buf = np.concatenate([self.buf, chunk])
        out: list[Window] = []
        while len(self.buf) >= self.win:
            w = self.buf[:self.win].copy()
            out.append(Window(self._idx,
                              self._idx * self.step / self.sr,
                              w, self.win / self.sr))
            self._idx += 1
            self.buf = self.buf[self.step:]
        return out

    def flush(self) -> list[Window]:
        """吐出末尾残余（≥ 半窗才吐，避免极短碎片）。"""
        out: list[Window] = []
        if len(self.buf) >= self.win // 2:
            out.append(Window(self._idx,
                              self._idx * self.step / self.sr,
                              self.buf.copy(), len(self.buf) / self.sr))
            self._idx += 1
            self.buf = np.zeros(0, dtype=np.float32)
        return out

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self._idx = 0

    @classmethod
    def from_file(cls, path, sample_rate: int = 16000,
                  window_s: float = 3.0, step_s: float = 1.0) -> list[Window]:
        """回放模式：把整段文件切成窗口（懒加载 soundfile）。"""
        from .audio_util import read_wav
        data = read_wav(path, sr=sample_rate)
        buf = cls(sample_rate, window_s, step_s)
        buf.push(data)
        return buf.flush()
