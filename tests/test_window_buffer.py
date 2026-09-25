# -*- coding: utf-8 -*-
"""窗口缓冲：3s 窗 / 1s 步进 的切分正确性（纯 numpy）。"""
import numpy as np
import pytest

from server.window_buffer import WindowBuffer

SR = 16000


def test_push_emits_full_windows():
    b = WindowBuffer(SR, 3, 1)
    wins = b.push(np.zeros(SR * 10, dtype=np.float32))
    assert len(wins) == 8                       # 10s -> 8 个 3s 窗（步进 1s）
    assert all(w.duration == 3.0 for w in wins)
    assert wins[0].idx == 0 and wins[-1].idx == 7
    assert wins[1].t_rel == pytest.approx(1.0)


def test_step_overlap():
    b = WindowBuffer(SR, 3, 1)
    wins = b.push(np.zeros(SR * 5, dtype=np.float32))
    assert len(wins) == 3                        # t=0,1,2
    assert wins[1].t_rel == pytest.approx(1.0)
    assert wins[2].t_rel == pytest.approx(2.0)


def test_short_chunk_no_window():
    b = WindowBuffer(SR, 3, 1)
    assert b.push(np.zeros(SR, dtype=np.float32)) == []   # 1s < 3s 窗


def test_flush_emits_remainder():
    b = WindowBuffer(SR, 3, 1)
    b.push(np.zeros(int(SR * 7.5), dtype=np.float32))     # 5 整窗 + 2.5s 残余
    rem = b.flush()
    assert len(rem) == 1
    assert rem[0].duration == pytest.approx(2.5, abs=0.05)


def test_no_overlap_mode():
    b = WindowBuffer(SR, 3, 3)                  # 步进=窗长 → 不重叠
    wins = b.push(np.zeros(SR * 9, dtype=np.float32))
    assert len(wins) == 3
    assert wins[1].t_rel == pytest.approx(3.0)


def test_reset():
    b = WindowBuffer(SR, 3, 1)
    b.push(np.zeros(SR * 4, dtype=np.float32))
    b.reset()
    assert b.buf.size == 0
    assert b._idx == 0
