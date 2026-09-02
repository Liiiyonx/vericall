# -*- coding: utf-8 -*-
"""evaluation.metrics 指标单测（需 numpy）。

核心不变量：
    - 完全可分离的数据 EER 应为 0
    - 完全重叠的数据 EER 应 ≈ 50%
    - 输入打乱后结果必须一致（曾因漏排序导致 searchsorted 算错）
    - far/frr 在阈值处计算正确
"""
import numpy as np

from metrics import compute_eer, far_frr_at, summarize


def test_eer_separable_is_zero():
    bon = [0.1, 0.2, 0.3, 0.4]
    spf = [0.6, 0.7, 0.8, 0.9]
    eer, _ = compute_eer(bon, spf)
    assert eer == 0.0


def test_eer_overlap_around_fifty():
    bon = [0.1, 0.2, 0.3, 0.4]
    spf = [0.15, 0.25, 0.35, 0.45]
    eer, _ = compute_eer(bon, spf)
    assert 40.0 <= eer <= 60.0


def test_shuffle_invariance():
    rng = np.random.default_rng(0)
    bon = rng.uniform(0.0, 0.4, 200)
    spf = rng.uniform(0.6, 1.0, 200)
    eer1, thr1 = compute_eer(bon, spf)
    rng.shuffle(bon)
    rng.shuffle(spf)
    eer2, thr2 = compute_eer(bon, spf)
    assert abs(eer1 - eer2) < 1e-9
    assert abs(thr1 - thr2) < 1e-9


def test_far_frr_at_threshold():
    bon = [0.1, 0.2, 0.3, 0.4]
    spf = [0.6, 0.7, 0.8, 0.9]
    far, frr = far_frr_at(bon, spf, 0.5)
    assert far == 0.0 and frr == 0.0


def test_summarize_keys_and_counts():
    bon = [0.1, 0.2, 0.3, 0.4]
    spf = [0.6, 0.7, 0.8, 0.9]
    d = summarize(bon, spf)
    for k in ("n_bonafide", "n_spoof", "eer_pct", "eer_threshold",
              "far_at_eer_pct", "frr_at_eer_pct"):
        assert k in d
    assert d["n_bonafide"] == 4 and d["n_spoof"] == 4
