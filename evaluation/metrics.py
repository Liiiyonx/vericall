# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 评测指标（零依赖，仅 numpy）
================================================================
把评测指标从 external/aasist 里剥离出来，这样即使 AASIST 没 clone，
本目录的统计/报告工具也能独立运行。

分数口径（全项目统一）：
    score = 伪造概率，0~1，越高越像 AIGC 合成/克隆语音。
    判定规则：score >= threshold 判为伪造(spoof)。

指标：
    EER     等错误率，FAR 与 FRR 相等时的错误率。越低越好。
    FAR     把真人误判成伪造的比例（误拦，伤害用户体验）。
    FRR     把伪造漏判成真人的比例（漏拦，直接造成诈骗损失）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_f64(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"分数必须是一维数组，收到 shape={a.shape}")
    return a


def compute_eer(bonafide_scores: Sequence[float],
                spoof_scores: Sequence[float]) -> tuple[float, float]:
    """在「伪造概率」口径下计算 EER 与对应阈值。

    参数:
        bonafide_scores: 真实人声样本的模型输出（伪造概率）
        spoof_scores:    伪造样本的模型输出（伪造概率）

    返回:
        (eer, threshold)，eer 为百分比形式（如 0.745 表示 0.745%）

    原理:
        取全体分数为候选阈值，逐点计算
            FAR(thr) = #{bonafide >= thr} / n_bonafide   误拦率
            FRR(thr) = #{spoof < thr}    / n_spoof       漏拦率
        EER 取 |FAR - FRR| 最小处，并返回二者均值。
    """
    bon = _as_f64(bonafide_scores)
    spf = _as_f64(spoof_scores)
    if bon.size == 0 or spf.size == 0:
        raise ValueError("bonafide 与 spoof 样本都不能为空")

    # 必须排序：下面用 searchsorted 做二分计数，输入无序会得到错误结果
    bon = np.sort(bon)
    spf = np.sort(spf)

    thresholds = np.unique(np.concatenate([bon, spf]))
    n_bon, n_spf = bon.size, spf.size

    # bon >= thr 的数量：n_bon - (bon < thr 的数量)
    far = (n_bon - np.searchsorted(bon, thresholds, side="left")) / n_bon
    # spf < thr 的数量
    frr = np.searchsorted(spf, thresholds, side="left") / n_spf

    diff = np.abs(far - frr)
    i = int(np.argmin(diff))
    eer = float((far[i] + frr[i]) / 2.0) * 100.0
    return eer, float(thresholds[i])


def far_frr_at(bonafide_scores: Sequence[float],
               spoof_scores: Sequence[float],
               threshold: float) -> tuple[float, float]:
    """给定阈值，返回 (FAR, FRR)，均为百分比。"""
    bon = _as_f64(bonafide_scores)
    spf = _as_f64(spoof_scores)
    if bon.size == 0 or spf.size == 0:
        raise ValueError("bonafide 与 spoof 样本都不能为空")
    far = float(np.count_nonzero(bon >= threshold) / bon.size) * 100.0
    frr = float(np.count_nonzero(spf < threshold) / spf.size) * 100.0
    return far, frr


def summarize(bonafide_scores: Sequence[float],
              spoof_scores: Sequence[float]) -> dict:
    """一次算全常用指标，供报告直接消费。"""
    bon = _as_f64(bonafide_scores)
    spf = _as_f64(spoof_scores)
    eer, thr = compute_eer(bon, spf)
    far, frr = far_frr_at(bon, spf, thr)
    return {
        "n_bonafide": int(bon.size),
        "n_spoof": int(spf.size),
        "eer_pct": round(eer, 4),
        "eer_threshold": round(thr, 6),
        "far_at_eer_pct": round(far, 4),
        "frr_at_eer_pct": round(frr, 4),
        "bonafide_mean": round(float(bon.mean()), 5),
        "bonafide_p95": round(float(np.percentile(bon, 95)), 5),
        "spoof_mean": round(float(spf.mean()), 5),
        "spoof_p5": round(float(np.percentile(spf, 5)), 5),
    }


def _selfcheck() -> None:
    """内置自检：可完全分离时应得 EER≈0，完全重叠时应得 EER≈50。"""
    bon = np.array([0.1, 0.2, 0.3, 0.4])
    spf = np.array([0.6, 0.7, 0.8, 0.9])
    eer, thr = compute_eer(bon, spf)
    assert eer == 0.0, f"可分离数据 EER 应为 0，实际 {eer}"

    bon2 = np.array([0.1, 0.2, 0.3, 0.4])
    spf2 = np.array([0.1, 0.2, 0.3, 0.4])
    eer2, _ = compute_eer(bon2, spf2)
    assert 40.0 <= eer2 <= 60.0, f"完全重叠 EER 应≈50，实际 {eer2}"

    d = summarize(bon, spf)
    assert d["n_bonafide"] == 4 and d["n_spoof"] == 4

    # 打乱顺序后结果必须一致（曾因漏排序导致 searchsorted 算错）
    rng = np.random.default_rng(0)
    bon_s, spf_s = bon.copy(), spf.copy()
    rng.shuffle(bon_s)
    rng.shuffle(spf_s)
    eer3, thr3 = compute_eer(bon_s, spf_s)
    assert abs(eer3 - eer) < 1e-9 and abs(thr3 - thr) < 1e-9, \
        f"打乱后结果应一致：{eer3}/{thr3} != {eer}/{thr}"

    print(f"metrics 自检通过：可分离 EER={eer:.2f}% (thr={thr})，重叠 EER={eer2:.2f}%")


if __name__ == "__main__":
    _selfcheck()
