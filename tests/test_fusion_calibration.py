# -*- coding: utf-8 -*-
"""P1-5 融合标定链路测试：build_scenarios 协议发现 + fusion_calibration 决策/代价。

不依赖 GPU/Ollama/真实数据：纯逻辑 + 合成分数。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fusion_calibration import decide, cost, _weights_grid, search_best  # noqa: E402
from fusion.fusion_orchestrator import (  # noqa: E402
    DEFAULT_WEIGHTS, HARD_BLOCK_SCORE, SOFT_ALERT, SOFT_BLOCK)


def _verdicts(ac, vp=0.1, sem=0.1, ca=0.9, cv=0.9, cs=0.9):
    return {"ac": ac, "vp": vp, "sem": sem, "ca": ca, "cv": cv, "cs": cs}


def test_weights_grid_sums_to_one():
    for w in _weights_grid():
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert w["semantic"] >= 0.1


def test_decide_hard_block():
    # 任一通道高置信高分 -> 直接 block
    d = decide(ac=0.92, vp=0.1, sem=0.1, ca=0.95, cv=0.9, cs=0.9,
               hard=0.85, alert=0.5, block=0.7, w=DEFAULT_WEIGHTS)
    assert d == "block"


def test_decide_all_normal_allow():
    d = decide(ac=0.05, vp=0.1, sem=0.05, ca=0.9, cv=0.9, cs=0.9,
               hard=0.85, alert=0.5, block=0.7, w=DEFAULT_WEIGHTS)
    assert d == "allow"


def test_decide_high_but_low_conf_not_hardblock():
    # 高分数但低置信（未成熟模型）不应触发硬拦截
    d = decide(ac=0.92, vp=0.1, sem=0.1, ca=0.3, cv=0.9, cs=0.9,
               hard=0.85, alert=0.5, block=0.7, w=DEFAULT_WEIGHTS)
    assert d != "block"  # conf 0.3 < 0.6 -> 不走硬拦，走软阈值


def test_cost_matrix():
    assert cost("block", "allow", 3.0) == 3.0      # 误拦家人
    assert cost("allow", "allow", 3.0) == 0.0
    assert cost("allow", "block", 3.0) == 1.0      # 漏放克隆
    assert cost("block", "block", 3.0) == 0.0
    assert cost("allow", "block|caution", 3.0) == 1.0  # 陌生人被 allow = 漏
    assert cost("caution", "block|caution", 3.0) == 0.0


def test_search_best_runs_and_returns_key():
    rows = [
        {**_verdicts(0.99, 0.9, 0.05), "id": "a", "expected": "block"},
        {**_verdicts(0.05, 0.1, 0.05), "id": "b", "expected": "allow"},
        {**_verdicts(0.99, 0.9, 0.05), "id": "c", "expected": "block"},
    ]
    best = search_best(rows, fa_cost=3.0)
    assert best is not None
    for k in ("hard", "alert", "block", "w", "cost"):
        assert k in best
    assert best["cost"] <= 6.0  # 全判对则 cost 应很低


def test_production_defaults_match_orchestrator():
    # 确保 calibration 用到的默认常数与 orchestrator 一致（避免两处漂移）
    assert HARD_BLOCK_SCORE == 0.85
    assert SOFT_ALERT == 0.50
    assert SOFT_BLOCK == 0.70
    assert set(DEFAULT_WEIGHTS) == {"acoustic", "voiceprint", "semantic"}


def test_build_scenarios_finds_real_protocol(tmp_path, monkeypatch):
    """回归：build_scenarios 的协议发现应命中真实文件命名
    ASVspoof2019.LA.cm.dev.trl.txt（旧 glob 'dev_protocol' 会漏配 → 骨架清单）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_scenarios", ROOT / "scripts" / "build_scenarios.py")
    bs = importlib.util.module_from_spec(spec)
    sys.modules["build_scenarios"] = bs
    spec.loader.exec_module(bs)

    # 模拟真实数据根：ASVspoof2019_LA/LA/ASVspoof2019_LA_cm_protocols/*.dev.trl.txt
    la = tmp_path / "ASVspoof2019_LA" / "LA"
    prots = la / "ASVspoof2019_LA_cm_protocols"
    prots.mkdir(parents=True)
    proto = prots / "ASVspoof2019.LA.cm.dev.trl.txt"
    proto.write_text("LA_0069 LA_D_1047731 - - bonafide\n"
                     "LA_0069 LA_D_2959188 - - spoof\n", encoding="utf-8")

    monkeypatch.setattr(bs, "ASVSPOOF_LA", la)
    found = bs._find_protocol()
    assert found is not None
    assert found.name == "ASVspoof2019.LA.cm.dev.trl.txt"

    # 解析协议应得到正确标签
    labels = bs._parse_protocol(found)
    assert labels["LA_D_1047731"] == "bonafide"
    assert labels["LA_D_2959188"] == "spoof"
