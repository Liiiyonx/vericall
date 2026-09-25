# -*- coding: utf-8 -*-
"""离线降级管线单测（任务书 P0-4 验收）。

必须在导入 paths 之前设 VERICALL_OFFLINE=1（paths 在模块加载时读取该变量）。
不依赖 torch / funasr / Ollama，可在纯 CI 或评委笔记本运行。
"""
import os

os.environ["VERICALL_OFFLINE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fusion.pipeline import VeriCallPipeline  # noqa: E402


def test_offline_demo_scenarios():
    pipe = VeriCallPipeline()
    out = {s: pipe.analyze("ignored.wav", scenario=s) for s in ("A", "B", "C")}

    # 任务书 P0-4 验收：A=放行 / B=拦截 / C=警惕|拦截
    assert out["A"].final == "allow", out["A"].rationale
    assert out["B"].final == "block", out["B"].rationale
    assert out["C"].final in ("caution", "block"), out["C"].rationale

    # 必须是离线产物（缓存通道 + 实时规则话术）
    for r in out.values():
        assert r.offline is True


def test_offline_semantic_runs_rule_scorer():
    pipe = VeriCallPipeline()
    r = pipe.analyze("x.wav", scenario="B")
    # 场景 B 话术命中规则器，语义通道给出高风险的诈骗类别
    sem = next(c for c in r.channels if c["name"] == "semantic")
    assert sem["score"] >= 0.7
    assert sem["label"] in ("impersonation", "money_request", "urgency_threat", "credential")
