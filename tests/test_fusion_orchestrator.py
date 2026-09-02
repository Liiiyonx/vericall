# -*- coding: utf-8 -*-
"""FusionOrchestrator 决策逻辑单测（纯 stdlib，无需模型 / GPU）。

覆盖四条核心规则：
    1. 纵深防御：单通道高置信危险 -> 拦截
    2. 高置信门控：低置信高分不误拦
    3. 加权汇总软阈值：allow / caution / block
    4. 单通道中危升级：不被其他通道稀释成放行
"""
from fusion.fusion_orchestrator import FusionOrchestrator, ChannelVerdict


def cv(name, score, conf=1.0, label="x", detail=""):
    return ChannelVerdict(name=name, score=score, label=label,
                          detail=detail, confidence=conf)


def test_hard_block_on_high_confidence():
    o = FusionOrchestrator()
    r = o.decide(acoustic=cv("acoustic", 0.92, 0.95, "spoof", "AASIST 命中"),
                 voiceprint=cv("voiceprint", 0.1),
                 semantic=cv("semantic", 0.1))
    assert r.final == "block"
    assert "纵深防御" in r.rationale


def test_hard_block_requires_confidence():
    # 高分但低置信：不应误拦截（防止模型抖动误杀老人正常通话）
    o = FusionOrchestrator()
    r = o.decide(acoustic=cv("acoustic", 0.95, conf=0.3),
                 voiceprint=cv("voiceprint", 0.05),
                 semantic=cv("semantic", 0.05))
    assert r.final != "block"


def test_all_normal_pass():
    o = FusionOrchestrator()
    r = o.decide(acoustic=cv("acoustic", 0.05),
                 voiceprint=cv("voiceprint", 0.1),
                 semantic=cv("semantic", 0.1))
    assert r.final == "allow"


def test_weighted_block():
    o = FusionOrchestrator()
    r = o.decide(acoustic=cv("acoustic", 0.8),
                 voiceprint=cv("voiceprint", 0.8),
                 semantic=cv("semantic", 0.8))
    assert r.final == "block"
    assert r.score >= 0.70


def test_single_channel_escalation_to_caution():
    # 单通道中危(score>=0.6, conf>=0.6) 应至少警惕，不被稀释放行
    o = FusionOrchestrator()
    r = o.decide(acoustic=cv("acoustic", 0.6, 0.9),
                 voiceprint=cv("voiceprint", 0.05),
                 semantic=cv("semantic", 0.05))
    assert r.final == "caution"


def test_safe_stub_all_neutral_pass():
    o = FusionOrchestrator()
    r = o.decide(acoustic=FusionOrchestrator.safe_stub("acoustic"),
                 voiceprint=FusionOrchestrator.safe_stub("voiceprint"),
                 semantic=FusionOrchestrator.safe_stub("semantic"))
    assert r.final == "allow"


def test_weights_override():
    # 只信声学通道时，单通道高可疑应被放大到拦截
    o = FusionOrchestrator(weights={"acoustic": 1.0, "voiceprint": 0.0, "semantic": 0.0})
    r = o.decide(acoustic=cv("acoustic", 0.9),
                 voiceprint=cv("voiceprint", 0.0),
                 semantic=cv("semantic", 0.0))
    assert r.score > 0.70


def test_low_confidence_lowers_fused_score():
    # 声学通道高可疑、其它通道正常时：声学置信度高 -> 汇总分更高；
    # 声学置信度低 -> 其贡献被打折，汇总分更低。
    # （注意：三通道分数全相同时置信度缩放会相互抵消，故这里制造分数差。）
    hi = FusionOrchestrator().decide(
        acoustic=cv("acoustic", 0.9, 0.9),
        voiceprint=cv("voiceprint", 0.1, 0.9),
        semantic=cv("semantic", 0.1, 0.9)).score
    lo = FusionOrchestrator().decide(
        acoustic=cv("acoustic", 0.9, 0.1),
        voiceprint=cv("voiceprint", 0.1, 0.9),
        semantic=cv("semantic", 0.1, 0.9)).score
    assert lo < hi
