# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 三通道融合决策编排器（通道①②③ 的汇总大脑）
================================================================

三个通道各管一件事，本模块负责把它们的结果融合成一个"放行/警惕/拦截"的最终裁决：

    通道① 声学伪造检测 (AASIST)   ->  acoustic:  这段语音是不是合成/克隆的
    通道② 家人声纹确认 (声纹比对) ->  voiceprint: 说话人是不是登记的家人本人
    通道③ 话术语义风险 (SenseVoice+Ollama) -> semantic: 内容是不是诈骗话术

融合策略（可解释优先，不黑箱）：
  1. 任一通道命中"高置信危险"即触发拦截（纵深防御，单点失效不影响安全）；
  2. 否则按权重汇总三通道的"可疑度"分数，超过阈值则警惕/拦截；
  3. 输出必须带每条通道的明细与最终裁决理由，便于适老提示与人工复核。

本模块与模型解耦：每个通道以 ChannelVerdict(score 0~1, label, detail) 的形式上报，
融合器只消费这个分数，不关心各通道内部实现。这样通道①②未训练/未接入时，
可用 safe_stub 占位，通道③（已验证）可真实参与。

用法：
    from fusion.fusion_orchestrator import FusionOrchestrator, ChannelVerdict
    orch = FusionOrchestrator()
    r = orch.decide(
        acoustic=ChannelVerdict("acoustic", 0.9, "spoof", "AASIST EER阈值命中"),
        voiceprint=ChannelVerdict("voiceprint", 0.1, "match", "声纹一致"),
        semantic=ChannelVerdict("semantic", 0.9, "impersonation", "冒充熟人要钱"),
    )
    print(r.final, r.rationale)   # block / 多通道高危...
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# 三通道权重（汇总分时；命中拦截的硬规则权重更高，见 decide）
DEFAULT_WEIGHTS = {
    "acoustic": 0.35,    # 合成语音——最客观、最该信
    "voiceprint": 0.30,  # 非本人——强信号（但家人声纹可能因感冒/环境变化波动）
    "semantic": 0.35,    # 诈骗话术——内容层，r1 可能误判，权重与声学持平
}

# 单通道"高置信危险"硬阈值：命中即直接拦截（纵深防御）
HARD_BLOCK_SCORE = 0.85
# 汇总可疑度软阈值：超过则警惕，再高则拦截
SOFT_ALERT = 0.50
SOFT_BLOCK = 0.70
# 出处：scripts/fusion_calibration.py（evaluation/fusion_calibration.md，P1-5）。
# 结论：ASVspoof 英文基准上声学分数双峰，网格最低(0.70/0.35/0.55)与生产默认
# (0.85/0.50/0.70) 决策完全相同(FA=1/MISS=3)。为保留"误拦家人更伤产品"的安全边，
# 生产维持偏高 hard_block；soft 阈值三通道差异需在中文/信道退化集上再验证。


@dataclass
class ChannelVerdict:
    name: str            # acoustic / voiceprint / semantic
    score: float         # 0~1，越高越可疑
    label: str          # 各通道语义标签（spoof/match/impersonation/normal...）
    detail: str          # 给人看的明细（日志/适老提示）
    confidence: float = 1.0   # 该通道结果的置信度（低置信时融合权重下调）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusionResult:
    final: str           # allow / caution / block
    score: float         # 汇总可疑度 0~1
    confidence: float    # 融合后整体置信度
    rationale: str       # 裁决理由（中文，面向提示/复核）
    channels: list = field(default_factory=list)  # [ChannelVerdict.to_dict()]
    offline: bool = False  # 是否离线降级模式产出（演示兜底）

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class FusionOrchestrator:
    """三通道融合决策。规则透明、可解释、纵深防御。"""

    def __init__(self, weights: Optional[dict] = None,
                 hard_block: float = HARD_BLOCK_SCORE,
                 soft_alert: float = SOFT_ALERT,
                 soft_block: float = SOFT_BLOCK):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.hard_block = hard_block
        self.soft_alert = soft_alert
        self.soft_block = soft_block

    # ------------------------------------------------------------------ #
    def decide(self, acoustic: ChannelVerdict,
               voiceprint: ChannelVerdict,
               semantic: ChannelVerdict) -> FusionResult:
        chans = [acoustic, voiceprint, semantic]

        # 1) 硬规则：任一通道高置信危险 -> 直接拦截
        for c in chans:
            if c.score >= self.hard_block and c.confidence >= 0.6:
                return FusionResult(
                    final="block",
                    score=round(max(c.score for c in chans), 2),
                    confidence=round(c.confidence, 2),
                    rationale=f"纵深防御拦截：{c.name} 通道高置信命中"
                              f"（{c.label}，{c.detail}）",
                    channels=[c.to_dict() for c in chans],
                )

        # 2) 加权汇总（按置信度调整权重：低置信通道贡献打折）
        wsum = 0.0
        weighted = 0.0
        for c in chans:
            w = self.weights.get(c.name, 0.33) * c.confidence
            weighted += c.score * w
            wsum += w
        fused = weighted / wsum if wsum > 0 else 0.0

        # 2.5) 单通道中危升级：任一通道 score>=0.60 且 conf>=0.6 → 至少 caution
        #      （避免单通道已示警却被其他通道稀释成"放行"，放过中危威胁）
        single_escalate = any(c.score >= 0.60 and c.confidence >= 0.6
                              for c in chans)

        # 3) 软阈值裁决
        if fused >= self.soft_block:
            final = "block"
        elif fused >= self.soft_alert or single_escalate:
            final = "caution"
        else:
            final = "allow"

        # 整体置信度：三通道均值；任一通道低置信（<0.4）时再打 8 折——
        # 残废通道会拉低整体结论的可信度，提示人工复核而非静默采信。
        avg_conf = sum(c.confidence for c in chans) / len(chans)
        if any(c.confidence < 0.4 for c in chans):
            avg_conf *= 0.8

        rationale = self._explain(final, fused, chans)
        return FusionResult(
            final=final,
            score=round(fused, 2),
            confidence=round(avg_conf, 2),
            rationale=rationale,
            channels=[c.to_dict() for c in chans],
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _explain(final: str, fused: float, chans: list) -> str:
        parts = []
        for c in chans:
            tag = "可疑" if c.score >= 0.5 else "正常"
            parts.append(f"{c.name}={c.score:.2f}({tag},{c.label})")
        verdict = {"allow": "放行", "caution": "警惕", "block": "拦截"}[final]
        return f"最终【{verdict}】汇总可疑度={fused:.2f}；" + "；".join(parts)

    # ------------------------------------------------------------------ #
    @staticmethod
    def safe_stub(name: str, note: str = "通道未接入") -> ChannelVerdict:
        """未训练/未接入通道的占位：给中性分 0.0 + 低置信，避免误拦截。"""
        return ChannelVerdict(name=name, score=0.0, label="stub",
                              detail=note, confidence=0.0)


# ---------------------------------------------------------------------- #
def _demo():
    from fusion.semantic_channel import SemanticChannel
    orch = FusionOrchestrator()

    print("=== 演示 1：诈骗话术（通道③真实，①②占位）===")
    # 通道③ 真实跑（需要 Ollama + 可选 SenseVoice）
    ch3 = SemanticChannel().analyze("妈我是同学号我手机摔了急用五万块别告诉爸")
    r = orch.decide(
        acoustic=FusionOrchestrator.safe_stub("acoustic", "AASIST 训练中"),
        voiceprint=FusionOrchestrator.safe_stub("voiceprint", "声纹待采集"),
        semantic=ChannelVerdict("semantic", ch3.risk, ch3.category, ch3.reason,
                               confidence=0.9 if ch3.category != "unknown" else 0.3),
    )
    print(r.rationale)
    print("→", r.final, "| score", r.score, "| conf", r.confidence)

    print("\n=== 演示 2：三通道全命中（假想，合成语音+非本人+诈骗话术）===")
    r2 = orch.decide(
        acoustic=ChannelVerdict("acoustic", 0.92, "spoof", "AASIST 判定合成语音", 0.95),
        voiceprint=ChannelVerdict("voiceprint", 0.85, "mismatch", "声纹不符", 0.9),
        semantic=ChannelVerdict("semantic", 0.9, "impersonation", "冒充熟人要钱", 0.9),
    )
    print(r2.rationale)
    print("→", r2.final)

    print("\n=== 演示 3：正常通话（三通道全正常）===")
    r3 = orch.decide(
        acoustic=ChannelVerdict("acoustic", 0.05, "bonafide", "真实人声", 0.9),
        voiceprint=ChannelVerdict("voiceprint", 0.1, "match", "声纹一致", 0.9),
        semantic=ChannelVerdict("semantic", 0.1, "normal", "正常闲聊", 0.9),
    )
    print(r3.rationale)
    print("→", r3.final)


if __name__ == "__main__":
    _demo()
