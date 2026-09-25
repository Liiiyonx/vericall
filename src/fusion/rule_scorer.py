# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 离线规则评分器（通道③ LLM 兜底 / 规则基线）
================================================================

两条用途（任务书 P0-4 指定）：
  1. 离线降级：Ollama 不可用 或 VERICALL_OFFLINE=1 时，用语义关键词命中估算
     诈骗话术风险，保证演示链路不中断。
  2. 规则基线：论文/答辩里 "规则基线 vs LLM" 的消融对比实验实现
     （P1-5 阈值标定 / P2-5 红队对抗 复用本模块）。

风险公式（任务书 P0-4 原文，封顶 1.0）：
    risk = 0.15 * 命中类别数 + 0.25 * 最高类别命中词数
    命中 0 类 -> normal / 0.05

关键词表对齐公安部反诈宣传口径，覆盖四类高发电话诈骗话术。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


# 四类高发电话诈骗话术关键词（任务书 P0-4 指定表，可在此按需扩展）
RULES = {
    "money_request":  ["转账", "汇款", "打钱", "安全账户", "垫付", "五万", "万块", "卡号", "二维码"],
    "impersonation":  ["是我啊", "同学的号", "手机摔", "新号码", "我是你", "儿子", "女儿", "领导"],
    "urgency_threat": ["公安", "涉嫌", "冻结", "逮捕", "出事", "来不及", "马上"],
    "credential":     ["验证码", "密码", "卡号", "人脸", "刷脸"],
}


@dataclass
class RuleResult:
    risk: float            # 0~1 诈骗风险分
    category: str          # RULES 键 / normal / unknown
    reason: str            # 中文理由
    hits: dict             # {类别: 命中词数}

    def to_dict(self) -> dict:
        return asdict(self)


def rule_score(text: str) -> RuleResult:
    """对一段转写文本跑规则评分，返回 RuleResult。零依赖、可解释。"""
    text = (text or "").strip()
    if not text:
        return RuleResult(0.05, "normal", "规则基线: 空文本", {})

    hits: dict[str, int] = {}
    for cat, kws in RULES.items():
        n = sum(1 for kw in kws if kw in text)
        if n:
            hits[cat] = n

    if not hits:
        return RuleResult(0.05, "normal", "规则基线: 未命中诈骗特征", {})

    n_cat = len(hits)
    max_hits = max(hits.values())
    risk = min(1.0, 0.15 * n_cat + 0.25 * max_hits)
    cat = max(hits, key=hits.get)
    return RuleResult(
        risk=round(risk, 2),
        category=cat,
        reason=f"规则基线命中{n_cat}类{max_hits}词({cat})",
        hits=hits,
    )


# ---------------------------------------------------------------------- #
def _demo():
    samples = [
        ("诈骗-冒充+要钱", "喂妈，是我啊，我手机摔坏了这是同学的号。我在学校出点事急用钱，"
                          "你先转五万到这个卡号，别告诉我爸，快点啊要不来不及了。"),
        ("诈骗-公检法", "这里是市公安局，你名下的银行卡涉嫌洗钱案件，需要把资金转入"
                       "安全账户配合清查，否则今天下午就会冻结你全部账户并逮捕你。"),
        ("正常-家人", "妈，我周末回来吃饭，你别买菜了我在路上买。对了爸的降压药吃完没？"),
        ("正常-业务", "您好，您的快递已放到小区丰巢柜，取件码是 3372，请及时领取。"),
    ]
    for name, text in samples:
        r = rule_score(text)
        flag = "⚠️" if r.risk >= 0.7 else "✓"
        print(f"[{flag}] {name}: risk={r.risk} cat={r.category} {r.reason}")


if __name__ == "__main__":
    _demo()
