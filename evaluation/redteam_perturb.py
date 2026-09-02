# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 红队对抗扰动（通道③ 鲁棒性压力测试）
================================================================
对诈骗话术文本施加**字符级**扰动，模拟攻击者绕过基于子串的关键词匹配，
检验 rule_scorer 的鲁棒性。扰动对人而言几乎不影响理解，却能让匹配失效——
这正是红队要找的盲区。

策略：
  - space      : 字符间插入空格，破坏多字关键词子串匹配（"转账" -> "转 账"）。
  - homophone  : 形近/同音/错别字替换（"账"->"帐"、"安"->"按"、"转"->"专"）。
两者都是确定性、可复现的（给定 seed 结果固定）。

注意：这是"红队找盲区"的演示，不是攻击实现。目的是量化现有规则基线的
脆弱性，推动后续引入文本归一化 + 语义向量检测等加固手段。
"""
from __future__ import annotations

from typing import Iterable, Sequence

# 形近/同音/错别 替换表：刻意选能破坏关键词匹配、但人仍能读懂的字。
_CONFUSABLE = {
    "转": "专", "账": "帐", "钱": "前", "安": "按", "全": "泉",
    "验": "严", "证": "正", "码": "马", "涉": "社", "嫌": "咸",
    "冻": "动", "捕": "补", "领": "令", "导": "到", "公": "工",
    "户": "护", "口": "扣", "密": "蜜", "人": "仁", "脸": "睑",
    "刷": "唰", "付": "副", "垫": "埝", "汇": "汇", "款": "欵",
    "转": "专",
}

VALID_STRATEGIES = ("space", "homophone")


def _insert_spaces(text: str) -> str:
    """每对相邻字符间插入空格（确定性）。"""
    return " ".join(text)


def _substitute(text: str, table: dict[str, str]) -> str:
    """按替换表逐字替换（确定性）。"""
    return "".join(table.get(ch, ch) for ch in text)


def perturb(text: str, strategy: str = "space", seed: int = 0) -> str:
    """对单条文本施加一种对抗扰动，返回扰动后文本。

    seed 参数保留以便将来扩展随机化策略，当前两种策略均确定性。
    """
    if strategy == "space":
        return _insert_spaces(text)
    if strategy == "homophone":
        return _substitute(text, _CONFUSABLE)
    if strategy == "none":
        return text
    raise ValueError(f"未知扰动策略: {strategy}（可选 {VALID_STRATEGIES}）")


def perturb_all(text: str,
                strategies: Iterable[str] | None = None,
                seed: int = 0) -> dict[str, str]:
    """对单条文本施加多种扰动，返回 {策略名: 扰动后文本}。"""
    strategies = list(strategies or VALID_STRATEGIES)
    return {s: perturb(text, s, seed) for s in strategies}


def perturb_samples(texts: Sequence[str],
                    strategies: Iterable[str] | None = None,
                    seed: int = 0) -> list[dict]:
    """对一组文本批量扰动，返回 [{orig, dialect?, ...perturbed}]。

    每条含 'orig' 原文与每个策略的扰动结果，便于评测统计逃避率。
    """
    strategies = list(strategies or VALID_STRATEGIES)
    out = []
    for t in texts:
        row: dict = {"orig": t}
        row.update(perturb_all(t, strategies, seed))
        out.append(row)
    return out


if __name__ == "__main__":
    demo = "妈我是同学号我手机摔了急用五万块你转卡号别告诉我爸"
    print(f"原文   : {demo}")
    for s in VALID_STRATEGIES:
        print(f"{s:10s}: {perturb(demo, s)}")
