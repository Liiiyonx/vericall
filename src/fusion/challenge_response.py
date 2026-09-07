# -*- coding: utf-8 -*-
"""challenge_response.py — B4-① 回放攻击防御：challenge-response（2026-09-07）

思路：对"声学存疑 / 声纹低置信 / 大额转账前"的来电，系统播报随机 4 位数字，
要求对方跟读。双校验：
  1) 跟读内容 == 挑战串（防预先录好的固定话术/全量录音直接放）；
  2) 跟读音色的声纹相似度 ≥ 阈值（防第二人代读 / 换人）。
纯逻辑层：识别文本与相似度由上层（ASR + CAMPPlus verify）提供，本模块只做判定与
可解释输出；可单测。实测拦截率表（回放设备 ≥95%）待真机链路（见 b4_antiattack_design.md）。
"""
from __future__ import annotations

import random

# 声纹相似度阈值：与 voiceprint_channel.MATCH_THRESHOLD 同源（voiceprint_calibration.md）
VOICE_THR = 0.520


def gen_challenge(rng: random.Random | None = None) -> str:
    """生成 4 位随机挑战串（避免 0000 等易猜全同）。"""
    r = rng or random.SystemRandom()
    return "".join(str(r.randrange(10)) for _ in range(4))


def assess(challenge: str, transcript: str,
           voice_sim: float, voice_thr: float = VOICE_THR) -> dict:
    """判定一次 challenge-response。

    transcript: ASR 跟读文本（数字串）；voice_sim: 跟读段与登记家人的声纹余弦。
    返回 {pass_ok, reasons, verdict}：
      - digits 必须完全匹配（含语音数字词→数字归一由上层做）；
      - voice_sim ≥ voice_thr 才视为"本人实时跟读"（防换人/录音拼接）。
    """
    digits = "".join(ch for ch in str(transcript or "") if ch.isdigit())
    reasons = []
    if digits == challenge:
        reasons.append("跟读数字匹配")
    else:
        reasons.append(f"跟读不匹配（期望 {challenge}，识得 {digits or '无'}）")
    if voice_sim >= voice_thr:
        reasons.append(f"音色匹配（sim={voice_sim:.3f}≥{voice_thr}）")
    else:
        reasons.append(f"音色不符（sim={voice_sim:.3f}<{voice_thr}，疑似换人/录音）")
    pass_ok = digits == challenge and voice_sim >= voice_thr
    return {"pass_ok": pass_ok, "reasons": reasons,
            "verdict": "ok" if pass_ok else "suspect"}


if __name__ == "__main__":
    c = gen_challenge()
    print("challenge:", c)
    for tr, sim in [(c, 0.61), (c, 0.42), ("1234", 0.61), ("", 0.1)]:
        print(tr, sim, assess(c, tr, sim))
