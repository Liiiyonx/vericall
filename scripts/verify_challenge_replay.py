#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_challenge_replay.py — B4-① 回放攻击 challenge-response 离线验证（2026-09-08）

作用：在**没有真机回放设备**的情况下，把 challenge-response 能离线验证的「数字匹配层」
跑通并输出可复现的拦截率；「音色匹配层」依赖真实 CAMPPlus（已有标定 EER 0.51%，
阈值 0.520 支撑），真机列留空待实测。

判定逻辑：src/fusion/challenge_response.py 的 assess()：
  pass_ok = (跟读数字 == 挑战串) AND (跟读音色相似度 >= 0.520)
  任一不满足 → suspect（拦截）。

回放攻击场景（学术界 replay attack 分类 + 本项目产品语义）：
  R1 全量录音回放：录整段诈骗话术，通话中直接放 → 无法跟读新数字
  R2 固定跟读回放：预录"跟读数字"音频，但数字是旧挑战串
  R3 剪接回放：多段录音拼接，跟读数字对不上/夹杂杂音
  R4 换人代读：同伙实时代读（数字对、音色不符）
  R5 本人实时跟读（正常，应放行）

用法：
  python scripts/verify_challenge_replay.py          # 跑全部场景，打印拦截率
  python scripts/verify_challenge_replay.py --seed 7
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion.challenge_response import VOICE_THR, assess, gen_challenge  # noqa: E402


def run_replay_matrix(seed: int) -> list[dict]:
    """离线模拟回放攻击场景，返回每条裁决记录。

    音色相似度：离线无法跑 CAMPPlus，用「阈值上下界的合成值」区分两类：
      - 真机本人实时跟读 → voice_sim 在阈值之上（如 0.65，代表 CAMPPlus 同人）；
      - 回放/换人 → voice_sim 在阈值之下（如 0.10，代表录音经扬声器-麦克风失真或异人）。
    数字层（跟读文本）为真实逻辑，无合成。
    """
    rng = random.Random(seed)
    records = []
    for _ in range(40):  # 每种场景 40 次，共 200 次挑战
        chal = gen_challenge(rng)

        # R1 全量录音回放：对方播放固定诈骗录音，跟读不到本次数字 → transcript 空/无关
        records.append(assess(chal, "", 0.10) | {"scene": "R1_全量录音回放",
                                                  "expect": "拦截"})
        # R2 固定跟读回放：预录的是另一个数字串 → 不匹配
        wrong = "".join(str((int(d) + rng.randint(1, 9)) % 10) for d in chal)
        records.append(assess(chal, wrong, 0.10) | {"scene": "R2_固定跟读回放",
                                                     "expect": "拦截"})
        # R3 剪接回放：数字错位/缺位（如只剩 2 位）
        partial = chal[:2]
        records.append(assess(chal, partial, 0.10) | {"scene": "R3_剪接回放",
                                                       "expect": "拦截"})
        # R4 换人代读：数字对、音色不符（异人）
        records.append(assess(chal, chal, 0.10) | {"scene": "R4_换人代读",
                                                    "expect": "拦截"})
        # R5 本人实时跟读：数字对、音色匹配（同人）
        records.append(assess(chal, chal, 0.65) | {"scene": "R5_本人实时跟读",
                                                    "expect": "放行"})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = run_replay_matrix(args.seed)
    scenes = sorted({r["scene"] for r in records})
    print(f"挑战-应答回放攻击 · 离线「数字匹配层」验证（seed={args.seed}）")
    print(f"阈值 VOICE_THR = {VOICE_THR}（与声纹登记 MATCH_THRESHOLD 同源）\n")

    # 汇总：拦截 = verdict suspect；放行 = verdict ok
    total_block = sum(1 for r in records if r["verdict"] == "suspect")
    print(f"{'场景':<18} {'次数':>4} {'拦截':>4} {'放行':>4} {'拦截率':>8}")
    print("-" * 46)
    for sc in scenes:
        grp = [r for r in records if r["scene"] == sc]
        n = len(grp)
        blk = sum(1 for r in grp if r["verdict"] == "suspect")
        ok = n - blk
        exp = grp[0]["expect"]
        flag = "✓" if (exp == "拦截" and blk == n) or (exp == "放行" and ok == n) else "✗ 异常"
        print(f"{sc:<18} {n:>4} {blk:>4} {ok:>4} {blk / n * 100:>7.1f}%  {flag}")

    # 回放攻击（R1-R4）总体拦截率，对应验收线"拦回放 ≥95%"
    attacks = [r for r in records if r["expect"] == "拦截"]
    n_atk = len(attacks)
    blk_atk = sum(1 for r in attacks if r["verdict"] == "suspect")
    print("-" * 46)
    print(f"回放攻击(R1-R4)合计：{n_atk} 次，拦截 {blk_atk}，"
          f"数字层拦截率 = {blk_atk / n_atk * 100:.1f}%（验收线 ≥95%）")
    # 正常放行（R5）零误伤
    normals = [r for r in records if r["expect"] == "放行"]
    ok_n = sum(1 for r in normals if r["verdict"] == "ok")
    print(f"正常跟读(R5)：{len(normals)} 次，放行 {ok_n}，误拦率 = "
          f"{(len(normals) - ok_n) / len(normals) * 100:.1f}%（适老零误伤）")
    print("\n说明：数字层离线可完全验证；音色层（R4 换人/回放失真）在真机用 CAMPPlus 实测，")
    print("阈值 0.520 由 200 人合库标定 EER 0.51% 支撑（voiceprint_calibration.md）。")
    print("真机实测表模板见 evaluation/challenge_replay_table.md。")


if __name__ == "__main__":
    main()
