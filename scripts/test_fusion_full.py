#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 三通道全真实融合实战测试。

为什么需要这个脚本:
  fusion.pipeline 的 demo 只有"同人/异人"两个场景, 缺了本项目最核心的威胁模型:
  【AIGC 冒充家人】—— 攻击者用家人的音色合成一段话, 此时通道②声纹会判"匹配"
  (音色确实一样), 只有通道①声学伪造检测能识破。这正是三通道融合存在的意义。

测试素材(全部来自 ASVspoof2019 LA dev 集, 真实攻击样本):
  选取同时拥有 bonafide 与 spoof 语句的说话人(如 LA_0069: 154 真声 / 2484 伪造声),
  构造三个场景:
    场景A 家人本人来电      : 该说话人的真声(非登记用的那条)     -> 期望 allow
    场景B AIGC 冒充家人 ⭐  : 同一说话人音色的伪造声              -> 期望 block
    场景C 陌生人来电        : 另一说话人的真声                    -> 期望 block/caution

  场景B 是分水岭: 若只有声纹, 它会放行(音色匹配); 三通道融合后必须被通道①拦下。

用法:
  python scripts/test_fusion_full.py                    # 全跑(需 GPU 空闲)
  python scripts/test_fusion_full.py --device cpu       # 声学通道走 CPU
  python scripts/test_fusion_full.py --skip-semantic    # 跳过 ASR+LLM, 只测①②
  python scripts/test_fusion_full.py --speaker LA_0069 --spoof-key LA_D_xxx
"""
import os
import sys
import argparse
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(ROOT, "..", "src"))
sys.path.insert(0, SRC)
from paths import DEV_FLAC, DEVICE  # noqa: E402

DB = str(DEV_FLAC.parent)
DEV_TRL = os.path.join(DB, "ASVspoof2019_LA_cm_protocols",
                       "ASVspoof2019.LA.cm.dev.trl.txt")


def load_pairs():
    """返回 {speaker: {'bonafide': [...], 'spoof': [...]}}"""
    groups = collections.defaultdict(lambda: {"bonafide": [], "spoof": []})
    with open(DEV_TRL, "r") as f:
        for line in f:
            spk, key, _, _, label = line.strip().split(" ")
            groups[spk][label].append(key)
    return groups


def pick_cast(groups, speaker=None):
    """挑一个同时有真声/伪造声的说话人当'家人', 另一个当'陌生人'"""
    both = sorted(s for s, g in groups.items()
                  if g["bonafide"] and g["spoof"])
    if not both:
        raise RuntimeError("dev 集中找不到同时含真声与伪造声的说话人")
    fam = speaker or both[0]
    if fam not in both:
        raise ValueError(f"{fam} 不是可用说话人, 可选: {both}")
    stranger = next(s for s in both if s != fam) if len(both) > 1 else None
    return fam, stranger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default=None, help="扮演'家人'的说话人 ID")
    ap.add_argument("--spoof-key", default=None, help="指定伪造语句(默认取第一条)")
    ap.add_argument("--device", default=DEVICE, choices=["cuda", "cpu"])
    ap.add_argument("--skip-semantic", action="store_true",
                    help="跳过通道③(ASR+Ollama), 只验证通道①②")
    ap.add_argument("--only", default=None, choices=["A", "B", "C"])
    args = ap.parse_args()

    groups = load_pairs()
    fam, stranger = pick_cast(groups, args.speaker)

    # 登记用真声 & 测试用真声(必须不同, 否则等于拿同一条音频自测)
    bon = sorted(groups[fam]["bonafide"])
    enroll_key = bon[0]
    test_bon_key = bon[1] if len(bon) > 1 else bon[0]
    spoof_key = args.spoof_key or sorted(groups[fam]["spoof"])[0]
    stranger_key = sorted(groups[stranger]["bonafide"])[0] if stranger else None

    def path(k):
        return os.path.join(DEV_FLAC, k + ".flac")

    print("=" * 68)
    print(" 谛听 VeriCall · 三通道全真实融合实战测试")
    print("=" * 68)
    print(f"'家人'说话人 : {fam}   (真声 {len(bon)} 条 / 伪造声 "
          f"{len(groups[fam]['spoof'])} 条)")
    print(f"  · 登记用真声   : {enroll_key}")
    print(f"  · 测试用真声   : {test_bon_key}")
    print(f"  · 伪造声(冒充) : {spoof_key}")
    if stranger:
        print(f"'陌生人'说话人 : {stranger}  -> {stranger_key}")

    from fusion.pipeline import VeriCallPipeline
    from fusion.acoustic_channel import AcousticChannel
    from fusion.fusion_orchestrator import ChannelVerdict

    pipe = VeriCallPipeline(acoustic=AcousticChannel(device=args.device))
    pipe.enroll("家人", path(enroll_key))
    print(f"[登记] 家人声纹 <- {enroll_key}.flac")

    if args.skip_semantic:
        # 用中性占位替代通道③, 专注验证声学+声纹
        pipe._semantic_verdict = lambda p: ChannelVerdict(
            name="semantic", score=0.0, label="normal",
            detail="(skip-semantic 占位)", confidence=0.0)

    results = {}

    def run(tag, title, audio, expect):
        print("\n" + "#" * 68)
        print(f"# 场景{tag}: {title}")
        print(f"# 音频: {os.path.basename(audio)}   期望: {expect}")
        print("#" * 68)
        r = pipe.analyze(audio)
        ok = (r.final == expect) or (expect == "block/caution"
                                     and r.final in ("block", "caution"))
        results[tag] = (r.final, expect, ok)
        print(f"[判定] {'✅ PASS' if ok else '❌ FAIL'}  "
              f"实际={r.final} 期望={expect}")
        return r

    if args.only in (None, "A"):
        run("A", "家人本人来电（真声，音色匹配）", path(test_bon_key), "allow")
    if args.only in (None, "B"):
        run("B", "AIGC 冒充家人 ⭐（同音色伪造声）", path(spoof_key), "block")
    if args.only in (None, "C") and stranger_key:
        run("C", "陌生人来电（真声，音色不匹配）", path(stranger_key),
            "block/caution")

    print("\n" + "=" * 68)
    print(" 汇总")
    print("=" * 68)
    for tag, (got, exp, ok) in results.items():
        print(f"  场景{tag}: {got:8s} (期望 {exp:14s}) {'✅' if ok else '❌'}")
    n_ok = sum(1 for _, _, ok in results.values() if ok)
    print(f"  通过 {n_ok}/{len(results)}")
    if not all(ok for _, _, ok in results.values()):
        print("\n  ⚠ 场景B 若未拦截, 说明通道①权重尚未训练到位"
              "（未训练时声学分数接近随机, 不会误拦也不会拦住）")
    return 0 if all(ok for _, _, ok in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
