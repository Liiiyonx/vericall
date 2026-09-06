# -*- coding: utf-8 -*-
"""
任务书 P1-5 第 2 步：跑全链路为场景集填充分数矩阵
==================================================
读 evaluation/scenario_manifest.csv，对每条场景跑三通道真实推理，落盘
evaluation/scenario_scores.csv（列：id, acoustic, voiceprint, semantic,
expected, confidence_a, confidence_v, confidence_s）。

三通道：
  通道① 声学 AASIST     —— 真实 GPU 推理（权重 external/aasist/exp_result/.../best.pth）
  通道② 声纹 CAMPPlus    —— 用每位「家人」的第一条真声 enroll，再 verify 该家人其余样本 /
                              陌生人对全体家人的最高相似度
  通道③ 话术 SenseVoice —— ASR 转写 + 规则评分器（rule_scorer，确定性、无需 Ollama，
                              标定实验要可复现，不掺 LLM 随机性）

用法：
  python scripts/populate_scenario_scores.py            # 全量
  python scripts/populate_scenario_scores.py --limit 10 # 快速冒烟
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import DEV_FLAC  # noqa: E402
from fusion.rule_scorer import rule_score  # noqa: E402

MANIFEST = ROOT / "evaluation" / "scenario_manifest.csv"
OUT = ROOT / "evaluation" / "scenario_scores.csv"


def _load_channels():
    """懒加载三通道，仅首次使用拉模型。"""
    from fusion.acoustic_channel import AcousticChannel
    from fusion.voiceprint_channel import VoiceprintChannel
    from fusion.semantic_channel import SenseVoiceASR
    ac = AcousticChannel()           # 自动找 best.pth + exp config
    vp = VoiceprintChannel()
    asr = SenseVoiceASR()
    return ac, vp, asr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    args = ap.parse_args()

    if not MANIFEST.is_file():
        print(f"[err] 缺 {MANIFEST}，先跑 scripts/build_scenarios.py")
        return
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("audio") and r["audio"] not in ("TODO",)]
    if not rows:
        print("[err] manifest 无有效音频行（说明场景数据未就位）")
        return
    if args.limit:
        rows = rows[:args.limit]

    ac, vp, asr = _load_channels()
    print(f"[load] 三通道就绪，待处理 {len(rows)} 条")

    # 家人 = manifest 中 kind 非 stranger 的说话人。登记声纹必须用「真声」，
    # 故优先取该说话人的 bonafide 样本；若清单里只有 spoof（克隆声）则跳过登记，
    # 该类场景 voiceprint 通道给不出可靠相似度（由声学通道负责拦截）。
    enroll_src: dict[str, str] = {}
    for r in rows:
        if r["kind"] == "bonafide" and r["speaker"] not in enroll_src:
            enroll_src[r["speaker"]] = r["audio"]
    family = {r["speaker"] for r in rows if r["kind"] in ("bonafide", "spoof")}
    no_enroll = sorted(family - set(enroll_src))
    if no_enroll:
        print(f"[info] 以下家人无 bonafide 样本，voiceprint 按陌生人处理: {no_enroll}")
    for spk, aud in enroll_src.items():
        try:
            vp.enroll(spk, aud)
            print(f"[enroll] {spk} <- {Path(aud).name}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 家人登记失败 {spk}: {e}")

    out_rows = []
    for i, r in enumerate(rows, 1):
        aud = r["audio"]
        t0 = time.time()
        try:
            # 通道① 声学（AASIST 常驻推理，不逐条卸载）
            acv = ac.analyze(aud)
            # 通道② 声纹：家人 → 对该说话人已登记向量比对；陌生人 → 对全体家人取最高
            try:
                vpv = vp.verify(aud)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 声纹失败 {Path(aud).name}: {e}")
                vpv = None
            # 通道③ 话术：ASR + 规则评分（确定性）
            try:
                txt = asr.transcribe(aud)
                rr = rule_score(txt)
                sem_score = rr.risk
                sem_conf = 0.9 if rr.category != "normal" else 0.7
            except Exception as e:  # noqa: BLE001
                print(f"[warn] ASR 失败 {Path(aud).name}: {e}")
                sem_score, sem_conf = 0.05, 0.3

            out_rows.append({
                "id": r["id"],
                "acoustic": round(acv.score, 4),
                "voiceprint": round(vpv.score, 4) if vpv else "",
                "semantic": round(sem_score, 4),
                "expected": r["expected"],
                "ca": round(acv.confidence, 3),
                "cv": round(vpv.confidence, 3) if vpv else "",
                "cs": sem_conf,
            })
            print(f"[{i}/{len(rows)}] {Path(aud).name} A={acv.score:.2f} "
                  f"V={vpv.score if vpv else -1:.2f} S={sem_score:.2f} "
                  f"({time.time()-t0:.1f}s) exp={r['expected']}")
        except Exception as e:  # noqa: BLE001
            print(f"[err] {Path(aud).name}: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()) if out_rows
                           else ["id", "acoustic", "voiceprint", "semantic",
                                 "expected", "ca", "cv", "cs"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n完成 {len(out_rows)}/{len(rows)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
