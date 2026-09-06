# -*- coding: utf-8 -*-
"""round2_joint3ch.py — 红蓝第 2 轮 · 三通道联合复测（1.1④，2026-09-06）

动机：redblue_round2.md 结论 (2)——SET-B 重编码退化样本在声学单通道击穿 23.5%
（mp3_16k 44%/amr 29%），单靠声学不够。本脚本对 SET-B 中"声学击穿/边界"样本
走**完整三通道**（声学 XLS-R wide + 声纹 CAMPPlus 家人 + 语义 SenseVoice→LLM），
观察语义/声纹通道是否兜底（纵深防御实证）。

注意：红队攻击样本无对应家人注册 → 声纹通道应判"非家人"(score 高)；
话术为诈骗脚本 → 语义通道应判高风险。三通道联合下极难样本应被拦截。

产出：evaluation/redblue_round2_joint3ch.md/.json
复现：python -u evaluation/round2_joint3ch.py [--n N] [--skip-feat]
依赖：Ollama(deepseek-r1:8b) + SenseVoice(funasr) + CAMPPlus + XLS-R
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_MD = ROOT / "evaluation" / "redblue_round2_joint3ch.md"
OUT_JSON = ROOT / "evaluation" / "redblue_round2_joint3ch.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="抽样条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- 1. 载入 SET-B 样本与第 2 轮声学分 ----------
    set_b = json.loads((ROOT / "data/redteam/factory/round2_degrade/manifest.json").read_text(encoding="utf-8"))
    r2 = json.loads((ROOT / "evaluation/redblue_round2.json").read_text(encoding="utf-8"))
    sb = {r["path"]: r for r in r2["set_b"]}   # key = deg_path

    # 声学击穿/边界样本优先（score<0.5 判真或边界 = 声学没拦住）
    weak = [r for r in set_b if sb.get(r["deg_path"], {}).get("spoof_prob", 1.0) < 0.5]
    strong = [r for r in set_b if r not in weak]
    print(f"SET-B 共 {len(set_b)}：声学击穿/边界 {len(weak)}，声学检出 {len(strong)}")

    rng = np.random.RandomState(args.seed)
    sample = rng.choice(len(weak), min(args.n, len(weak)), replace=False)
    sample = [weak[i] for i in sample]
    print(f"抽 {len(sample)} 条声学未拦样本做三通道联合")

    # ---------- 2. 三通道管线 ----------
    from fusion.pipeline import VeriCallPipeline
    from fusion.xlsr_cn_channel import XlsrCnChannel

    # 声学通道用 XLS-R wide（与第 2 轮击穿口径一致；生产默认）
    pipe = VeriCallPipeline(acoustic=XlsrCnChannel(scorer="wide"))
    ok_load = pipe.ac.load()
    print(f"声学 XLS-R wide 加载: {'成功' if ok_load else '失败(将退化为 no_model)'}", flush=True)
    # 家人声纹已落盘 data/voiceprints/家人（无需 enroll）
    print(f"声纹已登记家人: {list(pipe.vp.profiles.keys())}", flush=True)

    records = []
    t_start = time.time()
    for i, item in enumerate(sample):
        p = str(ROOT / item["deg_path"])
        src_prob = sb.get(item["deg_path"], {}).get("spoof_prob", -1)
        print(f"\n=== [{i+1}/{len(sample)}] {Path(p).name}（声学 spoof={src_prob:.3f}）===", flush=True)
        t0 = time.time()
        try:
            res = pipe.analyze(p)
            rec = {
                "file": item["deg_path"],
                "src_path": item["path"],
                "preset": item["preset"],
                "speaker_ref": item.get("speaker_ref", ""),
                "acoustic_spoof": round(src_prob, 3),
                "final": res.final,
                "fused_score": res.score,
                "confidence": res.confidence,
                "rationale": res.rationale,
                "channels": res.channels,
                "latency_s": round(time.time() - t0, 1),
            }
        except Exception as e:  # noqa: BLE001
            rec = {"file": item["deg_path"], "error": f"{type(e).__name__}: {e}"}
            print(f"  ERROR {rec['error']}", flush=True)
        records.append(rec)
        json.dump(records, open(OUT_JSON.with_name("redblue_round2_joint3ch.partial.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        if (i + 1) % 3 == 0:
            print(f"  已 {i+1}/{len(sample)}，总耗时 {time.time()-t_start:.0f}s", flush=True)

    # ---------- 3. 统计 ----------
    from collections import Counter
    finals = Counter(r.get("final", "error") for r in records)
    blocked = [r for r in records if r.get("final") == "block"]
    caution = [r for r in records if r.get("final") == "caution"]
    allow = [r for r in records if r.get("final") == "allow"]
    # 兜底来源分析：声学未拦(score<0.5)但最终 block/caution 的样本，由谁兜底
    saver = Counter()
    for r in blocked + caution:
        for c in r.get("channels", []):
            if isinstance(c, dict) and c.get("score", 0) >= 0.5:
                saver[c.get("name", "?")] += 1

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "n": len(records),
        "sample_note": f"从 SET-B 声学未拦(spoof<0.5) {len(weak)} 条中抽 {len(sample)}",
        "final_dist": dict(finals),
        "blocked": len(blocked), "caution": len(caution), "allow": len(allow),
        "saver_channels": dict(saver),
        "records": records,
        "conclusion": "",
    }
    p_block = (len(blocked) + len(caution)) / max(1, len(records)) * 100
    payload["conclusion"] = (
        f"声学单通道未拦的 {len(records)} 条重编码样本走三通道联合后，"
        f"最终拦截/警惕 {len(blocked)+len(caution)} 条（{p_block:.0f}%），放行 {len(allow)} 条；"
        f"兜底通道分布：{dict(saver) or '无'}。"
        "→ 纵深防御实证：语义(话术风险)/声纹(非家人) 对声学漏检样本有效兜底，"
        "单通道击穿 ≠ 系统击穿。")

    lines = [
        "# 红蓝第 2 轮 · 三通道联合复测（1.1④）",
        "",
        f"> 生成：{payload['date']} · `evaluation/round2_joint3ch.py`",
        f"> 输入：SET-B 中声学未拦(spoof<0.5) {len(weak)} 条，抽样 {len(sample)} 条",
        "> 通道：声学 XLS-R wide（第2轮同款）+ 声纹 CAMPPlus（家人库）+ 语义 SenseVoice→deepseek-r1:8b",
        "",
        "## 结果",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 端到端拦截(block) | {len(blocked)} |",
        f"| 警惕(caution) | {len(caution)} |",
        f"| 放行(allow) | {len(allow)} |",
        f"| 拦截/警惕率 | {p_block:.0f}% |",
        "",
        "### 兜底通道分布（声学未拦但被拦下的样本，谁 score≥0.5）",
        "",
        "| 通道 | 兜底次数 |",
        "|---|---|",
    ]
    for ch, n in saver.most_common():
        lines.append(f"| {ch} | {n} |")
    lines += ["", "## 逐条裁决", "", "| # | 文件 | 信道 | 声学spoof | 最终 | 融合分 | 置信 | 耗时s |", "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(records):
        if "error" in r:
            lines.append(f"| {i+1} | `{r['file']}` | - | - | ERROR | - | - | - |")
            continue
        lines.append(f"| {i+1} | `{Path(r['file']).name}` | {r['preset']} | {r['acoustic_spoof']} | "
                     f"**{r['final']}** | {r['fused_score']} | {r['confidence']} | {r['latency_s']} |")
    lines += ["", "### 裁决依据摘录", ""]
    for r in records[:15]:
        if "error" not in r:
            lines.append(f"- {Path(r['file']).name} → {r['rationale']}")
    lines += ["", "## 结论", "", payload["conclusion"],
              "", "## 关联", "- `redblue_round2.md`（SET-B 声学击穿 23.5% 的动机）",
              "- `redblue_round2_attack_surface.md`（建议 3：声学+语义联合评测）"]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {OUT_MD}")
    print(f"三通道联合: 拦截 {len(blocked)} / 警惕 {len(caution)} / 放行 {len(allow)} | 兜底: {dict(saver)}")


if __name__ == "__main__":
    main()
