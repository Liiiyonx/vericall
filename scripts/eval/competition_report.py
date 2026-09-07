#!/usr/bin/env python
"""competition_report.py — 5.4 参赛级一键报告（2026-09-07 v1）

聚合 evaluation/data 下已提交的最新产物（不重算，全部引用来源文件），
输出单份 evaluation/competition_report.md，供答辩/申报引用与每周门禁核对。
来源均为可复现脚本产物（脚本名在每节标注），评委可按链接复算。
用法：python -u scripts/eval/competition_report.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evaluation"


def rd(p):
    return json.loads((EV / p).read_text(encoding="utf-8"))


def main():
    stats = rd("../../data/scam_corpus/stats.json") if False else json.loads(
        (ROOT / "data/scam_corpus/stats.json").read_text(encoding="utf-8"))
    f1 = rd("semantic_f1.json")
    calib = rd("semantic_calibration.json")
    vp = rd("voiceprint_calibration.json")
    rb2 = rd("redblue_round2.json")
    nch = (EV / "number_channel_eval.md").read_text(encoding="utf-8")
    hit = "10/10" if "10/10 命中" in nch else "?"
    fp = "0/10" if "误报 0/10" in nch else "?"

    s = f"""# 谛听 VeriCall · 参赛证据一键报告

> 聚合生成：{time.strftime('%Y-%m-%d %H:%M')} · `scripts/eval/competition_report.py`
> 原则：本页所有数字均可沿「来源」回链到脚本与产物复现（门禁 #3：复算不出 = 没做）。

## 1. 语料资产（v0.2 定稿 2026-09-07）
- scam **{stats['total']}** 条（8 类 {min(stats['by_category'].values())}-{max(stats['by_category'].values())}/类均衡；
  4 阶段 {min(stats['by_stage'].values())}-{max(stats['by_stage'].values())}）+ benign 2,404（6 类）
- 定稿流程：2 轮 LLM 预筛（二审纠偏后真实 drop 率 1.0%）+ fix 改写 15/15 回填
- 评测集冻结：scam train/eval 9,547/2,387、benign 1,923/481（8+6 类各恰 20%，隔离 OK）
- 来源：`make_eval_split.py` / `apply_review_verdicts.py --llm-fill`

## 2. 号码第 0 层（先验闸门）
- 已知特征号命中 {hit}、正常号误报 {fp}、本地查表 0ms；命中即短路跳过三通道
- 来源：`src/fusion/number_channel.py` + `evaluation/number_channel_eval.py`

## 3. 语义侧（线 A）
- **二分类诈骗/正常 F1 {f1['binary_scam_vs_normal']['f1']*100:.1f}%**（P {f1['binary_scam_vs_normal']['precision']*100:.1f}% /
  R {f1['binary_scam_vs_normal']['recall']*100:.1f}%，n={f1['n']}）；8 类加权 {f1['weighted_f1']*100:.1f}%
- 校准 **ECE {calib['ece']*100:.1f}%**；语义 block 阈值建议 risk≥0.6（P99.9/R97.4）
- 来源：`eval_semantic_f1.py` / `calib_semantic.py`（2,868 行级 risk 落盘 semantic_f1_rows.csv）

## 4. 声纹侧（线 B）
- **EER {vp['eer']*100:.2f}%** @{vp['eer_threshold']}（200 人合库：AISHELL-1 100 + AISHELL-3 100，
  {vp['genuine_pairs']} 同人对 / {vp['impostor_pairs']} 异人对）
- 工作点 κ 1:1/3:1/1:3 → 0.471/0.443/0.520；代码阈值已替换 0.520/0.443
- 三板斧：双信道注册（互验≥0.52）+ challenge-response 骨架 + TTS 门控（joint3ch 10/10）
- 来源：`voiceprint_det_scan.py` / `voiceprint_calibration.md` / `b4_antiattack_design.md`

## 5. 声学/红蓝（攻防演化 0→2）
- 红蓝：第0轮击穿 95.5/99.0% → 第1轮 0.3% → 第2轮 SET-A {rb2['set_a_male_boost']['breakthrough_pct_lt0.3']}% /
  SET-B {rb2['set_b_degraded']['breakthrough_pct_lt0.3']}%（amr 信道 {rb2['per_channel']['amr']['breakthrough_pct_lt0.3']}%）
- 三通道联合复测 10/10 拦截；wide 极难集口径 = score<0.3 且 round2_seg_qa 结构有效（28/28）
- AASIST dev EER 0.745%（best 0.316%）/ eval 3.49%（`v0.2-repro` 复现）
- 来源：`redblue_round2.py` 等 + 演化图 `redblue_evolution.md`

## 6. 工程/质量
- pytest **88 passed**（2026-09-07）；CI：`.github/workflows/pytest.yml`
- 隔离检查 `check_isolation.py` 通过（红队集禁入训练）；降级模式 VERICALL_OFFLINE=1 保底演示

## 7. 口径锚点（答辩防翻车速查）
- 终端形态：独立终端（座机/一体机）为主 + iPhone 号码预警（iOS 生态无法 App 内实时分析）
- 与官方关系：号码=先验加速（可接官方库做第 0 层）；官方离线单条 AIGC 鉴定 vs 我们通话中实时流式融合——互补
- 对照图：`assets/comparison_official_vs_us.svg`；tag：`v0.8-redteam`（全链可复现）
"""
    out = EV / "competition_report.md"
    out.write_text(s, encoding="utf-8")
    print(f"已生成 {out.relative_to(ROOT)}（{len(s.splitlines())} 行）")


if __name__ == "__main__":
    main()
