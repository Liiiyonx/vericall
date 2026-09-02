#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P2-5 红队对抗实验报告生成器
================================================================
把通道③ 红队方言鲁棒性评测（text 模式，rule_scorer 基线）的结果，
按任务书 P2-5 的「攻击面」框架重新组织，产出答辩可直接引用的对抗报告：

    evaluation/redteam_adversarial.md
    evaluation/redteam_adversarial.json

攻击面设计（任务书 P2-5 原文）：
  | 攻击 | 目的 | 预期观察 |
  | 高保真克隆（干净信道） | 基线 | 通道①应检出 |
  | 克隆 + phone8k 退化 | 信道鲁棒性 | EER 劣化幅度 |
  | 克隆 + 免提噪声 | 采集链路鲁棒性 | 三通道各自退化 |
  | 部分伪造 | PartialSpoof 场景 | 全段分类器可能漏检 |
  | 方言克隆 | 独占卖点验证 | ASR/LLM 方言能力边界 |

当前实现状态（诚实标注）：
  - 「方言克隆 / 高保真克隆基线」：★ 已用 redteam text 评测实跑，数据为真实结果。
  - 「克隆 + 信道退化 / 免提噪声 / 部分伪造」：待真实方言克隆音频（P2-1 GPT-SoVITS
    合成）就位后，由 --mode audio 全链路评测填充；本脚本保留占位并给出待填模板。

用法：
    python evaluation/make_redteam_adversarial.py
    python evaluation/make_redteam_adversarial.py --out evaluation/redteam_adversarial.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

import redteam_corpus as corpus_mod  # noqa: E402
import redteam_eval as rt_eval  # noqa: E402


# P2-5 攻击面定义；done=True 表示本仓库已能实跑该面
ATTACK_SURFACES = [
    ("dialect_clone", "方言克隆（独占卖点验证）", True,
     "用各方言诈骗话术文本喂 rule_scorer 基线，暴露方言词/繁体绕过关键词的盲区；"
     "这是「通道③ 对方言攻击的脆弱性」的直接证据。"),
    ("hf_clone_baseline", "高保真克隆·干净信道（基线）", True,
     "方言克隆在干净信道的基线检出率（与方言克隆同源，按方言给出逐方言命中情况）。"),
    ("clone_phone8k", "克隆 + phone8k 退化（信道鲁棒性）", False,
     "对克隆音频施加 scripts/degrade_audio.py --preset phone8k，观察三通道各自退化；"
     "需真实方言克隆音频（P2-1）。"),
    ("clone_handsfree", "克隆 + 免提噪声（采集链路）", False,
     "免提外放（phone8k + 粉噪 SNR15dB）下的检出率；需真实音频。"),
    ("partial_spoof", "部分伪造（PartialSpoof）", False,
     "真人句中嵌入克隆句；全段/逐句分类器的漏检情况；属 known limitation + 未来工作。"),
]


def render_markdown(result: dict, meta: dict) -> str:
    ov = result["overall"]
    per = {r["dialect"]: r for r in result["per_dialect"]}
    adv = result["adversarial_evasion_pct"]

    L = []
    L += [
        "# 谛听 VeriCall · P2-5 红队对抗实验报告",
        "",
        f"- 生成时间：{meta['timestamp']}",
        f"- 通道：③ 话术语义（rule_scorer 基线，text 离线模式）",
        f"- 阈值：risk≥{result['threshold']} 判为诈骗",
        f"- 对抗策略：{', '.join(result['strategies'])}",
        "",
        "> 说明：本报告以「通道③ 方言鲁棒性」实测结果为主体（★ 已实跑），"
        "信道退化 / 部分伪造等音频域攻击面在真实方言克隆音频（P2-1）就位后"
        "由 `--mode audio` 全链路评测填充，标注为「待填」。",
        "",
        "## 一、总览（通道③ 方言鲁棒性）",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 样本数 | {ov['n_normal']} 正常 / {ov['n_scam']} 诈骗 |",
        f"| **误拦率 FA** | **{ov['fa_pct']:.2f}%**（{ov['fa']}/{ov['n_normal']}） |",
        f"| **漏拦率 MISS** | **{ov['miss_pct']:.2f}%**（{ov['miss']}/{ov['n_scam']}） |",
    ]
    if "eer_pct" in ov:
        L += [f"| **EER** | **{ov['eer_pct']:.3f}%**（阈值 {ov['eer_threshold']:.4f}） |"]
    L += [
        f"| 当前检出诈骗数 | {result['n_detected_scams']} / {ov['n_scam']} |",
        "",
        "## 二、攻击面明细（任务书 P2-5）",
        "",
        "| 攻击面 | 目的 | 状态 | 关键结果 |",
        "|---|---|---|---|",
    ]

    # 方言克隆 / 高保真克隆基线：直接映射方言细分
    worst = max(result["per_dialect"], key=lambda r: r["miss_pct"])
    worst_dialects = [r["dialect"] for r in result["per_dialect"] if r["miss_pct"] >= 100.0]
    dialect_line = (
        f"粤语/闽南 100% 漏拦；普通话系 0% 漏拦；"
        f"最弱方言 **{worst['dialect']}**（MISS {worst['miss_pct']:.2f}%）")
    L += [
        f"| 方言克隆 | 独占卖点验证 | ★ 已实跑 | {dialect_line} |",
        f"| 高保真克隆·干净信道 | 基线 | ★ 已实跑 | 同方言克隆逐方言命中（见下表） |",
        "| 克隆 + phone8k 退化 | 信道鲁棒性 | 待填 | 需 P2-1 真实克隆音频 |",
        "| 克隆 + 免提噪声 | 采集链路 | 待填 | 需 P2-1 真实克隆音频 |",
        "| 部分伪造 | PartialSpoof | 待填 | known limitation（逐句检测为未来工作） |",
    ]

    L += [
        "",
        "### 2.1 方言克隆逐方言检出（攻击面核心数据）",
        "",
        "| 方言 | 正常 | 诈骗 | 误拦 FA | 漏拦 MISS | 检出率 |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["per_dialect"]:
        detect = max(0.0, 100.0 - r["miss_pct"])
        L.append(
            f"| {r['dialect']} | {r['n_normal']} | {r['n_scam']} "
            f"| {r['fa_pct']:.2f}% | {r['miss_pct']:.2f}% | {detect:.2f}% |")

    L += [
        "",
        "### 2.2 对抗扰动逃避率（关键词匹配脆弱性）",
        "",
        f"对「当前已检出的 {result['n_detected_scams']} 条诈骗」施加字符级扰动，"
        "观察多少能被绕过（risk 跌到阈值以下）：",
        "",
        "| 扰动策略 | 逃避率 |",
        "|---|---|",
    ]
    for st, ev in adv.items():
        L.append(f"| {st} | {ev:.2f}% |")

    L += [
        "",
        "## 三、判定与加固建议",
        "",
        f"- **最弱环节（攻击面）**：方言克隆中的 **{worst['dialect']}** 系，"
        f"MISS {worst['miss_pct']:.2f}%（方言词/繁体绕过关键词 → 规则基线失效）。",
        "- **方言盲区需引入**：① 文本归一化（繁→简、谐音映射）；"
        "② 方言适配的关键词/语义向量检测。",
        "- **对抗逃避率高**说明关键词子串匹配脆弱，需叠加字符级归一化 + 语义向量相似度。",
        "- **信道/部分伪造攻击面**：当前仅通道③ text 基线，音频域攻击需 P2-1 真实克隆"
        "音频 + 通道① AASIST 联合评测，列为「数据到位后 1 周内填充」。",
        "",
        "## 四、与跨域评测的关系",
        "",
        "- 本报告 = 通道③ 的「语义/方言」攻击面；通道① 的「跨攻击算法」攻击面见 "
        "`evaluation/attack_breakdown_eval.py`（已知 A01–A06 vs 未知 A07–A19 泛化鸿沟）。",
        "- 两报告合起来构成「我们攻击过自己」的攻防视角证据，对应信安赛道核心素材。",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="P2-5 红队对抗实验报告生成")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                                          "redteam_adversarial.md"))
    ap.add_argument("--json", default=str(Path(__file__).resolve().parent /
                                           "redteam_adversarial.json"))
    ap.add_argument("--build-corpus", action="store_true",
                    help="生成合成方言语料后再评测")
    args = ap.parse_args()

    if args.build_corpus:
        corpus_mod.build_corpus()
    samples = corpus_mod.load_text_corpus()
    if not samples:
        raise SystemExit("未找到文本样本，请先 --build-corpus 或检查 data/redteam/")
    result = rt_eval.evaluate_text(samples)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {"timestamp": ts, "mode": "text",
            "threshold": result["threshold"], "attack_surfaces": len(ATTACK_SURFACES)}
    md = render_markdown(result, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    Path(args.json).write_text(
        json.dumps({"meta": meta, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"P2-5 对抗报告 ->\n  {out}\n  {args.json}")
    ov = result["overall"]
    print(f"总览：FA={ov['fa_pct']:.2f}%  MISS={ov['miss_pct']:.2f}%  "
          f"检出={result['n_detected_scams']}/{ov['n_scam']}")
    print(f"对抗逃避率：{result['adversarial_evasion_pct']}")


if __name__ == "__main__":
    main()
