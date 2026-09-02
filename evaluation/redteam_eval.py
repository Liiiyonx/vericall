#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 通道③ 红队方言评测（离线 text 模式 + 真实 audio 模式）
================================================================
两种运行模式：
  --mode text   （默认，离线）把 data/redteam/<方言>/manifest.json 里的 text 字段
                喂给规则评分器 rule_scorer，统计每方言 误拦率(FA)/漏拦率(MISS)/EER，
                并对诈骗样本施加对抗扰动算「逃避率」，产出可答辩的鲁棒性报表。
  --mode audio  （需模型）把真实 .wav 喂给 VeriCallPipeline 跑端到端裁决（见原逻辑）。

为何默认 text 离线模式：本 worktree 不含真实方言音频（在 Desktop 主仓库），且评委笔记本
可能无 GPU/Ollama。text 模式与真实管线共用同一套 rule_scorer 基线，能离线复现
「方言漏拦盲区 + 关键词易绕过」两个核心结论，比赛答辩最稳。

样本组织（data/redteam/README.md）：<方言>/manifest.json，项含
    {wav, text, dialect, label(scam|normal), note}。

用法：
    python evaluation/redteam_eval.py --build-corpus --mode text
    python evaluation/redteam_eval.py --mode text --list
    python evaluation/redteam_eval.py --mode text --out reports/redteam_text.json
    python evaluation/redteam_eval.py --mode audio --out reports/redteam_audio.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from paths import REDTEAM_DIR  # noqa: E402
import metrics  # noqa: E402
import redteam_corpus as corpus_mod  # noqa: E402
import redteam_perturb as rp  # noqa: E402


# ------------------------------------------------------------------ #
# 评测阈值与默认扰动策略
# ------------------------------------------------------------------ #
DEFAULT_THRESHOLD = 0.5          # risk >= 0.5 判为「检测到诈骗」
DEFAULT_STRATEGIES = ("space", "homophone")


# ------------------------------------------------------------------ #
# text 模式
# ------------------------------------------------------------------ #
def _score_text(text: str) -> float:
    from fusion.rule_scorer import rule_score
    return rule_score(text).risk


def evaluate_text(samples, threshold: float = DEFAULT_THRESHOLD,
                 strategies=DEFAULT_STRATEGIES) -> dict:
    """对文本样本做方言鲁棒性评测，返回结果字典（供 report 消费）。"""
    # 预打分（每条样本一次）
    recs = [{"dialect": s.dialect, "label": s.label,
             "risk": _score_text(s.text), "text": s.text} for s in samples]

    by_dialect: dict[str, list[dict]] = {}
    for r in recs:
        by_dialect.setdefault(r["dialect"], []).append(r)

    per_dialect = []
    all_norm: list[float] = []
    all_scam: list[float] = []
    for dialect in sorted(by_dialect):
        ss = by_dialect[dialect]
        normals = [r for r in ss if r["label"] == "normal"]
        scams = [r for r in ss if r["label"] == "scam"]
        fa = sum(1 for r in normals if r["risk"] >= threshold)
        miss = sum(1 for r in scams if r["risk"] < threshold)
        n_n, n_s = len(normals), len(scams)
        row = {
            "dialect": dialect,
            "n_normal": n_n, "n_scam": n_s,
            "fa": fa, "miss": miss,
            "fa_pct": round(fa / n_n * 100, 2) if n_n else 0.0,
            "miss_pct": round(miss / n_s * 100, 2) if n_s else 0.0,
        }
        if normals and scams:
            eer, thr = metrics.compute_eer(
                [r["risk"] for r in normals], [r["risk"] for r in scams])
            row["eer_pct"] = round(eer, 4)
            row["eer_threshold"] = round(thr, 6)
        else:
            row["eer_pct"] = None
            row["eer_threshold"] = None
        per_dialect.append(row)
        all_norm += [r["risk"] for r in normals]
        all_scam += [r["risk"] for r in scams]

    # 总体
    n_n = len(all_norm)
    n_s = len(all_scam)
    overall_fa = sum(1 for r in recs if r["label"] == "normal" and r["risk"] >= threshold)
    overall_miss = sum(1 for r in recs if r["label"] == "scam" and r["risk"] < threshold)
    overall = {
        "n_normal": n_n, "n_scam": n_s,
        "fa": overall_fa, "miss": overall_miss,
        "fa_pct": round(overall_fa / n_n * 100, 2) if n_n else 0.0,
        "miss_pct": round(overall_miss / n_s * 100, 2) if n_s else 0.0,
    }
    if all_norm and all_scam:
        eer, thr = metrics.compute_eer(all_norm, all_scam)
        overall["eer_pct"] = round(eer, 4)
        overall["eer_threshold"] = round(thr, 6)

    # 对抗鲁棒性：在「当前已检出的诈骗」上施加扰动，看多少被绕过
    detected_scams = [r["text"] for r in recs
                      if r["label"] == "scam" and r["risk"] >= threshold]
    adv = {}
    for st in strategies:
        if not detected_scams:
            adv[st] = 0.0
            continue
        evaded = sum(1 for t in detected_scams
                     if _score_text(rp.perturb(t, st)) < threshold)
        adv[st] = round(evaded / len(detected_scams) * 100, 2)

    return {
        "mode": "text",
        "threshold": threshold,
        "strategies": list(strategies),
        "overall": overall,
        "per_dialect": per_dialect,
        "adversarial_evasion_pct": adv,
        "n_detected_scams": len(detected_scams),
        "records": recs,
    }


def render_markdown(result: dict, meta: dict) -> str:
    ov = result["overall"]
    lines = [
        "# 谛听 VeriCall · 通道③ 红队方言鲁棒性评测报告",
        "",
        f"- 生成时间：{meta['timestamp']}",
        f"- 模式：text（离线规则基线 rule_scorer，阈值 risk≥{result['threshold']} 判诈骗）",
        f"- 对抗策略：{', '.join(result['strategies'])}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 样本数 | {ov['n_normal']} 正常 / {ov['n_scam']} 诈骗 |",
        f"| **误拦率 FA** | **{ov['fa_pct']:.2f}%**（{ov['fa']}/{ov['n_normal']}） |",
        f"| **漏拦率 MISS** | **{ov['miss_pct']:.2f}%**（{ov['miss']}/{ov['n_scam']}） |",
    ]
    if "eer_pct" in ov:
        lines += [
            f"| **EER** | **{ov['eer_pct']:.3f}%**（阈值 {ov['eer_threshold']:.4f}） |",
        ]
    lines += [
        f"| 当前检出诈骗数 | {result['n_detected_scams']} / {ov['n_scam']} |",
        "",
        "## 按方言细分",
        "",
        "| 方言 | 正常 | 诈骗 | 误拦FA | 漏拦MISS | EER |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["per_dialect"]:
        eer = f"{r['eer_pct']:.3f}%" if r["eer_pct"] is not None else "—"
        lines.append(
            f"| {r['dialect']} | {r['n_normal']} | {r['n_scam']} "
            f"| {r['fa_pct']:.2f}% | {r['miss_pct']:.2f}% | {eer} |")

    lines += ["", "## 对抗鲁棒性（逃避率）", ""]
    if result["n_detected_scams"]:
        lines.append(
            f"对「当前已检出的 {result['n_detected_scams']} 条诈骗」施加字符级扰动，"
            f"观察多少能被绕过（risk 跌到阈值以下）：")
        lines.append("")
        lines.append("| 扰动策略 | 逃避率 |")
        lines.append("|---|---|")
        for st, ev in result["adversarial_evasion_pct"].items():
            lines.append(f"| {st} | {ev:.2f}% |")
    else:
        lines.append("当前无检出的诈骗样本，对抗评测无意义（先解决漏拦）。")

    # 判定 / 建议
    worst = max(result["per_dialect"], key=lambda r: r["miss_pct"])
    lines += [
        "",
        "## 判定与加固建议",
        "",
        f"- 漏拦最严重方言：**{worst['dialect']}**，MISS {worst['miss_pct']:.2f}%"
        f"（方言词/繁体绕过关键词 → 规则基线失效）。",
        "- 方言盲区需引入：① 文本归一化（繁→简、谐音映射）；② 方言适配的关键词/语义向量检测。",
        f"- 对抗逃避率高说明关键词子串匹配脆弱，需叠加字符级归一化 + 语义向量相似度。",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# audio 模式（保留原逻辑，需真实管线 + 模型）
# ------------------------------------------------------------------ #
def collect_audio(scenario: str | None = None) -> list[tuple[str, Path, dict]]:
    out: list[tuple[str, Path, dict]] = []
    if not REDTEAM_DIR.is_dir():
        return out
    for d in sorted(REDTEAM_DIR.iterdir()):
        if not d.is_dir():
            continue
        if scenario and d.name != scenario:
            continue
        man = d / "manifest.json"
        rows = json.loads(man.read_text(encoding="utf-8")) if man.is_file() else []
        by_name = {Path(r.get("wav", "")).name: r for r in rows}
        for w in sorted(d.glob("*.wav")):
            out.append((d.name, w, by_name.get(w.name, {})))
    return out


def run_audio(samples: list, out_path: str | None = None) -> list[dict]:
    from fusion.pipeline import VeriCallPipeline
    pipe = VeriCallPipeline()
    records: list[dict] = []
    for scen, wav, row in samples:
        t0 = time.time()
        try:
            r = pipe.analyze(str(wav)).to_dict()
            status = "ok"
        except Exception as e:  # noqa: BLE001
            r, status = {"error": str(e)}, "fail"
        rec = {"scenario": scen, "wav": wav.name, "expected": row.get("label"),
               "note": row.get("note", ""), "status": status,
               "elapsed_s": round(time.time() - t0, 1), "result": r}
        records.append(rec)
        print(f"[{scen}] {wav.name} -> {status} {r.get('final')}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报表 -> {out_path}")
    return records


# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser(description="通道③ 红队方言评测")
    ap.add_argument("--mode", default="text", choices=["text", "audio"])
    ap.add_argument("--build-corpus", action="store_true",
                    help="评测前先(重)生成合成方言语料")
    ap.add_argument("--list", action="store_true", help="只列出样本，不评测")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                    help="对抗策略，逗号分隔")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent /
                                "reports" /
                                f"redteam_text_{datetime.now():%Y%m%d_%H%M%S}.json"))
    args = ap.parse_args()

    if args.build_corpus:
        ps = corpus_mod.build_corpus()
        print(f"[corpus] 已生成 {len(ps)} 个方言 manifest")

    if args.mode == "audio":
        samples = collect_audio(args.scenario)
        if not samples:
            print(f"未找到红队音频样本（wav）。请在 {REDTEAM_DIR} 下放置样本，"
                  f"或用 --mode text 跑离线文本评测。")
            return
        if args.list:
            for scen, wav, row in samples:
                print(f"  {scen}/{wav.name}  label={row.get('label')}  "
                      f"note={row.get('note', '')}")
            return
        run_audio(samples, args.out)
        return

    # ---- text 模式 ----
    samples = corpus_mod.load_text_corpus(REDTEAM_DIR)
    if not samples:
        print(f"未找到文本样本。请先 `python evaluation/redteam_corpus.py` "
              f"或加 --build-corpus 生成 data/redteam/<方言>/manifest.json。")
        return
    if args.list:
        for s in samples:
            print(f"  [{s.dialect}] {s.label:6s}  {s.text[:40]}")
        return

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    result = evaluate_text(samples, threshold=args.threshold, strategies=strategies)

    ov = result["overall"]
    print("=" * 64)
    print(f"通道③ 红队 text 评测  thr={args.threshold}  策略={strategies}")
    print("=" * 64)
    print(f"总体：FA={ov['fa_pct']:.2f}%  MISS={ov['miss_pct']:.2f}%  "
          f"EER={ov.get('eer_pct', '—')}")
    for r in result["per_dialect"]:
        print(f"  {r['dialect']:10s} n={r['n_normal']+r['n_scam']:2d}  "
              f"FA={r['fa_pct']:.1f}%  MISS={r['miss_pct']:.1f}%")
    print(f"对抗逃避率：{result['adversarial_evasion_pct']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"redteam_text_{ts}.json"
    md_path = out_dir / f"redteam_text_{ts}.md"
    meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "text", "threshold": args.threshold}
    json_path.write_text(json.dumps({"meta": meta, "result": result},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result, meta), encoding="utf-8")
    print("-" * 64)
    print(f"报告：\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()
