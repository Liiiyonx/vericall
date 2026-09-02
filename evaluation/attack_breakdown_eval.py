#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 通道① 跨攻击域细分评测
================================================================
回答一个总 EER 回答不了的问题：**模型遇到没见过的攻击算法还灵不灵？**

ASVspoof2019 LA 的攻击划分（这是本评测的核心设定）：
    dev  集 A01–A06  —— 训练集里出现过，**已知攻击**
    eval 集 A07–A19  —— 训练集里没见过，**未知攻击**

所以：
    dev EER  ≈ 已知攻击上的成绩（拟合能力）
    eval EER ≈ 未知攻击上的成绩（**泛化能力，比赛看的就是这个**）
两者差距就是"泛化鸿沟"，是答辩时最值钱的一张表。

产出：
    evaluation/reports/attack_breakdown_<split>_<时间戳>.md   人读的报告
    evaluation/reports/attack_breakdown_<split>_<时间戳>.json  机器读的数据
    --save-scores 可把每条样本的分数落盘，后续统计不必重跑推理。

用法：
    # dev 集全量细分（约 25k 条，GPU 上几分钟）
    python evaluation/attack_breakdown_eval.py --split dev

    # eval 集（约 71k 条，慢，建议先 --limit 2000 冒烟）
    python evaluation/attack_breakdown_eval.py --split eval --limit 2000

    # 指定权重 / 走 CPU / 复用已算好的分数
    python evaluation/attack_breakdown_eval.py --ckpt <path/to/best.pth>
    python evaluation/attack_breakdown_eval.py --device cpu
    python evaluation/attack_breakdown_eval.py --split dev --scores scores.jsonl

依赖：torch + aasist(external/)。指标计算不依赖 external（见 metrics.py）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_EVAL_DIR))

from paths import ASVSPOOF_LA, AASIST_DIR, EXP_RESULT_DIR, DEVICE  # noqa: E402
from metrics import compute_eer, far_frr_at, summarize  # noqa: E402

# 已知 / 未知攻击划分（ASVspoof2019 LA 官方设定）
KNOWN_ATTACKS = {f"A{i:02d}" for i in range(1, 7)}      # A01–A06
UNKNOWN_ATTACKS = {f"A{i:02d}" for i in range(7, 20)}   # A07–A19

# 合格线：通道①单独达标线 1.5%，未知攻击泛化放宽到 5%（AASIST 官方基线约 4~5%）
PASS_EER = 1.5
PASS_EER_UNKNOWN = 5.0


# ------------------------------------------------------------------ #
# 协议解析
# ------------------------------------------------------------------ #
def load_protocol(split: str) -> list[tuple[str, str, str]]:
    """返回 [(utt_id, system_id, label)]，label ∈ {bonafide, spoof}。

    protocol 每行 5 列：<speaker_id> <utt_id> - <system_id> <label>
    bonafide 行的 system_id 占位为 "-"。
    """
    name = f"ASVspoof2019.LA.cm.{'dev' if split == 'dev' else 'eval'}.trl.txt"
    path = Path(ASVSPOOF_LA) / "ASVspoof2019_LA_cm_protocols" / name
    if not path.is_file():
        raise FileNotFoundError(f"找不到协议文件: {path}")

    rows: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 5:
                continue
            rows.append((parts[1], parts[3], parts[4]))
    return rows


# ------------------------------------------------------------------ #
# 推理
# ------------------------------------------------------------------ #
def _latest_exp() -> str:
    subs = [d for d in Path(EXP_RESULT_DIR).iterdir() if d.is_dir()]
    if not subs:
        raise FileNotFoundError(f"无实验目录: {EXP_RESULT_DIR}")
    return str(max(subs, key=lambda d: d.stat().st_mtime))


def resolve_ckpt(exp_dir: str, explicit: str | None = None) -> str:
    """权重优先级：best.pth > swa.pth > epoch_{N}_{EER}.pth 中 EER 最小 > epoch_{N}.pth"""
    if explicit:
        return explicit
    wdir = Path(exp_dir) / "weights"
    for name in ("best.pth", "swa.pth"):
        p = wdir / name
        if p.exists():
            return str(p)
    cand = list(wdir.glob("epoch_*_*.pth"))
    if cand:
        return str(min(cand, key=lambda p: float(re.findall(r"_([\d.]+)\.pth", p.name)[0])))
    cand = list(wdir.glob("epoch_*.pth"))
    if cand:
        return str(max(cand, key=lambda p: (int(re.findall(r"epoch_(\d+)\.pth", p.name)[0]),
                                            p.stat().st_mtime)))
    raise FileNotFoundError(f"实验目录无可用权重: {wdir}")


def infer_scores(protocol_rows, exp_dir, ckpt, split, device, batch_size,
                 limit=None, use_amp=True):
    """跑 AASIST 推理，返回 {utt_id: 伪造概率}。

    只依赖 external/aasist 的 data_utils 与 models，指标计算走本地 metrics.py。
    """
    import torch
    from torch.utils.data import DataLoader

    if str(AASIST_DIR) not in sys.path:
        sys.path.insert(0, str(AASIST_DIR))
    from data_utils import Dataset_ASVspoof2019_devNeval  # noqa: E402

    cfg_path = Path(exp_dir) / "config.conf"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"缺少实验配置: {cfg_path}")
    mcfg = json.loads(cfg_path.read_text(encoding="utf-8"))["model_config"]

    arch = mcfg["architecture"]
    module = __import__("models." + arch, fromlist=["Model"])
    model = getattr(module, "Model")(mcfg).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    keys = [u for u, _, _ in protocol_rows]
    if limit:
        keys = keys[:limit]
        protocol_rows = protocol_rows[:limit]

    base = Path(ASVSPOOF_LA) / f"ASVspoof2019_LA_{'dev' if split == 'dev' else 'eval'}"
    cut = int(mcfg.get("nb_samp", 64600))
    ds = Dataset_ASVspoof2019_devNeval(list_IDs=keys, base_dir=base, cut=cut)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    scores: dict[str, float] = {}
    t0 = time.time()
    total = len(loader)
    with torch.no_grad():
        for i, (batch_x, utt_ids) in enumerate(loader, 1):
            batch_x = batch_x.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, out = model(batch_x)
            # AASIST 标签约定 bonafide=1 / spoof=0，故第 0 列才是伪造概率
            prob = torch.softmax(out.float(), dim=-1)[:, 0].cpu().numpy()
            for s, u in zip(prob, utt_ids):
                scores[u] = float(s)
            if i % 200 == 0 or i == total:
                el = time.time() - t0
                eta = el / i * (total - i)
                print(f"  [{split}] {i}/{total}  {el:.0f}s  ETA {eta:.0f}s", flush=True)
    return scores


# ------------------------------------------------------------------ #
# 分数落盘 / 复用
# ------------------------------------------------------------------ #
def save_scores(scores: dict[str, float], path: str, protocol_rows):
    """落盘为 jsonl：每行 {utt_id, label, system_id, score}"""
    meta = {u: (sid, lab) for u, sid, lab in protocol_rows}
    with open(path, "w", encoding="utf-8") as f:
        for u, s in scores.items():
            sid, lab = meta.get(u, ("-", "unknown"))
            f.write(json.dumps({"utt_id": u, "label": lab,
                                "system_id": sid, "score": s}) + "\n")
    print(f"[score] 已保存 {len(scores)} 条 -> {path}")


def load_scores(path: str) -> tuple[dict[str, float], list[tuple[str, str, str]]]:
    """从 jsonl 读回分数，返回 (scores, protocol_rows)"""
    scores, rows = {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scores[d["utt_id"]] = float(d["score"])
            rows.append((d["utt_id"], d.get("system_id", "-"), d.get("label", "unknown")))
    return scores, rows


# ------------------------------------------------------------------ #
# 分组统计
# ------------------------------------------------------------------ #
def breakdown(scores: dict[str, float], protocol_rows) -> dict:
    """按攻击算法分组算 EER；每组用「该组 spoof」vs「全体 bonafide」。"""
    bon_all: list[float] = []
    per_attack: dict[str, list[float]] = {}
    missing = 0

    for u, sid, lab in protocol_rows:
        s = scores.get(u)
        if s is None:
            missing += 1
            continue
        if lab == "bonafide":
            bon_all.append(s)
        else:
            per_attack.setdefault(sid, []).append(s)

    if not bon_all:
        raise ValueError("没有 bonafide 样本，无法计算 EER")

    overall = summarize(bon_all, [s for v in per_attack.values() for s in v])

    rows = []
    for sid in sorted(per_attack):
        spf = per_attack[sid]
        eer, thr = compute_eer(bon_all, spf)
        far, frr = far_frr_at(bon_all, spf, thr)
        rows.append({
            "attack": sid,
            "n_spoof": len(spf),
            "eer_pct": round(eer, 4),
            "threshold": round(thr, 6),
            "far_pct": round(far, 4),
            "frr_pct": round(frr, 4),
            "spoof_mean": round(float(np.mean(spf)), 5),
            "domain": "已知" if sid in KNOWN_ATTACKS else (
                "未知" if sid in UNKNOWN_ATTACKS else "未标注"),
        })

    def _group(ids: set[str]) -> dict | None:
        spf = [s for k, v in per_attack.items() if k in ids for s in v]
        if not spf:
            return None
        eer, thr = compute_eer(bon_all, spf)
        return {"eer_pct": round(eer, 4), "threshold": round(thr, 6),
                "n_spoof": len(spf), "n_attacks": len(
                    [k for k in per_attack if k in ids])}

    return {
        "overall": overall,
        "per_attack": rows,
        "known": _group(KNOWN_ATTACKS),
        "unknown": _group(UNKNOWN_ATTACKS),
        "n_missing_scores": missing,
    }


def render_markdown(result: dict, meta: dict) -> str:
    """生成人读的 Markdown 报告。"""
    ov = result["overall"]
    lines = [
        "# 谛听 VeriCall · 通道① 跨攻击域细分评测报告",
        "",
        f"- 生成时间：{meta['timestamp']}",
        f"- 数据集：ASVspoof2019 LA / `{meta['split']}` 集",
        f"- 权重：`{meta['ckpt']}`",
        f"- 设备：{meta['device']}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 样本数 | {ov['n_bonafide']} bonafide / {ov['n_spoof']} spoof |",
        f"| **EER** | **{ov['eer_pct']:.3f}%** |",
        f"| EER 处阈值 | {ov['eer_threshold']:.4f} |",
        f"| EER 处误拦率 FAR | {ov['far_at_eer_pct']:.3f}% |",
        f"| EER 处漏拦率 FRR | {ov['frr_at_eer_pct']:.3f}% |",
        f"| bonafide 分数均值 / p95 | {ov['bonafide_mean']:.4f} / {ov['bonafide_p95']:.4f} |",
        f"| spoof 分数均值 / p5 | {ov['spoof_mean']:.4f} / {ov['spoof_p5']:.4f} |",
        "",
    ]

    if result["known"] and result["unknown"]:
        k, u = result["known"], result["unknown"]
        gap = u["eer_pct"] - k["eer_pct"]
        lines += [
            "## 已知攻击 vs 未知攻击（泛化鸿沟）",
            "",
            "| 域 | 攻击算法 | 样本数 | EER |",
            "|---|---|---|---|",
            f"| 已知 A01–A06 | {k['n_attacks']} 类 | {k['n_spoof']} | {k['eer_pct']:.3f}% |",
            f"| 未知 A07–A19 | {u['n_attacks']} 类 | {u['n_spoof']} | {u['eer_pct']:.3f}% |",
            "",
            f"**泛化鸿沟 = {gap:+.3f} 个百分点**"
            f"（{k['eer_pct']:.3f}% → {u['eer_pct']:.3f}%）",
            "",
        ]

    lines += [
        "## 按攻击算法细分",
        "",
        "| 攻击 | 域 | 样本数 | EER | 阈值 | FAR | FRR | 分数均值 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["per_attack"]:
        lines.append(
            f"| {r['attack']} | {r['domain']} | {r['n_spoof']} | {r['eer_pct']:.3f}% "
            f"| {r['threshold']:.4f} | {r['far_pct']:.3f}% | {r['frr_pct']:.3f}% "
            f"| {r['spoof_mean']:.4f} |")

    worst = max(result["per_attack"], key=lambda r: r["eer_pct"]) if result["per_attack"] else None
    if worst:
        lines += [
            "",
            "## 判定",
            "",
            f"- 最弱环节：**{worst['attack']}**，EER {worst['eer_pct']:.3f}%（{worst['domain']}攻击）",
            f"- 通道① 合格线：{PASS_EER}%（已知）/ {PASS_EER_UNKNOWN}%（未知）",
        ]
        ref = result["unknown"] or result["known"]
        if ref:
            thr_line = PASS_EER_UNKNOWN if result["unknown"] else PASS_EER
            ok = "达标" if ref["eer_pct"] <= thr_line else f"未达标（超出 {ref['eer_pct'] - thr_line:.3f} 个百分点）"
            lines.append(f"- 当前成绩：**{ok}**")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="通道① 跨攻击域细分评测")
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--exp", default=None, help="实验目录，默认取最新")
    ap.add_argument("--ckpt", default=None, help="指定权重，优先于自动挑选")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="只测前 N 条（冒烟）")
    ap.add_argument("--device", default=DEVICE, choices=["cuda", "cpu"])
    ap.add_argument("--scores", default=None, help="复用已算好的分数 jsonl，跳过推理")
    ap.add_argument("--save-scores", default=None, help="把分数落盘为 jsonl")
    ap.add_argument("--out-dir", default=str(_EVAL_DIR / "reports"))
    args = ap.parse_args()

    print("=" * 64)
    print(f"谛听 VeriCall · 跨攻击域细分评测  split={args.split}")
    print("=" * 64)

    protocol_rows = load_protocol(args.split)
    print(f"协议载入：{len(protocol_rows)} 条")

    if args.scores:
        scores, protocol_rows = load_scores(args.scores)
        print(f"复用分数：{len(scores)} 条 <- {args.scores}")
        meta_ckpt = args.scores
    else:
        exp = args.exp or _latest_exp()
        ckpt = resolve_ckpt(exp, args.ckpt)
        device = args.device
        if device == "cuda":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"实验目录：{exp}")
        print(f"权重    ：{os.path.basename(ckpt)}")
        print(f"设备    ：{device}")
        meta_ckpt = os.path.basename(ckpt)
        scores = infer_scores(protocol_rows, exp, ckpt, args.split, device,
                              args.batch_size, args.limit,
                              use_amp=(device == "cuda"))
        if args.save_scores:
            save_scores(scores, args.save_scores, protocol_rows)

    result = breakdown(scores, protocol_rows)
    ov = result["overall"]
    print("-" * 64)
    print(f"总体 EER = {ov['eer_pct']:.3f}%   阈值 {ov['eer_threshold']:.4f}")
    for r in result["per_attack"]:
        print(f"  {r['attack']} [{r['domain']}] n={r['n_spoof']:>6}  EER={r['eer_pct']:7.3f}%")
    if result["known"] and result["unknown"]:
        print(f"  已知攻击 EER = {result['known']['eer_pct']:.3f}%")
        print(f"  未知攻击 EER = {result['unknown']['eer_pct']:.3f}%")
        print(f"  泛化鸿沟     = {result['unknown']['eer_pct'] - result['known']['eer_pct']:+.3f} pp")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split": args.split, "ckpt": meta_ckpt, "device": args.device}
    json_path = out_dir / f"attack_breakdown_{args.split}_{ts}.json"
    md_path = out_dir / f"attack_breakdown_{args.split}_{ts}.md"
    json_path.write_text(json.dumps({"meta": meta, "result": result},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result, meta), encoding="utf-8")
    print("-" * 64)
    print(f"报告已生成：\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()
