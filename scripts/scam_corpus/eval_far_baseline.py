#!/usr/bin/env python
"""通道③ FAR 基线评估（2026-09-05 新增）

用 SemanticChannel（Ollama deepseek-r1:8b 风险打分）在正负样本库上跑 FAR/召回曲线：
  - 负样本（529，含近边界 legit_money/legit_service）应低风险  -> FAR = 负样本被判诈骗占比
  - 正样本（抽 N 条诈骗话术）应高风险                        -> 召回 = 正样本被判诈骗占比
  - 输出工作点建议：家庭场景「误拒代价 >> 误受」，选低 FAR 阈值。

用法：
  python eval_far_baseline.py [--scam-n 529] [--seed 42] [--limit 0] [--backend ollama|cloud]
  --limit>0 表示仅跑前 N 条（冒烟/自检用）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # 仓库根
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import socket as _socket
_socket.setdefaulttimeout(300)  # 防半开连接挂起

DATA = ROOT / "data" / "scam_corpus"
BENIGN_PATH = DATA / "benign_corpus.jsonl"
CORPUS_PATH = DATA / "corpus_v0.1.jsonl"
OUT_MD = ROOT / "evaluation" / "far_baseline.md"
OUT_JSON = ROOT / "evaluation" / "far_baseline.json"


def load(path: Path):
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scam-n", type=int, default=529, help="正样本抽样数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help=">0 时仅跑前 N 条（冒烟）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from fusion.semantic_channel import SemanticChannel

    ch = SemanticChannel(asr=None)  # 纯文本打分，不走 ASR

    benign = load(BENIGN_PATH)
    scam_all = load(CORPUS_PATH)
    scam = rng.sample(scam_all, min(args.scam_n, len(scam_all)))
    near = [r for r in benign if "近边界" in str(r.get("category", ""))]

    total = len(benign) + len(scam)
    if args.limit:
        benign = benign[: args.limit]
        scam = scam[: args.limit]
        near = [r for r in benign if "近边界" in str(r.get("category", ""))]

    print(f"样本: benign={len(benign)} (近边界 {len(near)}) scam={len(scam)} limit={args.limit}")

    def score_one(rec, tag):
        try:
            t0 = time.time()
            r = ch.analyze(rec.get("text", ""))
            rec["_risk"] = r.risk
            rec["_latency"] = round(time.time() - t0, 2)
            print(f"  [{tag} {rec.get('id','?')}] risk={r.risk:.2f} lat={rec['_latency']}s")
            return True
        except Exception as e:  # noqa: BLE001
            rec["_risk"] = None
            print(f"  [fail {tag} {rec.get('id','?')}] {type(e).__name__}: {e}")
            return False

    n_ok = 0
    for rec in benign:
        n_ok += 1 if score_one(rec, "B") else 0
    for rec in scam:
        n_ok += 1 if score_one(rec, "S") else 0
    print(f"打分完成: {n_ok}/{len(benign)+len(scam)}")

    # ---------- 指标 ----------
    def risks(recs):
        return [r["_risk"] for r in recs if r.get("_risk") is not None]

    b_risks, s_risks = risks(benign), risks(scam)
    nb_risks = [r["_risk"] for r in benign
                if r.get("_risk") is not None and "近边界" in str(r.get("category", ""))]
    n_all = len(b_risks)
    if not n_all:
        print("无有效打分，退出")
        return

    rows = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        far = sum(1 for x in b_risks if x >= thr) / max(1, len(b_risks))
        far_nb = sum(1 for x in nb_risks if x >= thr) / max(1, len(nb_risks))
        rec = sum(1 for x in s_risks if x >= thr) / max(1, len(s_risks))
        rows.append({"threshold": thr, "FAR_all": round(far, 4),
                     "FAR_near_boundary": round(far_nb, 4), "recall_scam": round(rec, 4)})
        print(f"thr={thr} FAR={far:.3f} FAR_nb={far_nb:.3f} recall={rec:.3f}")

    row05 = next(r for r in rows if r["threshold"] == 0.5)
    avg_lat = sum(r.get("_latency", 0) for r in benign) / max(1, len(benign))

    md = [
        "# 通道③ FAR 基线报告",
        "",
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M')} · SemanticChannel(Ollama) · seed={args.seed}",
        f"> 样本：负样本 {len(b_risks)} 条（近边界 {len(nb_risks)}）+ 正样本抽样 {len(s_risks)} 条 · 单条均延迟 {avg_lat:.1f}s",
        "",
        "## FAR / 召回扫描（阈值 = 风险分卡点）",
        "",
        "| 阈值 | FAR(全负样本) | FAR(近边界) | 召回(正样本) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['threshold']} | {r['FAR_all']:.1%} | {r['FAR_near_boundary']:.1%} | {r['recall_scam']:.1%} |")
    md += [
        "",
        f"## 0.5 工作点：FAR={row05['FAR_all']:.2%}（近边界 {row05['FAR_near_boundary']:.2%}），召回={row05['recall_scam']:.2%}",
        "",
        "> 注：家庭场景「误拒代价>>误受」→ 正式部署建议取 FAR 最低且召回可接受的阈值，",
        "> 数值以上表为准；本报告同时服务《参赛优化任务书》评测证据链。",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_benign": len(b_risks), "n_near": len(nb_risks), "n_scam": len(s_risks),
               "avg_latency_s": round(avg_lat, 2), "rows": rows,
               "wp_0_5": row05}
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {OUT_MD}")


if __name__ == "__main__":
    main()
