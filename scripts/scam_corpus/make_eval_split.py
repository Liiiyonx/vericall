#!/usr/bin/env python
"""make_eval_split.py — 话术分类评测集切分（A1，2026-09-07 语料 v0.2 定稿后）

从定稿 corpus（11,934 scam / 2,404 benign）按类别分层切 80/20 留出：
  - scam 按 8 类诈骗 category 分层；
  - benign 按 6 类正常 category 分层（A4 多分类 F1 用：8 诈骗类 + 1 benign 合并类）；
只冻结 id 清单（不复制正文），正文从 corpus_v0.1.jsonl / benign_corpus.jsonl 按 id 现取，
避免双份数据漂移；seed 固定可复现。

产出：
  data/scam_corpus/eval_split/scam_train_ids.json / scam_eval_ids.json
  data/scam_corpus/eval_split/benign_train_ids.json / benign_eval_ids.json
  data/scam_corpus/eval_split/split_manifest.json（seed、条数、每类 train/eval 数、隔离校验）
用法：
  python -u scripts/scam_corpus/make_eval_split.py [--seed 2026] [--ratio 0.2]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "scam_corpus" / "corpus_v0.1.jsonl"
BENIGN = ROOT / "data" / "scam_corpus" / "benign_corpus.jsonl"
OUT_DIR = ROOT / "data" / "scam_corpus" / "eval_split"


def load_ids(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append({"id": r["id"], "category": r.get("category", "?"),
                            "stage": r.get("stage", "")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ratio", type=float, default=0.2, help="留出比例(默认0.2=20%)")
    args = ap.parse_args()

    scam = load_ids(CORPUS)
    benign = load_ids(BENIGN)
    print(f"[输入] scam {len(scam)}（8类） + benign {len(benign)}（6类）")

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "ratio": args.ratio, "note":
                "话术分类评测集(A1)：80/20 留出，冻结 id 清单，正文随语料版本现取",
                "sets": {}}

    for name, recs in (("scam", scam), ("benign", benign)):
        by_cat = defaultdict(list)
        for r in recs:
            by_cat[r["category"]].append(r)
        train_ids, eval_ids = [], []
        per_cat = {}
        for cat in sorted(by_cat):
            pool = by_cat[cat]
            rng.shuffle(pool)
            n_eval = max(1, round(len(pool) * args.ratio))
            ev, tr = pool[:n_eval], pool[n_eval:]
            eval_ids += [r["id"] for r in ev]
            train_ids += [r["id"] for r in tr]
            per_cat[cat] = {"total": len(pool), "train": len(tr), "eval": len(ev)}
        # 隔离校验
        assert len(set(train_ids)) == len(train_ids), "train 重复 id"
        assert len(set(eval_ids)) == len(eval_ids), "eval 重复 id"
        overlap = set(train_ids) & set(eval_ids)
        assert not overlap, f"{name} train∩eval 非空: {len(overlap)}"
        manifest["sets"][name] = {
            "total": len(recs), "train": len(train_ids), "eval": len(eval_ids),
            "isolation_ok": True, "by_category": per_cat}
        (OUT_DIR / f"{name}_train_ids.json").write_text(
            json.dumps(train_ids, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT_DIR / f"{name}_eval_ids.json").write_text(
            json.dumps(eval_ids, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{name}] train {len(train_ids)} / eval {len(eval_ids)}"
              f"（eval 占比 {len(eval_ids)/len(recs)*100:.1f}%）隔离 OK")

    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # 快速回显类别平衡
    for name in ("scam", "benign"):
        bc = manifest["sets"][name]["by_category"]
        print(f"\n[{name} 各类 eval 占比]")
        for cat, v in bc.items():
            print(f"  {cat}: {v['eval']}/{v['total']} ({v['eval']/v['total']*100:.0f}%)")
    print(f"\n产物目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
