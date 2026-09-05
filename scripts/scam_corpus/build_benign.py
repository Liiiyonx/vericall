# -*- coding: utf-8 -*-
"""build_benign.py — 正常家庭通话负样本库构建（话术库 v0.1 配套）

与 build_corpus.py 的关系：共用 LLM 后端 / 去重 / 清洗基建，差异在：
  - label=benign，不分话术阶段（stage="na"）；
  - 自洽过滤**反向**：risk 高的丢弃（>0.5 说明模型把正常通话也判险，
    这类样本要么是坏样本、要么是金子级难负样本——near_boundary 类保留并强制抽检）；
  - 输出 data/scam_corpus/benign_corpus.jsonl（规范见 docs/话术库标注规范.md §二）。

用法：
  python build_benign.py --per-category 10
  python build_benign.py --append --per-category 50
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benign_seeds import (BENIGN_AXES, BENIGN_CATEGORIES, BENIGN_PROMPT,  # noqa: E402
                          BENIGN_SEEDS, NEAR_BOUNDARY)
from build_corpus import Deduper, clean_text, pick_backend  # noqa: E402

OUT_DIR = ROOT / "data" / "scam_corpus"
BENIGN_PATH = OUT_DIR / "benign_corpus.jsonl"
REVIEW_PATH = OUT_DIR / "benign_review_sample.csv"


def make_benign_seed(category: str, rng: random.Random) -> str:
    seed = rng.choice(BENIGN_SEEDS[category])
    for m in set(re.findall(r"\{(\w+)\}", seed)):
        if m in BENIGN_AXES:
            seed = seed.replace("{" + m + "}", rng.choice(BENIGN_AXES[m]))
    return seed


def load_existing(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # 2026-09-05 加Ff1a全局 socket 超时Ff0c防止 urllib 半开连接无限挂起
    import socket as _socket
    _socket.setdefaulttimeout(180)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["auto", "cloud", "ollama"], default="auto")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    backend_name, call_llm = pick_backend(args.backend)
    cats = list(BENIGN_CATEGORIES)
    total = args.per_category * len(cats)
    print(f"后端={backend_name}，{len(cats)} 类 × {args.per_category} = 计划 {total} 条（负样本）")

    existing = load_existing(BENIGN_PATH) if args.append else []
    dedup = Deduper()
    for rec in existing:
        dedup.add(rec["text"])
    # 与诈骗库交叉去重（防同一句既正又负）
    scam_path = OUT_DIR / "corpus_v0.1.jsonl"
    if scam_path.exists():
        for rec in load_existing(scam_path):
            dedup.add(rec["text"])

    ch = None
    if not args.no_filter:
        from fusion.semantic_channel import SemanticChannel
        ch = SemanticChannel()

    out, dropped_dup, dropped_len, dropped_risk, failed = [], 0, 0, 0, 0
    today = dt.date.today().isoformat()

    for i in range(args.per_category):
        for cat in cats:
            seed = make_benign_seed(cat, rng)
            try:
                text = clean_text(call_llm(BENIGN_PROMPT.format(seed=seed)))
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  [fail] {type(e).__name__}: {e}")
                continue
            if not (10 <= len(text) <= 300):
                dropped_len += 1
                continue
            if dedup.is_dup(text):
                dropped_dup += 1
                continue

            near = cat in NEAR_BOUNDARY
            rec = {
                "id": f"BN-{len(existing) + len(out) + 1:06d}",
                "category": cat,
                "stage": "na",
                "risk_elements": [],
                "near_boundary": near,
                "label": "benign",
                "text": text,
                "dialect_style": "mandarin-日常",
                "source": f"{backend_name}",
                "seed_skeleton": seed,
                "created": today,
                # 近边界样本全部进人工抽检；其余按 20%
                "review": {"sampled": near or rng.random() < 0.2, "verdict": None},
            }
            if ch is not None:
                r = ch.analyze(text)
                rec["self_check_risk"] = r.risk
                if r.risk > 0.5 and not near:
                    dropped_risk += 1
                    print(f"  [drop] risk={r.risk}（正常样本判险，丢弃） {text[:30]}...")
                    continue
                if r.risk > 0.5 and near:
                    rec["review"]["note"] = "模型判险的近边界样本，必审"

            dedup.add(text)
            out.append(rec)
            print(f"  [ok {rec['id']}] {cat}{'(近边界)' if near else ''} {text[:36]}...")

    print(f"\n生成 {len(out)} | 近重复丢 {dropped_dup} | 长度丢 {dropped_len} "
          f"| 反检丢 {dropped_risk} | 失败 {failed}")
    if args.dry_run:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_recs = existing + out
    with open(BENIGN_PATH, "w", encoding="utf-8") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(REVIEW_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "near_boundary", "self_check_risk", "text", "verdict"])
        for rec in all_recs:
            if rec["review"]["sampled"]:
                w.writerow([rec["id"], rec["category"], rec["near_boundary"],
                            rec.get("self_check_risk", ""), rec["text"], ""])
    print(f"负样本 -> {BENIGN_PATH}（累计 {len(all_recs)}）\n抽检 -> {REVIEW_PATH}")


if __name__ == "__main__":
    main()
