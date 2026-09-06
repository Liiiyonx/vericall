# -*- coding: utf-8 -*-
"""apply_review_verdicts.py — 语料 v0.1 定稿应用器（M1 复核结果落地）

读取 evaluation/review_candidates_worksheet.csv 的 human_verdict/human_note
（pass / fix / drop），按 id 套用到 data/scam_corpus/corpus_v0.1.jsonl：
  - drop → 移除该条记录
  - pass → review.verdict='pass'（附 note）
  - fix  → review.verdict='fix'（附 note；文本待改写，作为 v0.1 定稿后修补项）
应用后重算 stats.json（by_category / by_stage / total）。

用法：
  python apply_review_verdicts.py                     # dry-run：只打印计划
  python apply_review_verdicts.py --apply             # 正式落地（重写 jsonl + stats）

安全约束：
  - 未填 verdict 的行（≥1 条）→ 拒绝 --apply，避免半成品定稿；
  - 全文件重写，**必须在 v0.2 扩量任务结束后运行**（build_corpus 轮次结束会整写
    corpus，覆盖本脚本改动），调用方自检；
  - 只改动 worksheet 涉及的 id，其余记录原样保留。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "evaluation" / "review_candidates_worksheet.csv"
CORPUS = ROOT / "data" / "scam_corpus" / "corpus_v0.1.jsonl"
STATS = ROOT / "data" / "scam_corpus" / "stats.json"
VALID = {"pass", "fix", "drop"}


def load_rows() -> list[dict]:
    with open(WS, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="正式落地（默认 dry-run）")
    args = ap.parse_args()

    rows = load_rows()
    verdicts = {r["id"]: (r.get("human_verdict") or "").strip().lower()
                for r in rows}
    notes = {r["id"]: (r.get("human_note") or "").strip() for r in rows}
    unknown = {v for v in verdicts.values() if v and v not in VALID}
    if unknown:
        print(f"[错误] 含非法 verdict 值: {sorted(unknown)}（仅允许 pass/fix/drop）")
        return 1

    pending = {k for k, v in verdicts.items() if not v}
    print(f"[复核单] 共 {len(rows)} 行；已填 verdict: {sum(1 for v in verdicts.values() if v)}；"
          f"未填: {len(pending)}")

    recs = load_corpus(CORPUS)
    by_id = {r["id"]: r for r in recs}
    print(f"[语料] 当前 {len(recs)} 条；去重后 id 唯一性 OK: {len(by_id) == len(recs)}")

    missing = [i for i in verdicts if i and verdicts[i] and i not in by_id]
    if missing:
        print(f"[告警] {len(missing)} 条 verdict 在语料中找不到对应 id: {missing[:10]}")

    cnt = Counter()
    plans = []
    for rid in sorted(verdicts):
        v, note = verdicts[rid], notes[rid]
        if not v:
            continue
        cnt[v] += 1
        plans.append((rid, v, by_id.get(rid) is not None,
                      f"note={note[:30]}" if note else ""))
    print(f"[计划] pass={cnt['pass']} fix={cnt['fix']} drop={cnt['drop']} "
          f"(找不到: {sum(1 for _, _, ok, _ in plans if not ok)})")
    for rid, v, ok, note in plans:
        if not ok or v in ("fix", "drop"):
            print(f"  - {rid} -> {v}  {'OK' if ok else '!!MISSING'}  {note}")

    if not args.apply:
        print("[dry-run] 未改动任何文件。加 --apply 落地。")
        return 0

    if pending:
        print(f"[中止] 还有 {len(pending)} 行未填 verdict，拒绝 --apply。")
        return 2

    n_drop = 0
    keep = []
    for r in recs:
        v = verdicts.get(r["id"])
        if v == "drop":
            n_drop += 1
            continue
        if v in ("pass", "fix"):
            rv = r.setdefault("review", {})
            rv["verdict"] = v
            if notes.get(r["id"]):
                rv["note"] = notes[r["id"]]
        keep.append(r)

    with open(CORPUS, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cats = sorted({r.get("category", "?") for r in keep})
    stages = sorted({r.get("stage", "?") for r in keep})
    stats = {
        "created": keep[0].get("created") if keep else None,
        "total": len(keep),
        "by_category": {c: sum(1 for r in keep if r.get("category") == c) for c in cats},
        "by_stage": {s: sum(1 for r in keep if r.get("stage") == s) for s in stages},
        "v0.1_finalized": "2026-09-06",
        "applied": {"pass": cnt["pass"], "fix": cnt["fix"], "drop": cnt["drop"],
                    "dropped_actually": n_drop, "missing": len(missing)},
    }
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[已应用] 保留 {len(keep)}（drop {n_drop}）；stats.json 已更新。"
          f"fix 行（{cnt['fix']}）待文本改写后回填。")


if __name__ == "__main__":
    sys.exit(main())
