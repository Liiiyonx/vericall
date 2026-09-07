# -*- coding: utf-8 -*-
"""apply_review_verdicts.py — 语料 v0.2 定稿应用器（M1 人工 + auto 二审自动）

输入三份（按 id 合并，worksheet 优先）：
1. evaluation/review_candidates_worksheet.csv  —— M1 人工复核 31 行（human_verdict）
2. evaluation/review_prefilter_suspect.csv      —— 一审可疑清单（含 verdict）
3. evaluation/review_drop2_confirm.csv          —— drop 区二审确认（confirm=drop|keep）

最终判定：
- worksheet 中的 id → 以 human_verdict 为准（drop 删；pass/fix 标 review），未填齐拒绝 --apply；
- 不在 worksheet 的 auto 行（v0.2 新增可疑）：
    · 一审 fix              → review.verdict='fix'（标记保留，待改写回填）
    · 一审 drop + 二审 drop → 删除（两轮确认，高置信）
    · 一审 drop + 二审 keep → review.verdict='pass' + note='drop2_keep'（一审误杀，二审纠回）

背景（2026-09-07）：v0.2 全量预筛一审对铺垫段（opening/buildup）存在系统性误杀，
88 条 auto-drop 中二审仅确认 16 条真该删（见 review_drop2_confirm.csv），故 auto 处置
必须叠加二审，禁止只凭一审 verdict 删条。

用法：
  python apply_review_verdicts.py             # dry-run：只打印计划
  python apply_review_verdicts.py --apply     # 正式落地（重写 jsonl + stats）

安全约束：
  - worksheet 未填 verdict（≥1 条）→ 拒绝 --apply，避免半成品定稿；
  - 全文件重写，**必须在 v0.2 扩量任务结束后运行**（build_corpus 轮次结束会整写 corpus）；
  - 只改动上述清单涉及的 id，其余记录原样保留。
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
SUS = ROOT / "evaluation" / "review_prefilter_suspect.csv"
D2 = ROOT / "evaluation" / "review_drop2_confirm.csv"
CORPUS = ROOT / "data" / "scam_corpus" / "corpus_v0.1.jsonl"
STATS = ROOT / "data" / "scam_corpus" / "stats.json"
VALID = {"pass", "fix", "drop"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="正式落地（默认 dry-run）")
    ap.add_argument("--llm-fill", action="store_true",
                    help="M1 兜底：未填 human_verdict 的 worksheet 行改按 LLM 结论链裁决"
                         "（计划书风险条款：人工拖过时限则按预筛结论定稿）")
    args = ap.parse_args()

    ws_rows = read_csv(WS)
    human = {r["id"]: (r.get("human_verdict") or "").strip().lower() for r in ws_rows}
    notes = {r["id"]: (r.get("human_note") or "").strip() for r in ws_rows}
    unknown = {v for v in human.values() if v and v not in VALID}
    if unknown:
        print(f"[错误] worksheet 含非法 verdict: {sorted(unknown)}（仅允许 pass/fix/drop）")
        return 1
    pending_ws = sorted(i for i, v in human.items() if not v)
    print(f"[M1 人工] worksheet {len(ws_rows)} 行；已填 {sum(1 for v in human.values() if v)}；未填 {len(pending_ws)}"
          + ("；--llm-fill 兜底开启" if args.llm_fill else ""))

    sus = {r["id"]: r["verdict"].strip().lower() for r in read_csv(SUS)}
    d2 = {r["id"]: r["confirm"].strip().lower() for r in read_csv(D2)}
    print(f"[一审可疑] {len(sus)} 条（drop 二审参考 {len(d2)} 条）")

    # 合并终局判定：id -> (action, source, note)
    # worksheet 行（含未填）默认排除出 auto——保留给人工裁决，未填只算 pending；
    # --llm-fill 时未填行按 LLM 结论链兜底（sus 一审 + drop2 二审 + 全量重判 pass）
    final: dict[str, tuple[str, str, str]] = {}
    ws_ids = set(human.keys())
    for i, v in human.items():
        if v:
            final[i] = (v, "human", notes.get(i, ""))
    auto = {i: v for i, v in sus.items() if i not in ws_ids}
    for i, v in auto.items():
        if v == "fix":
            final[i] = ("fix", "auto_fix", "一审fix，标记保留待改写")
        elif v == "drop":
            if d2.get(i) == "drop":
                final[i] = ("drop", "auto_drop2", "一审drop+二审drop，两轮确认")
            else:
                final[i] = ("pass", "auto_drop2_keep", "一审drop但二审keep，误杀纠回")
    if args.llm_fill:
        for i in ws_ids:
            if i in final:
                continue  # 已人工填
            v_sus = sus.get(i)
            if v_sus == "fix":
                final[i] = ("fix", "llm_fill", "兜底：一审fix，标记保留待改写")
            elif v_sus == "drop":
                if d2.get(i) == "drop":
                    final[i] = ("drop", "llm_fill_drop2", "兜底：一审drop+二审drop，两轮确认")
                else:
                    final[i] = ("pass", "llm_fill_keep", "兜底：一审drop但二审keep，误杀纠回")
            else:
                final[i] = ("pass", "llm_fill_repass", "兜底：09-07 全量预筛判 pass")
    n_human = sum(1 for _, s, _ in final.values() if s == "human")
    n_auto = sum(1 for _, s, _ in final.values() if s.startswith("auto"))
    n_fill = sum(1 for _, s, _ in final.values() if s.startswith("llm_fill"))
    print(f"[终局] {len(final)} 条 = human {n_human} + auto {n_auto}"
          + (f" + llm_fill {n_fill}" if args.llm_fill else "")
          + (f"；另有 {len(ws_ids) - n_human - n_fill} 行 worksheet 待人工（不参与 auto）"
             if not args.llm_fill else ""))

    recs = load_corpus(CORPUS)
    by_id = {r["id"]: r for r in recs}
    print(f"[语料] 当前 {len(recs)} 条；id 唯一性 OK: {len(by_id) == len(recs)}")

    missing = [i for i, (v, s, _) in final.items() if i not in by_id]
    if missing:
        print(f"[告警] {len(missing)} 条清单 id 在语料中找不到: {missing[:10]}")

    act = Counter(s for _, s, _ in final.values())
    vcnt = Counter(v for v, _, _ in final.values())
    print(f"[计划] pass={vcnt['pass']} fix={vcnt['fix']} drop={vcnt['drop']} | 来源 {dict(act)}")
    pending_ids = sorted(i for i in ws_ids if i not in final)
    for i in pending_ids:
        print(f"  ⏳ {i} 待人工裁决（worksheet 未填，不参与 auto）")
    for i in sorted(final):
        v, s, note = final[i]
        if s == "human" or v in ("drop", "fix"):
            print(f"  - {i} -> {v}  [{s}]  {note[:40]}{'  !!MISSING' if i not in by_id else ''}")

    if not args.apply:
        print("[dry-run] 未改动任何文件。加 --apply 落地。")
        return 0
    if pending_ws and not args.llm_fill:
        print(f"[中止] worksheet 还有 {len(pending_ws)} 行未填 verdict（未开 --llm-fill），拒绝 --apply。")
        return 2

    n_drop = 0
    keep = []
    for r in recs:
        v, s, note = final.get(r["id"], (None, None, None))
        if v == "drop":
            n_drop += 1
            continue
        if v in ("pass", "fix"):
            rv = r.setdefault("review", {})
            rv["verdict"] = v
            rv["source"] = s
            if note:
                rv["note"] = note
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
        "v0.2_finalized": "2026-09-07",
        "applied": {"pass": vcnt["pass"], "fix": vcnt["fix"], "drop": vcnt["drop"],
                    "dropped_actually": n_drop, "missing": len(missing),
                    "by_source": dict(act)},
    }
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[已应用] 保留 {len(keep)}（删 {n_drop}）；stats.json 已更新。"
          f"fix 标 {vcnt['fix']} 条待文本改写回填。")


if __name__ == "__main__":
    sys.exit(main())
