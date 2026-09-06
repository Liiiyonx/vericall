#!/usr/bin/env python
"""pre_review.py — 语料抽检 LLM 预筛（2026-09-05 跟进会话新增）

对 review_sample.csv / benign_review_sample.csv 的每条样本用云端 LLM 打分，
输出"可疑候选"清单（verdict != pass 者），人工只需聚焦复核候选，
把 677 条全量抽检压缩到可人审规模。

用法（先 source 密钥）：
  bash: source D:/VeriCall_data/secrets/vericall_secrets.env
  python -u scripts/scam_corpus/pre_review.py [--limit N] [--kind scam|benign]
产出：
  evaluation/review_prefilter.csv     全部逐条 LLM 初判（含 reason）
  evaluation/review_prefilter_suspect.csv 仅可疑候选（人工复核清单）
  evaluation/review_prefilter_summary.md  统计概览
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAM_SAMPLE = ROOT / "data" / "scam_corpus" / "review_sample.csv"
BENIGN_SAMPLE = ROOT / "data" / "scam_corpus" / "benign_review_sample.csv"
OUT_CSV = ROOT / "evaluation" / "review_prefilter.csv"
OUT_SUS = ROOT / "evaluation" / "review_prefilter_suspect.csv"
OUT_MD = ROOT / "evaluation" / "review_prefilter_summary.md"

SCAM_PROMPT = """你是反诈语料质检员。审阅一条诈骗话术样本，判断它是否适合进入模型训练语料。
重要背景：话术库按「8 类诈骗 × 4 阶段(opening开场/buildup铺垫/ask索要/pressure施压)」网格构建，
每条只是完整诈骗链路中的一段，独立成条是设计使然，不要因「缺乏完整场景铺垫」而 drop。
检查点：
1) 诈骗性：是否指向其标注类别的典型诈骗动作（如冒充身份、索要转账/验证码/押金、高息诱导等）
2) 类别契合：内容与标注类别是否相符（如 category=冒充公检法但内容像中奖 = fix/drop）
3) 阶段契合：内容与 stage 所处环节是否相符（pressure 段可以是纯催促施压，只要催的是该类别诈骗动作）
4) 真实性：是否像真实电话诈骗口语（自然、有上下文），还是明显 AI 味/模板腔
5) 硬伤：自相矛盾、数字/话术荒谬穿帮、超长绕口等
真正的 drop 情形：内容与标注类别完全无关（如正常对话误标诈骗）、自相矛盾无法使用、明显是误生成的垃圾文本。
输出 JSON：{"verdict":"pass|fix|drop","reason":"≤40字中文"}
pass=可直接入库；fix=小修后可留（说明修点）；drop=该删（说明为何）。
"""

BENIGN_PROMPT = """你是反诈语料质检员。审阅一条日常正常通话文本（负样本），判断是否适合训练。
重要背景：负样本库含两种——普通日常对话，以及 near_boundary=true 的「近边界难例」
（看着像正常、但贴近诈骗话术风格/偶含转账字眼的故意设计样本，用于校准误报率）。
near_boundary=true 是刻意设计，**只要它整体仍是正常生活场景（亲属/客服/还款/缴费等），就应 pass**。
检查点：
1) 无害性：是否确为正常日常对话（亲属借钱、正常还款、客服回访、缴费提醒等都算正常）
2) 类别契合：内容与 category(legit_service/legit_money/life_reminder/festival/family_chat/health_care) 是否相符
3) 真实性：是否自然口语，而非生硬模板
4) 真正 drop 的情形：内容实际是诈骗（冒充公检法/中奖缴费/高息理财诱导等，仅含转账字眼的正常场景不算）
输出 JSON：{"verdict":"pass|fix|drop","reason":"≤40字中文"}
pass=可直接入库；fix=小修后可留；drop=实际是诈骗或明显异常该删。
"""


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return s


class CloudScorer:
    def __init__(self, timeout: int = 60):
        self.base = os.environ["SCAM_LLM_BASE"].rstrip("/")
        self.key = os.environ["SCAM_LLM_KEY"]
        self.model = os.environ.get("SCAM_LLM_MODEL", "deepseek-chat")
        self.timeout = timeout

    def review(self, system: str, text: str, max_retry: int = 3):
        last = None
        for _ in range(max_retry):
            try:
                payload = json.dumps({
                    "model": self.model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": text.strip()[:1500]},
                    ],
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base}/chat/completions", data=payload,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parsed = json.loads(_strip_fence(data["choices"][0]["message"]["content"]))
                v = str(parsed.get("verdict", "fix")).lower()
                if v not in ("pass", "fix", "drop"):
                    v = "fix"
                return {"verdict": v, "reason": str(parsed.get("reason", ""))[:60]}
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (_ + 1))
        return {"verdict": "fix", "reason": f"API失败:{type(last).__name__}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help=">0 仅跑前 N 条(冒烟)")
    ap.add_argument("--kind", choices=["scam", "benign", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    socket.setdefaulttimeout(60)
    scorer = CloudScorer()
    os.makedirs(OUT_CSV.parent, exist_ok=True)

    tasks = []  # (tag, row, system_prompt)
    if args.kind in ("scam", "both"):
        rows = list(csv.DictReader(open(SCAM_SAMPLE, encoding="utf-8-sig")))
        if args.limit:
            rows = rows[: args.limit]
        tasks += [("scam", r, SCAM_PROMPT) for r in rows]
    if args.kind in ("benign", "both"):
        rows = list(csv.DictReader(open(BENIGN_SAMPLE, encoding="utf-8-sig")))
        if args.limit:
            rows = rows[: args.limit]
        tasks += [("benign", r, BENIGN_PROMPT) for r in rows]

    print(f"待预筛: {len(tasks)} 条 (kind={args.kind})", flush=True)
    results, t0 = [], time.time()

    # 既有结果合并保护：若 OUT_CSV 已有数据且本次只跑单一 kind，
    # 先把旧结果读回，避免覆盖另一 kind 的历史预筛（2026-09-06 修复）。
    legacy = []  # [(kind,id,category,stage,near_boundary,verdict,reason,text)]
    if OUT_CSV.exists() and args.kind in ("scam", "benign"):
        try:
            with open(OUT_CSV, encoding="utf-8-sig", newline="") as f:
                rd = csv.reader(f)
                _h = next(rd, None)
                for row in rd:
                    if row:
                        legacy.append(row)
            print(f"  合并保护：读回既有结果 {len(legacy)} 行（仅覆盖 kind={args.kind} 的部分）", flush=True)
        except Exception:  # noqa: BLE001
            legacy = []

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "id", "category", "stage", "near_boundary",
                    "verdict", "reason", "text"])
        # 先写非本次 kind 的既有行（保留）
        for row in legacy:
            if len(row) >= 8 and row[0] != args.kind:
                w.writerow(row[:8])
        # 写本次 kind 既有行（本次重新跑，跳过；防止累积重复）
        run_ids = {r.get("id", "?") for tag, r, _ in tasks if tag == args.kind}
        for row in legacy:
            if len(row) >= 8 and row[0] == args.kind and row[1] not in run_ids:
                w.writerow(row[:8])
        for tag, r, sysp in tasks:
            text = r.get("text", "")
            # 附加元数据（near_boundary 等）帮模型正确判断难例
            meta_bits = []
            if r.get("near_boundary") and str(r["near_boundary"]).strip().lower() == "true":
                meta_bits.append("[元数据] near_boundary=true（刻意设计的近边界难例）")
            user_msg = text.strip()[:1500]
            if meta_bits:
                user_msg = "\n".join(meta_bits + ["话术文本：", user_msg])
            got = scorer.review(sysp, user_msg)
            results.append({"kind": tag, "id": r.get("id", "?"),
                            "category": r.get("category", ""),
                            "stage": r.get("stage", ""),
                            "near_boundary": r.get("near_boundary", ""),
                            "verdict": got["verdict"], "reason": got["reason"]})
            w.writerow([tag, r.get("id", "?"), r.get("category", ""),
                        r.get("stage", ""), r.get("near_boundary", ""),
                        got["verdict"], got["reason"], text.replace("\n", " ")[:500]])
            f.flush()
            dt = time.time() - t0
            print(f"  [{tag} {r.get('id','?')}] {got['verdict']} ({dt:.0f}s, {dt/max(1,len(results)):.1f}s/条)",
                  flush=True)

    # 可疑候选 = verdict != pass（从合并后的全量 CSV 读取，保证含既有 kind）
    all_rows = []
    with open(OUT_CSV, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            row.setdefault("text", "")
            all_rows.append(row)
    results = [{"kind": r["kind"], "id": r["id"], "category": r["category"],
                "stage": r["stage"], "near_boundary": r["near_boundary"],
                "verdict": r["verdict"], "reason": r["reason"]} for r in all_rows]
    sus = [x for x in results if x["verdict"] != "pass"]
    with open(OUT_SUS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "id", "category", "stage", "near_boundary",
                    "verdict", "reason"])
        for x in sus:
            w.writerow([x["kind"], x["id"], x["category"], x["stage"],
                        x["near_boundary"], x["verdict"], x["reason"]])

    from collections import Counter
    cnt = Counter((x["kind"], x["verdict"]) for x in results)
    total = len(results)
    bad = sum(1 for x in results if x["verdict"] == "drop")
    lines = [
        "# 语料抽检 LLM 预筛摘要", "",
        f"- 预筛时间：{time.strftime('%Y-%m-%d %H:%M')}（云端 {scorer.model}）",
        f"- 样本：{total} 条（正 scam + 负 benign），可疑候选 {len(sus)} 条，其中 drop 建议 {bad} 条",
        f"- 预筛 drop 率 {bad/max(1,total)*100:.1f}%（人工复核后才定论；>30% 才触发整轮重做）",
        "", "| 类型 | verdict | 条数 |", "|---|---|---|",
    ]
    for k in ("scam", "benign"):
        for v in ("pass", "fix", "drop"):
            lines.append(f"| {k} | {v} | {cnt.get((k, v), 0)} |")
    lines += [
        "", "## 人工复核指引",
        "- 打开 `review_prefilter_suspect.csv`，逐条在抽检原表 verdict 列填 pass/fix/drop；",
        "- 可疑候选之外，建议再随机抽 5~10% pass 条复核防漏；",
        "- 若 drop（含 fix 中需删者）占比 >30%，整轮语料重做。",
    ]
    Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n完成 {total} 条，可疑 {len(sus)}，drop 建议 {bad}")
    print(f"全部结果: {OUT_CSV}")
    print(f"可疑候选: {OUT_SUS}")
    print(f"摘要:     {OUT_MD}")


if __name__ == "__main__":
    main()
