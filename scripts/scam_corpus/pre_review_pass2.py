#!/usr/bin/env python
"""pre_review_pass2.py — pass 区假阴二次复核（2026-09-06）

LLM 首轮预筛对 433 scam 判 pass 403 条。首轮 prompt 偏向"别误伤"，可能放过
类别错配/阶段错配的假阴。本脚本对 pass 中含强要钱动作词的高风险子集（约 65 条）
用"假阴猎手"反向 prompt 重审：找出其实不像该类别诈骗 / 自相矛盾 / 明显 AI 味的漏网。

用法：
  source D:/VeriCall_data/secrets/vericall_secrets.env
  python -u scripts/scam_corpus/pre_review_pass2.py [--limit N]
产出：
  evaluation/review_pass2_flagged.csv  二轮复核标记的漏网候选（人工二次确认用）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 项目根
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/scam_corpus
from pre_review import CloudScorer, SCAM_SAMPLE, _strip_fence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREFILTER = ROOT / "evaluation" / "review_prefilter.csv"
OUT_FLAG = ROOT / "evaluation" / "review_pass2_flagged.csv"

MONEY_KEYS = ["转账", "验证码", "银行卡", "卡号", "押金", "打钱", "汇款", "手续费", "账号", "转钱", "转款"]

HUNTER_PROMPT = """你是反诈语料质检员，任务是"假阴猎手"——找上一轮漏掉的坏样本。
上轮已用宽松标准判这批话术 pass；现在你反过来专挑毛病，判断它是否**不该**入库。
重点怀疑：
1) 类别/阶段错配：文本的核心诈骗动作与其标注 category 不符（如标 romance_pig 却在谈退税）
2) 自相矛盾/穿帮：数字、机构名、话术逻辑荒谬，真实电话里不会这么说
3) 明显 AI 味/模板腔：不像人打电话，像文本生成器硬凑（超长堆砌、书面腔、无口语感）
4) 诈骗性缺失：虽有"转账"字样但无真实诈骗逻辑，或标注类别与内容风马牛不相及
注意：完整诈骗链中单段独立是设计使然，不要因"缺乏完整铺垫"而误杀；你只抓上面 4 类硬伤。
输出 JSON：{"verdict":"ok|flag|drop","reason":"≤40字中文"}
ok=确实没问题可入库；flag=有疑点建议人工再看（说明疑点，不算硬伤）；drop=明显该删。
严禁输出 fix——本环节只区分 ok/flag/drop 三档。
"""


def raw_review(scorer: CloudScorer, text: str, max_retry: int = 3) -> dict:
    """直连 LLM，verdict 三值 ok/flag/drop（绕过 CloudScorer 的 pass/fix/drop 白名单）。"""
    import urllib.parse
    import json as _json
    for attempt in range(max_retry):
        try:
            payload = _json.dumps({
                "model": scorer.model, "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": HUNTER_PROMPT},
                    {"role": "user", "content": text.strip()[:1500]},
                ],
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{scorer.base}/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {scorer.key}"})
            with urllib.request.urlopen(req, timeout=scorer.timeout) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            parsed = _json.loads(_strip_fence(data["choices"][0]["message"]["content"]))
            v = str(parsed.get("verdict", "flag")).lower()
            if v not in ("ok", "flag", "drop"):
                v = "flag"
            return {"verdict": v, "reason": str(parsed.get("reason", ""))[:80]}
        except Exception as e:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return {"verdict": "flag", "reason": f"API失败"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(PREFILTER, encoding="utf-8-sig")))
    scam_pass = [r for r in rows if r["kind"] == "scam" and r["verdict"] == "pass"]
    risk = [r for r in scam_pass if any(k in r.get("text", "") for k in MONEY_KEYS)]
    if args.limit:
        risk = risk[: args.limit]
    print(f"pass 高风险子集 {len(risk)} 条（含强要钱动作词）", flush=True)

    scorer = CloudScorer()
    flagged = []
    t0 = time.time()
    for i, r in enumerate(risk):
        got = raw_review(scorer, r.get("text", "")[:1500])
        print(f"  [{i+1}/{len(risk)} {r['id']}] {got['verdict']} — {got['reason']}", flush=True)
        if got["verdict"] != "ok":
            flagged.append({"id": r["id"], "category": r["category"], "stage": r["stage"],
                            "text": r.get("text", "")[:500], "pass2": got["verdict"],
                            "reason": got["reason"]})

    with open(OUT_FLAG, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "stage", "text", "pass2", "reason"])
        w.writeheader()
        w.writerows(flagged)
    n_drop = sum(1 for r in flagged if r["pass2"] == "drop")
    n_flag = sum(1 for r in flagged if r["pass2"] == "flag")
    print(f"\n二轮复核 {len(risk)} 条 → 疑点/硬伤 {len(flagged)} 条（其中 drop 硬伤 {n_drop}，flag 疑点 {n_flag}）")
    print(f"漏网候选 -> {OUT_FLAG}")


if __name__ == "__main__":
    main()
