#!/usr/bin/env python
"""pre_review_pass2_calib.py — pass 区校准复核（2026-09-06）

二轮"假阴猎手"误杀率高（把单段话术当逻辑断裂），本脚本用校准 prompt 对
全部 scam pass 重判一次：既保留首轮"单段独立是设计使然"的核心原则，
又叠加猎手对类别错配/AI味/硬伤的怀疑。三值 verdict。

产出：evaluation/review_pass2_calib.csv（全 pass 行，含 calib verdict）
      evaluation/review_pass2_calib_flagged.csv（非 ok 的候选）
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pre_review import CloudScorer, _strip_fence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREFILTER = ROOT / "evaluation" / "review_prefilter.csv"
OUT_ALL = ROOT / "evaluation" / "review_pass2_calib.csv"
OUT_FLAG = ROOT / "evaluation" / "review_pass2_calib_flagged.csv"

CALIB_PROMPT = """你是反诈语料质检员，做**终审复核**。下面这条话术首轮已判 pass，请你做一次平衡复核。

必须遵守的设计背景：话术库按「8 类诈骗 × 4 阶段(opening/buildup/ask/pressure)」构建，
每条只是完整诈骗链路中的一段，**单段独立是设计使然**——pressure 段可以只有威胁施压，
ask 段可以只有索要转账，不要因"缺乏完整场景铺垫"而判坏。

复核维度：
1) 是否有其标注类别的典型诈骗动作（冒充身份/索要转账验证码/高息诱导/威胁保密等）
2) 类别与阶段标注是否基本契合（内容明显属另一类别 = 需 fix；完全无关 = drop）
3) 是否像真实口语电话（AI 味/模板腔/书面堆砌 = 扣分项但单条不足以 drop）
4) 硬伤：自相矛盾、荒谬穿帮、明显语义断裂无法使用 = drop

输出 JSON：{"verdict":"ok|flag|drop","reason":"≤45字中文"}
ok=可入库；flag=有疑点建议人工再看；drop=明确该删（只删真有硬伤/类别完全错配的）。
"""


def raw_review(scorer: CloudScorer, text: str, max_retry: int = 3) -> dict:
    for attempt in range(max_retry):
        try:
            payload = json.dumps({
                "model": scorer.model, "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": CALIB_PROMPT},
                    {"role": "user", "content": text.strip()[:1500]},
                ],
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{scorer.base}/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {scorer.key}"})
            with urllib.request.urlopen(req, timeout=scorer.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(_strip_fence(data["choices"][0]["message"]["content"]))
            v = str(parsed.get("verdict", "flag")).lower()
            if v not in ("ok", "flag", "drop"):
                v = "flag"
            return {"verdict": v, "reason": str(parsed.get("reason", ""))[:60]}
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return {"verdict": "flag", "reason": "API失败"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(PREFILTER, encoding="utf-8-sig")))
    scam_pass = [r for r in rows if r["kind"] == "scam" and r["verdict"] == "pass"]
    if args.limit:
        scam_pass = scam_pass[: args.limit]
    print(f"待校准复核 scam pass: {len(scam_pass)} 条", flush=True)

    scorer = CloudScorer()
    out = []
    t0 = time.time()
    for i, r in enumerate(scam_pass):
        got = raw_review(scorer, r.get("text", "")[:1500])
        out.append({"id": r["id"], "category": r["category"], "stage": r["stage"],
                    "text": r.get("text", "")[:500],
                    "calib": got["verdict"], "reason": got["reason"]})
        print(f"  [{i+1}/{len(scam_pass)} {r['id']}] {got['verdict']} — {got['reason']}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(scam_pass)} ({time.time()-t0:.0f}s)", flush=True)

    flagged = [x for x in out if x["calib"] != "ok"]
    for name, data in ((OUT_ALL, out), (OUT_FLAG, flagged)):
        with open(name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "category", "stage", "text", "calib", "reason"])
            w.writeheader()
            w.writerows(data)
    from collections import Counter
    cnt = Counter(x["calib"] for x in out)
    print(f"\n校准复核 {len(out)} 条 → ok {cnt.get('ok', 0)} / flag {cnt.get('flag', 0)} / drop {cnt.get('drop', 0)}")
    print(f"全量 -> {OUT_ALL} | 候选 -> {OUT_FLAG}")


if __name__ == "__main__":
    main()
