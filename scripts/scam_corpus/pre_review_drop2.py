#!/usr/bin/env python
"""pre_review_drop2.py — drop 区二审确认（2026-09-07 新增）

背景：v0.2 全量预筛发现一审对「铺垫类话术」（opening/buildup 无直接索要动作）
存在系统性误杀——88 条 auto-drop 中 77% 是 opening/buildup 段，抽查显示杀猪盘
"项目邀约"、冒充子女"手机被收"等典型合格话术被误判 drop。

本脚本对一审判 drop 的全部候选做**独立二审**：只传文本+类别+阶段标注（不给一审
verdict/reason，防锚定），用保守口径复核——只有「类别完全无关 / 自相矛盾穿帮 /
明显语义断裂 / 垃圾文本」才允许 drop。两轮都 drop 才视为高置信可删。

用法（先 source 密钥）：
  python -u scripts/scam_corpus/pre_review_drop2.py [--limit N]
输入：evaluation/review_prefilter_suspect.csv（一审 verdict=drop 的行）
产出：evaluation/review_drop2_confirm.csv（id, category, stage, text, confirm, reason）
      confirm ∈ {drop, keep}；keep=一审误杀，应保留
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pre_review import CloudScorer, _strip_fence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUS = ROOT / "evaluation" / "review_prefilter_suspect.csv"
PREFILTER = ROOT / "evaluation" / "review_prefilter.csv"
OUT = ROOT / "evaluation" / "review_drop2_confirm.csv"

DROP2_PROMPT = """你是反诈语料质检终审员。下面这条话术是「8 类诈骗 × 4 阶段(opening开场/buildup铺垫/ask索要/pressure施压)」语料库中的一段。
必须遵守：每条只是完整诈骗链路中的**一段**，单段独立是设计使然——opening 可以只做人设/情境铺设，buildup 可以只做关系升温/项目铺垫，pressure 可以只有威胁施压，**不要因"未直接索要钱财/缺乏完整铺垫"而判删**。
你只做一件事：判断这条是否**确实该删**。仅以下 4 类硬伤才允许 drop：
1) 类别完全错配：内容核心与其标注类别风马牛不相及（如标 romance_pig 却在谈退税，且无该类常见铺垫要素）
2) 自相矛盾/荒谬穿帮：数字、机构、话术逻辑在真实电话里不可能成立
3) 语义断裂无法使用：前言不搭后语、明显是误生成的乱码文本
4) 垃圾/AI 硬凑：纯书面腔堆砌、超长绕口无口语感、无法用于任何类别训练
只要内容与标注类别沾边、作为该类某阶段话术可成立（哪怕是铺垫或氛围营造），就应 keep。
输出 JSON：{"verdict":"drop|keep","reason":"≤40字中文"}
"""


def raw_review(scorer: CloudScorer, text: str, max_retry: int = 3) -> dict:
    for attempt in range(max_retry):
        try:
            payload = json.dumps({
                "model": scorer.model, "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": DROP2_PROMPT},
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
            v = str(parsed.get("verdict", "keep")).lower()
            if v not in ("drop", "keep"):
                v = "keep"
            return {"verdict": v, "reason": str(parsed.get("reason", ""))[:60]}
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return {"verdict": "keep", "reason": "API失败(保守保留)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help=">0 仅跑前 N 条(冒烟)")
    args = ap.parse_args()

    sus = list(csv.DictReader(open(SUS, encoding="utf-8-sig")))
    text_of = {r["id"]: r.get("text", "") for r in
               csv.DictReader(open(PREFILTER, encoding="utf-8-sig"))}
    drops = [r for r in sus if r["verdict"] == "drop"]
    if args.limit:
        drops = drops[: args.limit]
    print(f"待二审 drop 候选: {len(drops)} 条（kind={drops[0]['kind'] if drops else '?'}）", flush=True)

    scorer = CloudScorer()
    out, t0 = [], time.time()
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "stage", "text", "confirm", "reason"])
        for i, r in enumerate(drops):
            txt = text_of.get(r["id"], "")
            got = raw_review(scorer, txt)
            out.append(got)
            w.writerow([r["id"], r.get("category", ""), r.get("stage", ""),
                        txt.replace("\n", " ")[:300], got["verdict"], got["reason"]])
            f.flush()
            print(f"  [{i+1}/{len(drops)} {r['id']}] {got['verdict']} — {got['reason']}", flush=True)

    from collections import Counter
    cnt = Counter(x["verdict"] for x in out)
    print(f"\n二审完成 {len(out)} 条: {dict(cnt)}")
    print(f"确认可删 drop: {cnt['drop']} / 误杀保留 keep: {cnt['keep']}")
    print(f"产物: {OUT}")


if __name__ == "__main__":
    main()
