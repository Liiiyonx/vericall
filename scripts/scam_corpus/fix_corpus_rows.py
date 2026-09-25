#!/usr/bin/env python
"""fix_corpus_rows.py — 语料 v0.2 fix 区改写回填（2026-09-07）

处理 corpus 中 review.verdict=='fix' 的 15 条：预筛标注为「内容错配/话题跳接」，
用云端 LLM 按标注类别+阶段改写正文 → 校验 → 回写 jsonl（text 替换，
review.verdict: fix→fixed，source=fix_rewrite，note 留改写原因）。
用法（先 source 密钥）：python -u scripts/scam_corpus/fix_corpus_rows.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "scam_corpus" / "corpus_v0.1.jsonl"

STAGE_ZH = {"opening": "开场", "buildup": "铺垫", "ask": "索要", "pressure": "施压"}
PROMPT = """你是反诈语料编辑。下面这段中文通话话术标注为「{cat}」类诈骗的「{stage}」阶段，
但内容存在错配或话题跳接（例如：客服身份却谈日结利息、体检报告后突兀推销无关产品）。
要求改写成：
1) 贴合「{cat}」话术在「{stage}」阶段应有的身份与内容（可含该类常见铺垫，如该阶段本
   就需要铺垫投资/情感/健康话题则正常保留，但身份与话术逻辑必须自洽，不许把
   客服话术硬写成婚恋、不许体检类硬接与健康无关的投资品）；
2) 中文口语、单段自然、字数 140-280 字；
3) 只输出改写后的正文，不要任何解释或引号。"""

CAT_ZH = {"elder_healthcare": "养老健康/保健品", "fake_investment": "虚假投资/荐股",
          "impersonate_authority": "冒充公检法/领导", "impersonate_family": "冒充亲友/子女",
          "lottery_prize": "中奖/兑奖", "online_loan": "网贷", "refund_cs": "退款/客服",
          "romance_pig": "杀猪盘/婚恋诱导"}


def main():
    base = os.environ["SCAM_LLM_BASE"].rstrip("/")
    key = os.environ["SCAM_LLM_KEY"]
    model = os.environ.get("SCAM_LLM_MODEL", "deepseek-chat")

    recs = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    fix_ids = [r["id"] for r in recs if r.get("review", {}).get("verdict") == "fix"]
    print(f"fix 待改写: {len(fix_ids)}")
    by_id = {r["id"]: r for r in recs}

    for k, rid in enumerate(fix_ids, 1):
        r = by_id[rid]
        cat, stage = r["category"], r.get("stage", "opening")
        sys_p = PROMPT.format(cat=CAT_ZH.get(cat, cat), stage=STAGE_ZH.get(stage, stage))
        for _ in range(3):
            try:
                payload = json.dumps({"model": model, "temperature": 0.5,
                                      "messages": [{"role": "system", "content": sys_p},
                                                   {"role": "user", "content": r["text"]}]}
                                     ).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/chat/completions", data=payload,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                new_text = data["choices"][0]["message"]["content"].strip().strip("“”\"'")
                if 60 < len(new_text) <= 800:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        else:
            print(f"  !! {rid} 改写失败，跳过（保留原 fix 标记）")
            continue
        r["text"] = new_text
        rv = r["review"]
        rv["verdict"] = "fixed"
        rv["source"] = "fix_rewrite"
        rv["note"] = (rv.get("note", "") + " | 09-07 fix_rewrite 改写回填").strip(" |")
        print(f"  [{k}/{len(fix_ids)}] {rid} [{cat}/{stage}] 改写完成 {len(new_text)}字")

    n_fixed = sum(1 for r in recs if r.get("review", {}).get("verdict") == "fixed")
    n_remain = sum(1 for r in recs if r.get("review", {}).get("verdict") == "fix")
    CORPUS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                      encoding="utf-8")
    print(f"回写完成：fixed {n_fixed} / 仍 fix {n_remain}；总条数 {len(recs)}")


if __name__ == "__main__":
    main()
